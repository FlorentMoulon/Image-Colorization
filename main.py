import torch
from model.colorization_network import ColorizationNetwork, ColorLoss, ColorBalancer
from dataloader import get_dataloader
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
from skimage import color
import datetime

# Saves the weights of the model
def save_model(model, folder_path, model_name, nb_epoch=0, add_timestamp=True):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        model_name = timestamp + ' ' + model_name

    if nb_epoch > 0:
        model_name += f'_{nb_epoch}-epochs'
    
    torch.save(model, os.path.join(folder_path, model_name + '.pth'))


# Show loss graph and loss_val graph
def plot_loss(table_loss, table_loss_val):
    plt.plot(range(1, len(table_loss_val) + 1), table_loss_val, label='val')
    plt.plot(np.linspace(1, len(table_loss_val), len(table_loss)), table_loss, label='train')
    plt.legend()
    plt.xticks(range(1, len(table_loss_val) + 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def create_log_file(save_model_folder_path):
    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    loss_filename = os.path.join(save_model_folder_path, f"training_losses_{timestamp}.csv")
    
    # Create and write header to loss file
    with open(loss_filename, 'w') as f:
        f.write("Epoch,Training Loss,Validation Loss\n")
    return loss_filename

def log_loss(loss_filename, epoch, avg_loss, avg_val_loss=None):
    with open(loss_filename, 'a') as f:
        if avg_val_loss:
            f.write(f"{epoch},{avg_loss:.4f},{avg_val_loss:.4f}\n")
        else:
            f.write(f"{epoch},{avg_loss:.4f},N/A\n")


# Train the colorization model
def train_colorization_model(train_loader, val_loader=None, epochs=50, patience=4, min_delta=0.001, 
                             save_interval=5, save_model_folder_path='./weights/', checkpoint_path=None, model_name="default_name_model"):
    best_model_name = '_best-model'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ColorizationNetwork().to(device)
    
    loss_filename = create_log_file(save_model_folder_path)
    
    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'[1] - Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1
        epochs_no_improve = checkpoint['epochs_no_improve']
    else:
        print('[1] - No checkpoint found, starting from scratch')
        best_val_loss = float('inf')
        start_epoch = 0
        epochs_no_improve = 0
    
    #  Initialize weights and optimizer
    print('[2] - Loading weights')
    init_type = 'empirical' # 'empirical' or 'ones'
    if(init_type == 'empirical'):
        empirical_prob = np.load('./model/empirical_prob.npy')
        lambda_param = 0.5
        Q = 313

        # Compute smoothed distribution
        smoothed_prob = (1 - lambda_param) * empirical_prob + lambda_param/Q

        # Compute unnormalized weights (inverse of smoothed distribution)
        weights = 1 / (smoothed_prob + 1e-8)  # Add epsilon to avoid division by zero

        # Normalize weights to satisfy E[w] = 1
        normalization_factor = np.sum(empirical_prob * weights)
        weights_normalized = weights / normalization_factor

        # Convert to tensor and send to device
        weights = torch.from_numpy(weights_normalized).float().to(device)
    else:
        weights = torch.ones(313).to(device)
        
    criterion = ColorLoss(weights, 1, 0, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Load optimizer state if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Early stopping variable
    early_stop = False
        
    table_loss = []
    table_loss_val = []
    
    print('[3] - Training model')
    for epoch in range(start_epoch, epochs):
        if early_stop:
            print(f'\nEarly stopping triggered after {epoch} epochs!')
            break
            
        start = time.time()
        print(f'\nEpoch {epoch+1}/{epochs}')
        model.train()
        total_loss = 0.0
        processed_images = 0
        
        # Training phase
        for batch_idx, (gray, target) in enumerate(train_loader):
            batch_size = gray.size(0)
            gray = gray.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            pred = model(gray)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            processed_images += batch_size
            
            # Print batch progress
            print(f'\rTrain: [{processed_images}/{len(train_loader.dataset)}] '
                  f'({100. * batch_idx / len(train_loader):.0f}%) | '
                  f'Loss: {loss.item():.4f} | '
                  f'Avg Loss: {total_loss/(batch_idx+1):.4f}'
                  f' | Time: {time.time()-start:.2f} sec', end='')
            
        # Epoch summary
        avg_loss = total_loss / len(train_loader)
        table_loss.append(avg_loss)
        print(f'\nEpoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}')
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                print('\nValidation:')
                for batch_idx, (gray, target) in enumerate(val_loader):
                    batch_size = gray.size(0)
                    gray = gray.to(device)
                    target = target.to(device)
                    
                    pred = model(gray)
                    loss = criterion(pred, target)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    pred_classes = torch.argmax(pred, dim=1)
                    target_classes = torch.argmax(target, dim=1)
                    correct += (pred_classes == target_classes).sum().item()
                    total += target.numel()
                    
                    # Print validation progress
                    print(f'\rVal: [{batch_idx+1}/{len(val_loader)}] '
                          f'Loss: {loss.item():.4f} | '
                          f'Acc: {100.*correct/total:.2f}%', end='')
            
            # Validation summary
            avg_val_loss = val_loss / len(val_loader)
            table_loss_val.append(avg_val_loss)
            val_acc = 100. * correct / total
            print(f'\nValidation complete | Avg Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%')

            # Early stopping check
            if (best_val_loss - avg_val_loss) > min_delta:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Save best model
                save_model(model.state_dict(), save_model_folder_path, model_name + best_model_name, nb_epoch=epoch+1)
                print(f'Validation loss improved. Saving best model.')
            else:
                epochs_no_improve += 1
                print(f'Validation loss did not improve. Patience: {epochs_no_improve}/{patience}')
                if epochs_no_improve >= patience:
                    early_stop = True
            # Saving model regardless of improvement at specific interval
            if((epoch+1) % save_interval == 0 and epoch != 0):
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve
                }
                save_model(checkpoint, save_model_folder_path, model_name+'_checkpoint', nb_epoch=epoch+1)
                print(f'Model saved at {epoch+1} epochs')
                
        # Write to loss file
        log_loss(loss_filename, epoch+1, avg_loss, avg_val_loss if val_loader else None)
                
    # Save final model if early stopping didn't occur
    if not early_stop:
        save_model(model.state_dict(), save_model_folder_path, model_name, nb_epoch=epochs)
    print('\nTraining complete. Model saved.')
    

    plot_loss(table_loss, table_loss_val)



"""
Test model
"""

# Load the color prior
def load_color_prior():
    color_prior = np.load('./model/pts_in_hull.npy')  # Shape: (313, 3)
    return color_prior

def load_model_weights(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Load from checkpoint (new format)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # Load from standalone model file (old format)
        model.load_state_dict(checkpoint)
        print("Loaded model weights from standalone file")
    
    model.eval()
    return model

# Colorize a single image using a pre-trained model
def colorize_image(image_path, model, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    gray_image = transform(image).unsqueeze(0).to(device)  # (1, 1, H, W)
    
    # Perform inference
    with torch.no_grad():
        output = model(gray_image)  # Shape: (1, 313, H, W)
    
    # Annealed mean decoding
    # Load color prior (ab bins)
    color_prior = np.load('./model/pts_in_hull.npy')[:, :2]  # Shape: (313, 2)
    color_balancer = ColorBalancer(bin_centers=color_prior, T=0.38)#.to(device)
    
    # Convert model output to probabilities and apply annealing
    ab_channels = color_balancer(output)  # Shape: (1, 2, H, W)
    
    # Get L channel and combine with ab
    L_channel = gray_image * 100  # Already in [0, 100] range
    Lab = torch.cat([L_channel, ab_channels], dim=1)  # (1, 3, H, W)
    
    # Convert to numpy and process
    Lab_np = Lab.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    
    # Convert Lab to RGB
    rgb = color.lab2rgb(Lab_np) * 255
    rgb = rgb.astype(np.uint8)
    
    # Convert to PIL image and save/display
    colorized_image = Image.fromarray(rgb)
    colorized_image = colorized_image.resize(image.size) # Resize to original dimensions
    
    return colorized_image
    
# Colorize all images in a folder with a pre-trained model
def colorize_images(input_folder_path, output_folder_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = load_model_weights(ColorizationNetwork().to(device), model_path, device)
    print("Model loaded")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Colorize each image in the input folder
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_folder_path, filename)
            colorized_image = colorize_image(image_path, model, device)
            colorized_image.save(os.path.join(output_folder_path, filename))
            print(f'Colorized image saved at {os.path.join(output_folder_path, filename)}')

# Test the colorization models on a folder of grayscale images
def test_models(models_folder_path, input_images_folder_path, output_images_folder_path):
    if not os.path.exists(output_images_folder_path):
        os.makedirs(output_images_folder_path)

    for model_filename in os.listdir(models_folder_path):
        if model_filename.endswith(".pth"):
            print(f'\nColorizing images using model {model_filename}...\n\n')

            # Load the model
            model_path = os.path.join(models_folder_path, model_filename)

            # Create output folder for the colorized images
            output_folder_path = os.path.join(output_images_folder_path, model_filename)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            colorize_images(input_images_folder_path, output_folder_path, model_path)
            print(f'Images colorized using model {model_filename} saved at {output_folder_path} \n\n')


def main():
    mode = 'test' # 'train' or 'test'

    if(mode == 'train'):
        ground_truth_images_folder_path = './DatasetVegetableFruitV2/GroundTruth/'
        grayscale_images_folder_path = './DatasetVegetableFruitV2/Grayscale/'
        save_model_folder_path = './weights/'
        model_name = "low_dropout" # without '.pth'
        
        print('[0] - Loading data...')

        train_loader = get_dataloader(ground_truth_images_folder_path, grayscale_images_folder_path, split='train')
        val_loader = get_dataloader(ground_truth_images_folder_path, grayscale_images_folder_path, split='val')
        
        print('[1] - Data loaded')
        train_colorization_model(train_loader, val_loader, epochs=80, checkpoint_path='./weights/20250128_1914 low_dropout_checkpoint_70-epochs.pth', model_name=model_name, save_model_folder_path=save_model_folder_path)
    
    elif(mode == 'test'):
        input_images_folder_path = './DatasetVegetableFruitV2/Grayscale/Test/'
        output_images_folder_path = './TestOutput/'

        models_folder_path = './weights/'
        model_name = '20250128_1914 low_dropout_70-epochs' # if test_mode == 'single' # without '.pth'

        test_mode = 'single' # 'single' or 'multiple'

        if(test_mode == 'single'): 
            model_path = os.path.join(models_folder_path, model_name+'.pth')
            output_images_folder_path = os.path.join(output_images_folder_path, model_name)
            colorize_images(input_images_folder_path, output_images_folder_path, model_path)
        else:
            test_models(models_folder_path, input_images_folder_path, output_images_folder_path)


if __name__ == "__main__":
    main()