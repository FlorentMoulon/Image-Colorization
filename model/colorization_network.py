# models/colorization_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout_prob=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                     stride=1, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )
        
    def forward(self, x):
        return self.conv(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, 
                              stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )
        
    def forward(self, x):
        return self.upconv(x)

class ColorizationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder with progressive dropout
        self.encoder = nn.Sequential(            
            ConvBlock(1, 64, dropout_prob=0.05),
            ConvBlock(64, 128, stride=2, dropout_prob=0.1),
            ConvBlock(128, 256, stride=2, dropout_prob=0.2),
            ConvBlock(256, 512, dilation=2, dropout_prob=0.25),
            ConvBlock(512, 512, dilation=4, dropout_prob=0.3),
        )
        
        # Decoder with decreasing dropout
        self.decoder = nn.Sequential(
            UpConvBlock(512, 256, dropout_prob=0.2),
            ConvBlock(256, 256, dropout_prob=0.15),
            UpConvBlock(256, 128, dropout_prob=0.1),
            ConvBlock(128, 128, dropout_prob=0.05),
            nn.Conv2d(128, 313, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ColorLoss(torch.nn.Module):
    def __init__(self, weight, lambda_cross = 1, lambda_grad=1, lambda_foc = 0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.register_buffer('weight', weight)
        self.lambda_cross = lambda_cross # Balancing factor for cross-entropy loss
        self.lambda_grad = lambda_grad
        self.lambda_foc = lambda_foc  # Balancing factor for focal loss
        self.alpha = alpha  # Balancing factor for focal loss
        self.gamma = gamma  # Focusing parameter for focal loss
        
    def compute_cross_entropy(self, pred, target):
        ce_per_pixel = -torch.sum(target * torch.log(pred + 1e-8), dim=1)
        q_star = torch.argmax(target, dim=1)
        weights = self.weight[q_star] 
        return (ce_per_pixel * weights).mean()

    def compute_gradients(self, tensor):
        # Horizontal gradients
        grad_h = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        # Vertical gradients
        grad_v = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        return grad_h, grad_v
    
    def focal_loss(self, pred, target):
        # Ensure numerical stability
        pred = torch.clamp(pred, min=1e-8, max=1.0)
        
        # Compute p_c (probabilities of the true class)
        p_t = torch.sum(target * pred, dim=1)  # Shape: (N, H, W)

        # Compute the focal loss
        focal_loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)
        
        # Apply class weights
        q_star = torch.argmax(target, dim=1)
        weights = self.weight[q_star]
        
        # Weighted loss
        weighted_loss = focal_loss * weights

        return weighted_loss.mean()
    
    def forward(self, pred, target):

        if(self.lambda_cross > 0):
            l_hist = self.compute_cross_entropy(pred, target)
        else:
            l_hist = 0
        
        if(self.lambda_grad > 0):     
            # Compute gradients for pred and target for L_grad
            grad_h_pred, grad_v_pred = self.compute_gradients(pred)
            grad_h_target, grad_v_target = self.compute_gradients(target)
            l_grad_h = torch.norm(grad_h_pred - grad_h_target, p=2, dim=1).pow(2).mean()
            l_grad_v = torch.norm(grad_v_pred - grad_v_target, p=2, dim=1).pow(2).mean()
            l_grad = l_grad_h + l_grad_v
        else:
            l_grad = 0
        
        if(self.lambda_foc > 0):
            l_foc = self.focal_loss(pred, target)
        else:
            l_foc = 0

        # Combine terms
        loss_function = (self.lambda_cross * l_hist) + (self.lambda_grad * l_grad * 1e-3) + (self.lambda_foc * l_foc)
        return loss_function

class ColorBalancer:
    def __init__(self, bin_centers, T=0.38):
        self.bin_centers = torch.tensor(bin_centers, dtype=torch.float32)
        self.T = T
        
    def __call__(self, z):
        logits = torch.log(z + 1e-8) / self.T
        tempered = F.softmax(logits, dim=1)
        ab = torch.einsum('bchw,cd->bdhw', tempered, self.bin_centers.to(z.device))
        return ab