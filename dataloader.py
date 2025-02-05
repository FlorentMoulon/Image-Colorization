import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import color
from sklearn.neighbors import KDTree

class CustomColorDataset(Dataset):
    bin_centers = None
    kdtree = None
    
    @classmethod
    def generate_bins(cls):
        # Load precomputed 313 AB bins from official resources
        return np.load('./model/pts_in_hull.npy')[:, :2]  # Shape: (313, 2)
    
    def __init__(self, gt_folder, grayscale_folder, split='train', transform=None):
        self.gt_folder = gt_folder
        self.grayscale_folder = os.path.join(grayscale_folder, split)
        self.split = split
        self.transform = transform or transforms.Resize((100, 100))
        
        if CustomColorDataset.bin_centers is None:
            CustomColorDataset.bin_centers = self.generate_bins()
            CustomColorDataset.kdtree = KDTree(CustomColorDataset.bin_centers)
            
        self.files = os.listdir(self.grayscale_folder)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        gray = Image.open(os.path.join(self.grayscale_folder, file)).convert('L')
        rgb = Image.open(os.path.join(self.gt_folder, file)).convert('RGB')
        
        if self.transform:
            gray = self.transform(gray)
            rgb = self.transform(rgb)
            
        # Convert RGB to Lab and extract AB channels
        lab = color.rgb2lab(np.array(rgb)/255.0)
        ab = lab[:, :, 1:].reshape(-1, 2)
        
        # Find nearest bins using precomputed 313 bins
        dists, indices = self.kdtree.query(ab, k=5)
        weights = np.exp(-dists**2 / (2*5**2))
        weights /= weights.sum(axis=1, keepdims=True)
        
        # Create target tensor with exactly 313 classes
        target = np.zeros((ab.shape[0], 313))
        target[np.arange(ab.shape[0])[:, None], indices] = weights
        target = target.reshape(lab.shape[0], lab.shape[1], 313)
        
        return (transforms.functional.to_tensor(gray), 
                torch.tensor(target, dtype=torch.float32).permute(2, 0, 1))
        
        
def get_dataloader(gt_folder, grayscale_folder, batch_size=32, split='train'):
    dataset = CustomColorDataset(gt_folder, grayscale_folder, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))