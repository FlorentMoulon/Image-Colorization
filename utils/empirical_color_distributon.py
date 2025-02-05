import numpy as np
from scipy.spatial import KDTree
import os
from PIL import Image
from skimage import color
from tqdm import tqdm

# Load quantized ab bins (313 color clusters)
ab_bins = np.load('./model/pts_in_hull.npy')[:, :2]  # Shape: (313, 2)

# Create KDTree for fast nearest neighbor search
kdtree = KDTree(ab_bins)

# Initialize histogram
histogram = np.zeros(len(ab_bins), dtype=np.int64)

# Path to your training images
train_image_dir = "./DatasetVegetableFruit/GroundTruth/"

i=0
# Process all training images
for img_name in tqdm(os.listdir(train_image_dir), desc="Processing images"):
    img_path = os.path.join(train_image_dir, img_name)
    
    # Load image and convert to Lab
    img = np.array(Image.open(img_path).convert("RGB"))
    lab = color.rgb2lab(img)  # Convert to Lab
    
    # Extract ab channels and reshape to (H*W, 2)
    ab = lab[:, :, 1:].reshape(-1, 2)
    
    # Find nearest bin indices for all pixels
    _, indices = kdtree.query(ab, k=1)
    
    # Update histogram
    unique, counts = np.unique(indices, return_counts=True)
    histogram[unique] += counts
    
    i+=1
    
    #print(f'Image {i}/2498 has been processed')

# Convert to probability distribution
total_pixels = histogram.sum()
empirical_prob = histogram / total_pixels

# Save for future use
np.save('./model/empirical_prob.npy', empirical_prob)
print("Empirical probability distribution saved!")