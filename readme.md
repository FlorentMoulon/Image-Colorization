# Image Colorization using Deep Learning
*Boissot Aurélien*, 
*Moulon Florent*, 
*Skowronek Arthur*


## Overview
This project implements a deep learning-based approach for colorizing grayscale images using a neural network. The model is trained on the LAB color space, where the L-channel represents grayscale intensity, and the A/B channels store color information. The goal is to predict the A/B color channels given a grayscale image.


## Acknowledgments
This project is inspired by the approache presented by Richard Zhang, Phillip Isola, and Alexei A. Efros. in Colorful image colorization. (arXiv, 1603.08511, 2016.)


## Project Structure
```
├── model
│   ├── colorization_network.py       # Defines the colorization neural network
│   ├── empirical_prob.npy            # Precomputed empirical color distribution
│   ├── prior_probs.npy               # Prior probability distribution
│   ├── pts_in_hull.npy               # Cluster centers for quantized colors
├── utils
│   ├── empirical_color_distributon.py # Computes empirical color distributions
├── dataloader.py                      # Defines the data pipeline for training
├── main.py                            # Main script for training and testing models
```

## Requirements
To run this project, install the required dependencies:
```sh
pip install -r requirements.txt
```

## Model Architecture
The model consists of an encoder-decoder structure with convolutional and transposed convolutional layers:
- **Encoder:** Extracts features from grayscale images.
- **Decoder:** Predicts color information from extracted features.
- **Color Loss:** Uses a weighted cross-entropy loss to handle color imbalances.

For information about our method, refer to the pdf report.


## Training
To train the model, modify `main.py` and set `mode = 'train'`. Then run:
```sh
python main.py
```

Training involves:
- Loading grayscale and color images.
- Converting to LAB color space.
- Predicting A/B channels and computing loss.
- Saving model checkpoints and loss logs.

## Testing
To test the model on grayscale images, set `mode = 'test'` in `main.py` and execute:
```sh
python main.py
```
This will:
- Load a pretrained model.
- Convert grayscale images to LAB format.
- Predict A/B color channels and reconstruct RGB images.
- Save colorized images in the output directory.

## Pretrained Weights
If you have pretrained weights, you can load them by specifying the `checkpoint_path` in `main.py` before training or testing.

