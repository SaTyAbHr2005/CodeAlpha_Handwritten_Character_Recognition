# Enhanced CNN Character Recognition

This project implements an **Enhanced Convolutional Neural Network (CNN)** for recognizing **handwritten digits (0–9)** and **uppercase letters (A–Z)**.  
It combines the MNIST dataset for digits with synthetically generated letter patterns, enhanced through data augmentation techniques such as rotation, scaling, noise addition, and dilation.  

## ✨ Features
- Supports two modes:
  - **Digits only (0–9)**
  - **Digits + Letters (0–9 + A–Z)**
- Enhanced synthetic data generation for letters with realistic variations
- Deep CNN architecture with multiple convolutional, batch normalization, and dropout layers
- Early stopping and learning rate scheduling
- Saves and reloads trained models
- Image preprocessing and prediction with confidence scores
- Visual display of predictions with top-k results

## 🧠 Model Architecture
- **Conv2D** layers (64 → 128 → 256 filters)
- **Batch Normalization** and **Dropout**
- **Dense layers**: 1024 → 512 → num_classes
- **Softmax output** for classification

## 📂 Dataset
- **Digits**: MNIST dataset (28x28 grayscale images)
- **Letters**: Synthetic patterns (A–Z) with augmentation

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/enhanced-cnn-character-recognition.git
   cd enhanced-cnn-character-recognition

2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python pillow


3. Run the script:
   ```bash
   python task3.py


4. Choose an option:

- 1 → Train a new model

- 2 → Test an existing model with custom images

- 3 → Exit

Training

- Trains on MNIST + synthetic letters

- Saves the model as:

  ```bash
  enhanced_digits_letters_cnn_model.h5
  enhanced_digits_letters_cnn_model_config.npy

Testing

- Loads the saved model

- Prompts for an image path

- Predicts top-k character classes with confidence scores

- Displays:

   - Original image

   - Processed input (28x28)

   - Prediction results with probabilities

## 📊 Example Output

- Classification accuracy on test set

- Prediction window showing:

   - Original vs preprocessed image

   - Predicted character with confidence

   - Top-5 predictions

## 🔮 Future Improvements

- Train with EMNIST dataset for more realistic letter samples

- Add lowercase characters

- Support cursive/handwritten styles

- Deploy with a simple web app (Flask/Streamlit)
