import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image


class Enhanced_CNN_CharacterRecognizer:
    def __init__(self, mode='digits_and_letters'):
        self.mode = mode
        if mode == 'digits_only':
            self.num_classes = 10
            self.class_names = [str(i) for i in range(10)]
        else:
            self.num_classes = 36
            self.class_names = self._get_extended_class_names()
        self.model = None
        self.input_shape = (28, 28, 1)
        self.is_trained = False
    
    def _get_extended_class_names(self):
        classes = []
        for i in range(10):
            classes.append(str(i))
        for i in range(26):
            classes.append(chr(65 + i))
        return classes
    
    def _create_realistic_letter_data(self):
        letters_x_train = []
        letters_y_train = []
        letters_x_test = []
        letters_y_test = []
        
        for letter_idx in range(26):
            letter_class = 10 + letter_idx
            
            for i in range(400):
                pattern = self._create_letter_pattern(letter_idx)
                pattern = self._add_variations(pattern)
                letters_x_train.append(pattern)
                letters_y_train.append(letter_class)
            
            for i in range(80):
                pattern = self._create_letter_pattern(letter_idx)
                pattern = self._add_variations(pattern, test_mode=True)
                letters_x_test.append(pattern)
                letters_y_test.append(letter_class)
        
        return (np.array(letters_x_train), np.array(letters_y_train), 
                np.array(letters_x_test), np.array(letters_y_test))
    
    def _create_letter_pattern(self, letter_idx):
        pattern = np.zeros((28, 28))
        letter = chr(65 + letter_idx)
        
        if letter == 'A':
            for row in range(6, 22):
                width = max(0, (row - 6) // 2)
                left = max(0, 14 - width)
                right = min(28, 14 + width + 1)
                if left < right:
                    pattern[row, left:right] = 0.8
            pattern[14, 10:18] = 0.9
            
        elif letter == 'B':
            pattern[6:22, 8] = 0.8
            pattern[6:14, 8:16] = 0.6
            pattern[14:22, 8:16] = 0.6
            pattern[6, 8:16] = 0.8
            pattern[13, 8:16] = 0.8
            pattern[21, 8:16] = 0.8
            
        elif letter == 'C':
            pattern[8:20, 10] = 0.8
            pattern[8, 10:18] = 0.8
            pattern[19, 10:18] = 0.8
            
        elif letter == 'D':
            pattern[6:22, 8] = 0.8
            pattern[6:22, 8:16] = 0.6
            pattern[6, 8:16] = 0.8
            pattern[21, 8:16] = 0.8
            
        elif letter == 'E':
            pattern[6:22, 8] = 0.8
            pattern[6, 8:18] = 0.8
            pattern[13, 8:15] = 0.8
            pattern[21, 8:18] = 0.8
            
        elif letter == 'F':
            pattern[6:22, 8] = 0.8
            pattern[6, 8:18] = 0.8
            pattern[13, 8:15] = 0.8
            
        elif letter == 'H':
            pattern[6:22, 8] = 0.8
            pattern[6:22, 18] = 0.8
            pattern[14, 8:19] = 0.8
            
        elif letter == 'I':
            pattern[6, 10:18] = 0.8
            pattern[7:21, 14] = 0.8
            pattern[21, 10:18] = 0.8
            
        elif letter == 'L':
            pattern[6:22, 8] = 0.8
            pattern[21, 8:18] = 0.8
            
        elif letter == 'O':
            center = (14, 14)
            for i in range(28):
                for j in range(28):
                    dist = ((i-center[0])**2 + (j-center[1])**2)**0.5
                    if 6 <= dist <= 8:
                        pattern[i, j] = 0.8
                        
        elif letter == 'T':
            pattern[6, 8:20] = 0.8
            pattern[7:22, 14] = 0.8
            
        else:
            pattern[8:20, 10:18] = np.random.rand(12, 8) * 0.6
            pattern[10:18, 12] = 0.8
            pattern[14, 10:18] = 0.7
        
        return pattern
    
    def _add_variations(self, pattern, test_mode=False):
        angle = np.random.uniform(-8, 8)
        h, w = pattern.shape
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        pattern = cv2.warpAffine(pattern, M, (w, h))
        
        if not test_mode:
            scale = np.random.uniform(0.85, 1.15)
            pattern = cv2.resize(pattern, None, fx=scale, fy=scale)
            pattern = cv2.resize(pattern, (28, 28))
        
        noise = np.random.normal(0, 0.03, pattern.shape)
        pattern = np.clip(pattern + noise, 0, 1)
        
        if np.random.random() > 0.6:
            kernel = np.ones((2,2), np.uint8)
            pattern = cv2.dilate(pattern, kernel, iterations=1)
        
        return pattern
    
    def load_complete_dataset(self):
        (x_train_digits, y_train_digits), (x_test_digits, y_test_digits) = mnist.load_data()
        
        x_train_digits = x_train_digits.astype('float32') / 255.0
        x_test_digits = x_test_digits.astype('float32') / 255.0
        
        if self.mode == 'digits_only':
            x_train_digits = np.expand_dims(x_train_digits, -1)
            x_test_digits = np.expand_dims(x_test_digits, -1)
            
            y_train_digits = tf.keras.utils.to_categorical(y_train_digits, self.num_classes)
            y_test_digits = tf.keras.utils.to_categorical(y_test_digits, self.num_classes)
            
            return (x_train_digits, y_train_digits), (x_test_digits, y_test_digits)
        
        else:
            x_train_letters, y_train_letters, x_test_letters, y_test_letters = self._create_realistic_letter_data()
            
            x_train_combined = np.concatenate([x_train_digits, x_train_letters])
            y_train_combined = np.concatenate([y_train_digits, y_train_letters])
            x_test_combined = np.concatenate([x_test_digits, x_test_letters])
            y_test_combined = np.concatenate([y_test_digits, y_test_letters])
            
            x_train_combined = np.expand_dims(x_train_combined, -1)
            x_test_combined = np.expand_dims(x_test_combined, -1)
            
            y_train_combined = tf.keras.utils.to_categorical(y_train_combined, self.num_classes)
            y_test_combined = tf.keras.utils.to_categorical(y_test_combined, self.num_classes)
            
            return (x_train_combined, y_train_combined), (x_test_combined, y_test_combined)
    
    def build_enhanced_cnn_model(self):
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        model.summary()
        
        return model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=25):
        callbacks = [
            EarlyStopping(
                patience=8, 
                restore_best_weights=True, 
                monitor='val_accuracy',
                verbose=1
            ),
            ReduceLROnPlateau(
                factor=0.2, 
                patience=5, 
                monitor='val_loss',
                verbose=1,
                min_lr=1e-7
            )
        ]
        
        history = self.model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        return history
    
    def evaluate_model(self, x_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return test_accuracy
    
    def save_model(self, filepath='enhanced_cnn_model.h5'):
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        self.model.save(filepath)
        
        config = {
            'mode': self.mode,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'input_shape': self.input_shape
        }
        np.save(filepath.replace('.h5', '_config.npy'), config)
    
    def load_model(self, filepath='enhanced_cnn_model.h5'):
        try:
            self.model = tf.keras.models.load_model(filepath)
            
            config = np.load(filepath.replace('.h5', '_config.npy'), allow_pickle=True).item()
            self.mode = config['mode']
            self.num_classes = config['num_classes']
            self.class_names = config['class_names']
            self.input_shape = config['input_shape']
            self.is_trained = True
            
            return True
        except Exception as e:
            return False
    
    def preprocess_input_image(self, image_path):
        try:
            if isinstance(image_path, str):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    img = np.array(Image.open(image_path).convert('L'))
            else:
                img = image_path
            
            img = cv2.resize(img, (28, 28))
            
            if np.mean(img) > 127:
                img = 255 - img
            
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=(0, -1))
            
            return img
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def predict_character(self, image_path, top_k=5):
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        processed_image = self.preprocess_input_image(image_path)
        predictions = self.model.predict(processed_image, verbose=0)
        
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.class_names):
                character = self.class_names[idx]
                confidence = predictions[0][idx]
                results.append((character, confidence))
        
        return results
    
    def display_prediction_results(self, image_path, predictions):
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            original_img = np.array(Image.open(image_path).convert('L'))
        
        processed_img = cv2.resize(original_img, (28, 28))
        if np.mean(processed_img) > 127:
            processed_img = 255 - processed_img
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(processed_img, cmap='gray')
        axes[1].set_title('Enhanced CNN Input', fontsize=14)
        axes[1].axis('off')
        
        axes[2].axis('off')
        result_text = f"Enhanced Prediction: '{predictions[0][0]}'\n"
        result_text += f"Confidence: {predictions[0][1]:.3f} ({predictions[0][1]*100:.1f}%)\n\n"
        result_text += "Top Predictions:\n"
        
        for i, (char, conf) in enumerate(predictions):
            result_text += f"{i+1}. '{char}': {conf:.3f} ({conf*100:.1f}%)\n"
        
        axes[2].text(0.1, 0.5, result_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        axes[2].set_title('Enhanced CNN Results', fontsize=14)
        
        plt.tight_layout()
        plt.show()


def train_enhanced_digits_and_letters():
    recognizer = Enhanced_CNN_CharacterRecognizer('digits_and_letters')
    
    (x_train, y_train), (x_test, y_test) = recognizer.load_complete_dataset()
    recognizer.build_enhanced_cnn_model()
    
    history = recognizer.train_model(x_train, y_train, x_test, y_test, epochs=25)
    
    recognizer.evaluate_model(x_test, y_test)
    recognizer.save_model('enhanced_digits_letters_cnn_model.h5')
    
    return recognizer


def test_enhanced_recognition():
    recognizer = Enhanced_CNN_CharacterRecognizer('digits_and_letters')
    
    if not recognizer.load_model('enhanced_digits_letters_cnn_model.h5'):
        return
    
    while True:
        choice = input('Select (1-Test Image, 2-Exit): ').strip()
        
        if choice == '2':
            break
        elif choice == '1':
            image_path = input('Enter image path: ').strip().strip('"').strip("'")
            
            if not os.path.exists(image_path):
                continue
            
            try:
                predictions = recognizer.predict_character(image_path)
                recognizer.display_prediction_results(image_path, predictions)
            except Exception:
                continue


if __name__ == "__main__":
    while True:
        choice = input('Enter choice (1-Train, 2-Test, 3-Exit): ').strip()
        
        if choice == '1':
            train_enhanced_digits_and_letters()
            break
        elif choice == '2':
            test_enhanced_recognition()
            break
        elif choice == '3':
            break
        else:
            continue
