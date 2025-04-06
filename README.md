# captcha-verification
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50
NUM_CHANNELS = 1  # Grayscale
NUM_DIGITS = 6
DIGIT_CLASSES = 10  # 0-9
BATCH_SIZE = 32
EPOCHS = 50

# Load and preprocess data
def load_data(csv_path, image_folder):
    data = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for _, row in data.iterrows():
        img_path = os.path.join(image_folder, row['image_path'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
            
        # Resize and normalize
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        
        # Convert label to individual digits
        label_str = str(row['label']).zfill(NUM_DIGITS)
        label_digits = [int(c) for c in label_str]
        
        images.append(img)
        labels.append(label_digits)
    
    return np.array(images), np.array(labels)

# Load datasets
train_images, train_labels = load_data('train_labels.csv', 'train_images')
val_images, val_labels = load_data('val_labels.csv', 'val_images')
test_images, test_labels = load_data('test_labels.csv', 'test_images')

# Prepare labels for multi-output model
train_labels_digits = [train_labels[:, i] for i in range(NUM_DIGITS)]
val_labels_digits = [val_labels[:, i] for i in range(NUM_DIGITS)]
test_labels_digits = [test_labels[:, i] for i in range(NUM_DIGITS)]

# Convert labels to one-hot encoding
train_labels_digits = [to_categorical(train_labels[:, i], num_classes=DIGIT_CLASSES) for i in range(NUM_DIGITS)]
val_labels_digits = [to_categorical(val_labels[:, i], num_classes=DIGIT_CLASSES) for i in range(NUM_DIGITS)]
test_labels_digits = [to_categorical(test_labels[:, i], num_classes=DIGIT_CLASSES) for i in range(NUM_DIGITS)]

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    shear_range=0.05
)

# Model architecture
def build_model():
    input_layer = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
    
    # Shared convolutional base
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Separate output layers for each digit
    outputs = []
    for _ in range(NUM_DIGITS):
        digit_output = Dense(DIGIT_CLASSES, activation='softmax', name=f'digit_{_}')(x)
        outputs.append(digit_output)
    
    model = Model(inputs=input_layer, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = build_model()
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train the model
history = model.fit(
    datagen.flow(train_images, train_labels_digits, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(val_images, val_labels_digits),
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
test_loss, *test_accs = model.evaluate(test_images, test_labels_digits, verbose=0)

# Calculate overall accuracy (all digits correct)
def calculate_full_accuracy(model, images, true_labels):
    pred_digits = model.predict(images)
    correct = 0
    
    for i in range(len(images)):
        predicted = []
        for j in range(NUM_DIGITS):
            predicted.append(np.argmax(pred_digits[j][i]))
        
        actual = true_labels[i]
        
        if np.array_equal(predicted, actual):
            correct += 1
    
    return correct / len(images)

train_full_acc = calculate_full_accuracy(model, train_images, train_labels)
val_full_acc = calculate_full_accuracy(model, val_images, val_labels)
test_full_acc = calculate_full_accuracy(model, test_images, test_labels)

print("\nTraining Results:")
print(f"Individual digit accuracies: {[acc*100 for acc in test_accs]}")
print(f"Full captcha accuracy (all digits correct): {train_full_acc*100:.2f}%")

print("\nValidation Results:")
print(f"Full captcha accuracy (all digits correct): {val_full_acc*100:.2f}%")

print("\nTest Results:")
print(f"Full captcha accuracy (all digits correct): {test_full_acc*100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['digit_0_accuracy'], label='Digit 1 Accuracy')
plt.plot(history.history['digit_5_accuracy'], label='Digit 6 Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Example prediction
def predict_captcha(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    
    pred_digits = model.predict(img)
    captcha = ''.join([str(np.argmax(d[0])) for d in pred_digits])
    return captcha

# Test on a sample image
sample_image_path = 'test_images/123456.png'  # Replace with actual path
print(f"\nPredicted captcha: {predict_captcha(model, sample_image_path)}")
