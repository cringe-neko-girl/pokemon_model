import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from tqdm import tqdm  

# Paths for training and validation data
train_dir = "t_data/train"
val_dir = "t_data/test"

# Ensure data directories exist
if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
    raise FileNotFoundError(f"Training directory is missing or empty: {train_dir}")
if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
    raise FileNotFoundError(f"Validation directory is missing or empty: {val_dir}")

# Data augmentation and preprocessing
image_gen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=40,  # Rotate images up to 40 degrees
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2,  # Shift images vertically
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flips
    fill_mode='nearest'  # Fill missing pixels with nearest values
)

# Load training data
train_data_gen = image_gen.flow_from_directory(
    train_dir,
    target_size=(28, 28), 
    batch_size=16,
    class_mode='categorical'
)

# Load validation data
val_data_gen = image_gen.flow_from_directory(
    val_dir,
    target_size=(28, 28), 
    batch_size=16,
    class_mode='categorical'
)

# Build the CNN model with Batch Normalization
def build_cnn_with_batch_norm():
    model = Sequential([
        # Input layer
        tf.keras.layers.InputLayer(input_shape=(28, 28, 3)),
        
        # First convolutional layer
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        
        # Second convolutional layer
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        
        # Third convolutional layer
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        
        # Flatten layer
        Flatten(),
        
        # Dense layer
        Dense(512, activation='relu'),
        Dropout(0.5),
        
        # Output layer
        Dense(train_data_gen.num_classes, activation='softmax')
    ])
    return model

# Load or create the model
model_file = "created_models/best_model.keras"
if os.path.exists(model_file):
    model = load_model(model_file)
    print("Loaded pre-trained model.")
else:
    model = build_cnn_with_batch_norm()
    print("Built a new model.")

# Adaptive learning rate scheduler with warm-up and cosine decay
def scheduler(epoch, lr):
    warmup_epochs = 3  # Warm-up period
    initial_lr = 1e-4  # Base learning rate
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs  # Linear warm-up
    else:
        # Cosine decay after warm-up
        return initial_lr * 0.5 * (1 + tf.math.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

# Callback for learning rate scheduling
lr_scheduler = LearningRateScheduler(scheduler)

# Compile the model with an adaptive optimizer
optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-5)  # Adaptive optimizer with weight decay
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for training
early_stopping = EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True, min_delta=0.01
)

checkpoint = ModelCheckpoint(
    model_file, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# Training parameters
epochs = 20  
steps_per_epoch = ceil(train_data_gen.samples / train_data_gen.batch_size)
validation_steps = ceil(val_data_gen.samples / val_data_gen.batch_size)

# Initialize tqdm for progress tracking
progress_bar = tqdm(total=epochs, desc='Epoch Progress', position=0)

# Store training history
history_list = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Train the model for one epoch and save history
    history = model.fit(
        train_data_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data_gen,
        validation_steps=validation_steps,
        epochs=1, 
        callbacks=[early_stopping, checkpoint, reduce_lr, lr_scheduler]
    )
    
    # Add the current epoch's history to the history_list
    history_list.append(history.history)
    
    progress_bar.update(1)
    
    # Save the model after each epoch
    model.save(model_file)
    print(f"Model saved after epoch {epoch + 1}")

# Close the tqdm progress bar
progress_bar.close()

# Plot training progress
def plot_training(history_list):
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    # Collect data from each epoch
    for history in history_list:
        acc.extend(history['accuracy'])
        val_acc.extend(history['val_accuracy'])
        loss.extend(history['loss'])
        val_loss.extend(history['val_loss'])

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_training(history_list)
