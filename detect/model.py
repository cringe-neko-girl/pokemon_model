import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.applications import EfficientNetB0  # Pre-trained model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
from math import ceil
import numpy as np

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
    target_size=(224, 224),  # Resize images to 224x224 for EfficientNet
    batch_size=32,
    class_mode='categorical'
)

# Load validation data
val_data_gen = image_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Build the model using EfficientNetB0 as the base
def build_model():
    # Load EfficientNetB0 without the top layer (pre-trained on ImageNet)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers to retain pre-trained features
    base_model.trainable = False
    
    # Add custom classification layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  # Flatten the feature maps
        Dense(512, activation='relu'),  # Fully connected layer
        Dropout(0.5),  # Regularization
        Dense(train_data_gen.num_classes, activation='softmax')  # Output layer
    ])
    return model

# Load or create the model
model_file = "best_model_efficientnet.keras"
if os.path.exists(model_file):
    model = load_model(model_file)
    print("Loaded pre-trained model.")
else:
    model = build_model()
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
    monitor='val_accuracy', patience=10, restore_best_weights=True, min_delta=0.01
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

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data_gen,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[early_stopping, checkpoint, reduce_lr, lr_scheduler]
)

# Plot training progress
def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
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

plot_training(history)

