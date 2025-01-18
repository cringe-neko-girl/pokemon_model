import os

# Suppress TensorFlow logs to minimize console clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt  # For plotting training/validation accuracy and loss
from math import ceil  # For calculating the number of steps per epoch
import tensorflow as tf  # For TensorFlow and Keras functionality
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image data generators
from tensorflow.keras.models import Sequential, load_model  # For defining and loading the model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Input  # Layers for the CNN
from tensorflow.keras.optimizers import Adam  # Optimizer for model training
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # Callbacks for training

# Directories for training and validation datasets
train_dir = "t_data/train"  # Path to training data
val_dir = "t_data/test"     # Path to validation data

# Check if directories exist and are not empty
if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
    raise FileNotFoundError(f"Training directory is missing or empty: {train_dir}")
if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
    raise FileNotFoundError(f"Validation directory is missing or empty: {val_dir}")

# Data preprocessing and augmentation
image_generator = ImageDataGenerator(
    rescale=1.0 / 255,       # Normalize pixel values
    shear_range=0.2,         # Shear transformations
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Random horizontal flips
    validation_split=0.2     # Split data into training and validation sets
)

# Training data generator
train_data_gen = image_generator.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize all images to 150x150
    batch_size=32,          # Number of samples per batch
    class_mode='categorical', # Categorical labels for multi-class classification
    subset='training'        # Training subset
)

# Validation data generator
val_data_gen = image_generator.flow_from_directory(
    train_dir,               # Use training directory for validation split
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'      # Validation subset
)

# Define the CNN model
def build_model():
    model = Sequential([
        Input(shape=(150, 150, 3)),              # Input layer
        Conv2D(32, (3, 3), activation='relu'),   # Convolutional layer
        MaxPooling2D((2, 2)),                   # Max pooling layer
        Conv2D(64, (3, 3), activation='relu'),   # Convolutional layer
        MaxPooling2D((2, 2)),                   # Max pooling layer
        Conv2D(128, (3, 3), activation='relu'),  # Convolutional layer
        MaxPooling2D((2, 2)),                   # Max pooling layer
        Flatten(),                              # Flatten for dense layers
        Dense(256, activation='relu'),          # Fully connected layer
        BatchNormalization(),                   # Batch normalization
        Dropout(0.5),                           # Dropout for regularization
        Dense(128, activation='relu'),          # Fully connected layer
        BatchNormalization(),                   # Batch normalization
        Dropout(0.3),                           # Dropout for regularization
        Dense(train_data_gen.num_classes, activation='softmax')  # Output layer
    ])
    return model

# Load pre-trained model if exists
model_file = "best_model.keras"
if os.path.exists(model_file):
    model = load_model(model_file)  # Load the existing model
    print("Loaded pre-trained model from:", model_file)
else:
    model = build_model()  # If no pre-trained model, build a new one
    print("Building a new model")

# Compile the model with an adaptive learning rate scheduler and gradient clipping
# Learning rate warmup
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,  # Lower initial learning rate for stability
    decay_steps=100000,           # Number of steps before decay
    decay_rate=0.96,              # Decay rate
    staircase=True                 # Apply decay in discrete intervals
)

# Define the optimizer with gradient clipping
adaptive_optimizer = Adam(learning_rate=lr_schedule, clipvalue=1.0)  # Apply gradient clipping here

# Compile the model
model.compile(
    optimizer=adaptive_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks with stricter early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Stop training if validation loss doesn't improve
    patience=10,         # Wait for 10 epochs before stopping
    restore_best_weights=True,  # Restore the best weights
    min_delta=0.0001  # Set a min_delta to be stricter about stopping
)

checkpoint = ModelCheckpoint(
    "best_model.keras",  # Save the best model to this file
    save_best_only=True,
    monitor='val_loss',  # Monitor validation loss
    mode='min',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Reduce learning rate on plateau
    factor=0.2,          # Reduce LR by a factor of 0.2
    patience=5,          # Wait for 5 epochs before reducing LR
    min_lr=1e-6          # Minimum learning rate
)

# Training parameters
epochs = 50  # Total number of epochs
batch_size = 32

# Calculate steps per epoch
steps_per_epoch = ceil(train_data_gen.samples / batch_size)
validation_steps = ceil(val_data_gen.samples / batch_size)

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=validation_steps,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# Load the best model saved during training
model.load_weights("best_model.keras")

# Plot training and validation accuracy/loss
def plot_history(history):
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

# Display training progress
plot_history(history)
