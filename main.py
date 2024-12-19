import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

# Load and preprocess dataset with validation split
data = tf.keras.utils.image_dataset_from_directory(
    "animals",  # Path to dataset directory
    image_size=(128, 128),  # Resize images to smaller size
    batch_size=32,
    validation_split=0.2,  # Split 20% of data for validation
    subset="training",  # This subset is for training
    seed=123  # Set random seed for reproducibility
)

# Create validation dataset from the remaining 20%
val_data = tf.keras.utils.image_dataset_from_directory(
    "animals",  # Path to dataset directory
    image_size=(128, 128),  # Resize images to smaller size
    batch_size=32,
    validation_split=0.2,  # Split 20% of data for validation
    subset="validation",  # This subset is for validation
    seed=123  # Same random seed for reproducibility
)

# Normalize dataset
data = data.map(lambda x, y: (x / 255.0, y))  # Normalize pixel values to [0, 1]
val_data = val_data.map(lambda x, y: (x / 255.0, y))

# Data augmentation for training
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1)
])

# Apply data augmentation to training data
train = data.map(lambda x, y: (data_augmentation(x), y))

# Use transfer learning with MobileNetV2
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model layers

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(46, activation='softmax')  # Output layer for 46 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_schedule = LearningRateScheduler(lambda epoch: 1e-3 * 0.1 ** (epoch // 20))
tensorboard_callback = TensorBoard(log_dir="logs")

# Train the model
hist = model.fit(
    train,
    epochs=50,
    validation_data=val_data,
    callbacks=[early_stop, lr_schedule, tensorboard_callback]
)

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the trained model
model.save("new_data.h5")
