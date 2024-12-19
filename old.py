import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

# Load and preprocess dataset
data = tf.keras.utils.image_dataset_from_directory(
    "animals",  # Path to dataset directory
    image_size=(128, 128),  # Resize images to smaller size
    batch_size=32
)

# Normalize dataset
data = data.map(lambda x, y: (x / 255.0, y))  # Normalize pixel values to [0, 1]

# Data augmentation for training
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1)
])

# Apply data augmentation to training data
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)

train = data.take(train_size)
train = train.map(lambda x, y: (data_augmentation(x), y))

val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size)

# Use transfer learning with MobileNetV2
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model layers

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(46, activation='softmax')  # Output layer for 90 classes
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
    validation_data=val,
    callbacks=[early_stop, lr_schedule, tensorboard_callback]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save("animal_classifier_model.h5")
