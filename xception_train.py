import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, Conv2D, Multiply, Concatenate, Add, Activation, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import seaborn as sns


# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Running on CPU only.")

# Define paths
data_dir = "C:\project\png_images"

# Image preprocessing
img_size = (256, 256) # Input image size
batch_size = 256  # Number of samples processed per batch

# Function to process filenames and extract labels
def process_filename(filename):
    """
    Processes filenames to extract dataset type (train/test) and label (pneumothorax/normal).
    Args:
        filename (str): The filename of the image.
    Returns:
        tuple: A tuple containing the dataset type (train/test) and the label (pneumothorax/normal).
    """
    parts = filename.split("_")
    set_type = parts[1]  # train or test
    label = "pneumothorax" if int(parts[2]) == 1 else "normal"  # Convert to string
    return set_type, label

# Load data
train_images = []
train_labels = []
test_images = []
test_labels = []

for file in os.listdir(data_dir):
    if file.endswith(".png"):
        set_type, label = process_filename(file)
        image_path = os.path.join(data_dir, file)
        if set_type == "train":
            train_images.append(image_path)
            train_labels.append(label)
        else:
            test_images.append(image_path)
            test_labels.append(label)

# Convert to NumPy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Data Generators with improved augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    rotation_range=30,  # Increased rotation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # Add shearing
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_images, 'label': train_labels}),
    x_col='filename',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_images, 'label': train_labels}),
    x_col='filename',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

test_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_images, 'label': test_labels}),
    x_col='filename',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Load Xception pre-trained model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = True  # Enable fine-tuning

# Optionally freeze first few layers to retain low-level features
for layer in base_model.layers[:30]:  # Freezing fewer layers for better adaptation
    layer.trainable = False

# Define Spatial Attention Module
def spatial_attention_module(input_feature):
    """ Implements a Spatial Attention Module to highlight relevant regions.
    Args:
        input_feature (Tensor): Input feature map.
    Returns:
        Tensor: Feature map after applying spatial attention.    
    """
    avg_pool = GlobalAveragePooling2D(keepdims=True)(input_feature)
    max_pool = GlobalMaxPooling2D(keepdims=True)(input_feature)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    conv = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(concat)
    return Add()([input_feature, Multiply()([input_feature, conv])])

def channel_attention_module(input_feature, ratio=8):
    """
    Implements the Channel Attention Module (CAM).
    Args:
        input_feature (Tensor): Input feature map.
        ratio (int): Reduction ratio for dimensionality reduction.
    Returns:
        Tensor: Feature map after applying channel attention.
    """
    channel = input_feature.shape[-1]

    # Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)

    # Global Max Pooling
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)

    # Shared MLP (Fully Connected Layers)
    shared_layer = Dense(channel // ratio, activation="relu")
    shared_layer_output = Dense(channel, activation="sigmoid")

    avg_out = shared_layer(avg_pool)
    avg_out = shared_layer_output(avg_out)

    max_out = shared_layer(max_pool)
    max_out = shared_layer_output(max_out)

    # Combine attention scores
    attention = Add()([avg_out, max_out])
    attention = Activation("sigmoid")(attention)

    # Apply attention weights
    return Add()([input_feature, attention])

def cbam_module(input_feature):
    """
    Combines Channel and Spatial Attention into CBAM.
    Args:
        input_feature (Tensor): Input feature map.
    Returns:
        Tensor: Feature map after applying CBAM.
    """
    attn_feature = channel_attention_module(input_feature)  # Apply channel attention first
    attn_feature = spatial_attention_module(attn_feature)  # Then apply spatial attention
    return attn_feature


# Apply attention mechanism to the feature extractor
attn_feature = cbam_module(base_model.output)

# Add custom classification layers
x = GlobalAveragePooling2D()(attn_feature)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)  
x = Dropout(0.6)(x)  # Reduced dropout for stability
output_layer = Dense(1, activation='sigmoid')(x)

# Compile model with AdamW optimizer
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=AdamW(learning_rate=0.000035, weight_decay=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Adaptive Learning Rate Reduction
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Define class weights to balance the dataset
class_weights = {0: 1.0, 1: 1.75}  # Give higher weight to pneumothorax class

# Train model with 25 epochs
epochs = 25 
history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, class_weight=class_weights, callbacks=[lr_callback])

# Evaluate model
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Predict on test set with adjusted threshold
y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.50).astype(int)  
y_true = test_generator.classes

# Generate classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["normal", "pneumothorax"]))

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract False Positives, False Negatives, True Positives, True Negatives
TN, FP, FN, TP = cm.ravel()
print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"True Negatives: {TN}")
print(f"False Negatives: {FN}")

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pneumothorax"], yticklabels=["Normal", "Pneumothorax"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Identify False Negatives
false_negatives_indices = np.where((y_true == 1) & (y_pred.flatten() == 0))[0]

# Get decision scores (probabilities) for false negatives
fn_scores = y_pred_probs[false_negatives_indices]

print("\nPredicted probabilities for False Negatives:")
for i, score in enumerate(fn_scores):
    print(f"{i+1}. Score: {score[0]:.4f}")

model.save("pneumothorax_xception_cbam_256_2.keras")
print("Model saved successfully!")

# Plot training history
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot Training Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()