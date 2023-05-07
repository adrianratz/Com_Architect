#!/usr/bin/env python
# coding: utf-8

# # Import Libraries 

# In[1]:


import numpy as np
import tensorflow as tf
import scipy.io as sio
import keras
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import glob
import re
import random
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from random import randint
import pandas as pd


# # Load data

# In[2]:


def load_filenames_labels(mode):
    label_dict, class_description = build_label_dicts()
    filenames_labels = []

    if mode == 'train':
        # Load training data
        filenames = glob.glob('/Users/aratwatte2/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/Nebraska-Lincoln/Spring-2023/Deep_Learning/Project/tiny-imagenet-200 2/train/*/images/*.JPEG')
        for filename in filenames:
            match = re.search(r'n\d+', filename)
            label = label_dict[match.group()]  # Convert label to integer
            filenames_labels.append((filename, label))

    elif mode == 'val':
        # Load validation data
        with open('/Users/aratwatte2/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/Nebraska-Lincoln/Spring-2023/Deep_Learning/Project/tiny-imagenet-200 2/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                filename = '/Users/aratwatte2/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/Nebraska-Lincoln/Spring-2023/Deep_Learning/Project/tiny-imagenet-200 2/val/images/' + split_line[0]
                label = label_dict[split_line[1]]  # Convert label to integer
                filenames_labels.append((filename, label))

    return filenames_labels

def build_label_dicts():

  label_dict, class_description = {}, {}
  with open('/Users/aratwatte2/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/Nebraska-Lincoln/Spring-2023/Deep_Learning/Project/tiny-imagenet-200 2/wnids.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset = line[:-1]  # remove \n
      label_dict[synset] = i
  with open('/Users/aratwatte2/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/Nebraska-Lincoln/Spring-2023/Deep_Learning/Project/tiny-imagenet-200 2/words.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset, desc = line.split('\t')
      desc = desc[:-1]  # remove \n
      if synset in label_dict:
        class_description[label_dict[synset]] = desc

  return label_dict, class_description



def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [64, 64])
    image = image / 255.0  # Normalize the image
    return image

def create_dataset(filenames_labels, batch_size=32, shuffle=True):
    filenames = [filename for filename, label in filenames_labels]
    labels = [label for filename, label in filenames_labels]

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filenames_labels))
    dataset = dataset.map(lambda filename, label: (load_and_preprocess_image(filename), label))
    dataset = dataset.batch(batch_size)
    return dataset




# Load train filenames and labels
print("Loading train filenames and labels...")
train_filenames_labels = load_filenames_labels('train')

# Load validation filenames and labels
print("Loading validation filenames and labels...")
test_filenames_labels = load_filenames_labels('val')

# Separate train images and labels
print("Separating train images and labels...")
train_images = []
train_labels = []
for filename, label in tqdm(train_filenames_labels, desc="Processing train data"):
    image = load_and_preprocess_image(filename)
    train_images.append(image)
    train_labels.append(label)


# Separate validation images and labels
print("Separating validation images and labels...")
test_images = []
test_labels = []
for filename, label in tqdm(test_filenames_labels, desc="Processing validation data"):
    image = load_and_preprocess_image(filename)
    test_images.append(image)
    test_labels.append(label)


# Convert train images and labels to NumPy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Convert validation images and labels to NumPy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Split train images and labels into train and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.15, random_state=22
)

print("Train images shape after splitting:", train_images.shape)
print("Train labels shape after splitting:", train_labels.shape)
print("Validation images shape after splitting:", val_images.shape)
print("Validation labels shape after splitting:", val_labels.shape)
print("Test images shape after splitting:", test_images.shape)
print("Test labels shape after splitting:", test_labels.shape)


# # Mobile net 

# In[3]:


def build_model(dropout_rate, regularization_rate):
    keras.backend.clear_session()
    # Load the MobileNet model (pre-trained on ImageNet)
    mobilenet = MobileNet(input_shape=(64, 64, 3), include_top=False, weights='imagenet')

    # Freeze the pre-trained layers
    mobilenet.trainable = False

    # Build your model by adding the MobileNet base, additional convolutional layers, and regularization
    model = Sequential([
        mobilenet,
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(regularization_rate)),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(regularization_rate)),
        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(regularization_rate)),
        GlobalAveragePooling2D(),
        Dropout(dropout_rate),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)),
        Dropout(dropout_rate),
        Dense(200, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Set up early stopping to prevent overfitting
#     early_stop = EarlyStopping(monitor='val_loss', patience=3)

#     return model, early_stop
    return model


# # OUR SVHN MODEL

# In[4]:


# def build_model(dropout_rate, regularization_rate):
#     keras.backend.clear_session()

#     model = keras.Sequential([
#         keras.layers.Conv2D(32, (3, 3), padding='same', 
#                                activation='relu',
#                                input_shape=(64, 64, 3),
#                                kernel_regularizer=regularizers.l2(regularization_rate)),
#         keras.layers.BatchNormalization(),
#         keras.layers.Conv2D(32, (3, 3), padding='same', 
#                             activation='relu',
#                             kernel_regularizer=regularizers.l2(regularization_rate)),
#         keras.layers.MaxPooling2D((2, 2)),
#         keras.layers.Dropout(dropout_rate),

#         keras.layers.Conv2D(64, (3, 3), padding='same', 
#                                activation='relu',
#                                kernel_regularizer=regularizers.l2(regularization_rate)),
#         keras.layers.BatchNormalization(),
#         keras.layers.Conv2D(64, (3, 3), padding='same',
#                             activation='relu',
#                             kernel_regularizer=regularizers.l2(regularization_rate)),
#         keras.layers.MaxPooling2D((2, 2)),
#         keras.layers.Dropout(dropout_rate),

#         keras.layers.Conv2D(128, (3, 3), padding='same', 
#                                activation='relu',
#                                kernel_regularizer=regularizers.l2(regularization_rate)),
#         keras.layers.BatchNormalization(),
#         keras.layers.Conv2D(128, (3, 3), padding='same',
#                             activation='relu',
#                             kernel_regularizer=regularizers.l2(regularization_rate)),
#         keras.layers.MaxPooling2D((2, 2)),
#         keras.layers.Dropout(dropout_rate),

#         keras.layers.Flatten(),
#         keras.layers.Dense(256, activation='relu',
#                            kernel_regularizer=regularizers.l2(regularization_rate)),
#         keras.layers.Dropout(dropout_rate),    
#         keras.layers.Dense(200,  activation='softmax')
#     ])

#     # Compile the model
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     return model


# # Training with Multi-processing

# In[ ]:


import concurrent.futures
hyperparams = [
    {'reg_rate': 0.001, 'dropout_rate': 0.1},
    {'reg_rate': 0.001, 'dropout_rate': 0.15},
    {'reg_rate': 0.001, 'dropout_rate': 0.2},
]
def train_model(params):
    model= build_model(params['dropout_rate'], params['reg_rate'])
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, verbose=1)
    return model, history

models = []
with concurrent.futures.ThreadPoolExecutor(max_workers=len(hyperparams)) as executor:
    futures = [executor.submit(train_model, params) for params in hyperparams]
    for idx, future in enumerate(concurrent.futures.as_completed(futures)):
        model, history = future.result()
        models.append(model)
        print(f"Finished training model {idx+1}")


# # Training 

# In[5]:


hyperparams = [
    {'reg_rate': 0.001, 'dropout_rate': 0.1},
    {'reg_rate': 0.001, 'dropout_rate': 0.15},
    {'reg_rate': 0.001, 'dropout_rate': 0.2},
]


models = []

for idx, params in enumerate(hyperparams):
    print(f"Training model {idx+1} with parameters:")
    print(params)
    print("--------------------------")
    model= build_model(params['dropout_rate'], params['reg_rate'])
#     model, early_stop = build_model(params['dropout_rate'], params['reg_rate'])
#     history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=100, callbacks=[early_stop], verbose=1)
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, verbose=1)
    models.append(model)
    print("--------------------------")
    print(f"Finished training model {idx+1}")
    print("--------------------------")


# # Train Val Accuracy and Loss

# In[6]:


# Define a color palette
palette = sns.color_palette("husl", len(models))

for idx,model in enumerate(models):
    # Evaluate train and validation accuracies and losses
    train_acc = model.history.history['accuracy']
    val_acc = model.history.history['val_accuracy']
    train_loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']
    title_loss = f'Model {idx+1}: Loss'
    title_acc = f'Model {idx+1}: Accuracy'

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].plot(train_loss, label='Training ', linewidth=3, color='#50C878')
    ax[0].plot(val_loss, label='Validation ', linewidth=3, color='#FF5733')
    ax[0].set_title(title_loss, fontsize=24, fontweight='bold')
    ax[0].set_xlabel('Epochs', fontsize=28)
    ax[0].set_ylabel('Loss', fontsize=28)
    ax[0].legend(fontsize=28)

    ax[1].plot(train_acc, label='Training ', linewidth=3, color='#50C878')
    ax[1].plot(val_acc, label='Validation ', linewidth=3, color= '#FF5733')
    ax[1].set_title(title_acc, fontsize=24, fontweight='bold')
    ax[1].set_xlabel('Epochs', fontsize=28)
    ax[1].set_ylabel('Accuracy', fontsize=28)
    ax[1].legend(fontsize=28)

    # Set tick locations for x-axis
    epochs = len(train_acc)
    tick_values = np.arange(0, epochs, 5)
    ax[0].set_xticks(tick_values)
    ax[1].set_xticks(tick_values)

    # Set tick font size
    ax[0].tick_params(axis='both', which='major', labelsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=20)

    # Set background color to white
    ax[0].set_facecolor('white')
    ax[1].set_facecolor('white')

    # Set border color to black
    ax[0].spines['bottom'].set_color('black')
    ax[0].spines['top'].set_color('black')
    ax[0].spines['right'].set_color('black')
    ax[0].spines['left'].set_color('black')

    ax[1].spines['bottom'].set_color('black')
    ax[1].spines['top'].set_color('black')
    ax[1].spines['right'].set_color('black')
    ax[1].spines['left'].set_color('black')

    plt.tight_layout()
    plt.savefig(f'model_{idx+1}_loss_and_accuracy.png', dpi=300)
    plt.show()


# # Evaluating on test set

# In[7]:


# Define lists to store results
model_names = []
test_losses = []
test_accs = []

# Evaluate each model on test data
print("Evaluating models on test data...")
for idx, model in enumerate(models):
    # Get model name
    model_name = f"Model {idx+1}"
    model_names.append(model_name)
    
    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, verbose=1)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # Print results
    print(f"{model_name}: Test accuracy = {test_acc:.4f}, Test loss = {test_loss:.4f}")

# Print results in a table
print("\nResults:")
print(f"{'Model':<10}{'Accuracy':<15}{'Loss':<15}")
for i in range(len(models)):
    print(f"{model_names[i]:<10}{test_accs[i]:<15.4f}{test_losses[i]:<15.4f}")


# # Adversarial Image Generator 

# In[8]:


def generate_image_adversary(pretrained_model, image, label, eps, num_steps, step_size):
    # Cast the image
    image = tf.cast(image, tf.float32)
    # Rescale the image
    # image = image / 255.0
    # Add a batch dimension
    image = tf.expand_dims(image, axis=0)

    # Convert the label to one-hot encoded format
    label = tf.one_hot(label, depth=200)
    label = tf.expand_dims(label, axis=0)

    for i in range(num_steps):
        with tf.GradientTape() as tape:
            # Indicate that our image should be
            # watched for gradient updates
            tape.watch(image)
            # Use pretrained model to make prediction
            # Then compute the loss
            prediction = pretrained_model(image)
            loss = tf.keras.losses.categorical_crossentropy(label, prediction)
        # Calculate gradients of loss, then get sign
        gradient = tape.gradient(loss, image)
        signed_gradient = tf.sign(gradient)

        # Make adversarial image
        adv_image = (image + (signed_gradient * step_size)).numpy()

        # Update the image with the adversarial perturbation
        image = image + (signed_gradient * step_size)
        image = tf.clip_by_value(image, 0, 1)

        # Remove batch dimension and rescale the image back to [0, 255]
        adv_image = tf.squeeze(adv_image, axis=0)
        adv_image = tf.clip_by_value(adv_image, 0, 1)

        return adv_image.numpy()


# # Adversarial Evaluation on Test set

# In[22]:


adversarial_accuracies = []
test_accuracies = []

for model in models:
    # Evaluate test accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    test_accuracies.append(test_acc)
    
    # Generate adversarial examples for test set
    adversarial_images = []
    for i in tqdm(range(len(test_images)), desc='Generating adversarial examples'):
        adv_image = generate_image_adversary(pretrained_model=model, 
                                             image=test_images[i], 
                                             label=test_labels[i], 
                                             eps=0.1, 
                                             num_steps=1, 
                                             step_size=0.0005)
        adversarial_images.append(adv_image)
    adversarial_images = np.array(adversarial_images)
    
    # Evaluate adversarial accuracy
    adv_loss, adv_acc = model.evaluate(adversarial_images, test_labels, verbose=1)
    adversarial_accuracies.append(adv_acc)



# # Adversarial Confusion Matrices (Largest Frequency Labels)

# In[23]:


def plot_confusion_matrix(matrix, ax, title, cmap, top_labels=5):
    matrix_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    top_idx = np.argsort(-matrix_norm.sum(axis=0))[:top_labels]  # get the indices of the top labels
    matrix_norm = matrix_norm[top_idx][:, top_idx]  # select only the top labels
    labels = np.arange(top_labels)
    sns.set(font_scale=1.6)
    sns.heatmap(matrix_norm, annot=True, annot_kws={"size": 16}, cmap=cmap, ax=ax, fmt=".2f", xticklabels=labels, yticklabels=labels)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Predicted label", fontsize=18)
    ax.set_ylabel("True label", fontsize=18)
    ax.tick_params(labelsize=14)

    
for i, model in enumerate(models):
    # Get the predictions of the model on the original test set and the adversarial images
    original_preds = model.predict(np.array(test_images))
    adversarial_preds = model.predict(np.array(adversarial_images))

    # Get the actual labels of the original test set and the adversarial images
    original_labels = np.array(test_labels)
    adversarial_labels = np.array(test_labels)

    # Get the confusion matrices for the original test set and the adversarial images
    original_confusion_matrix = confusion_matrix(original_labels, np.argmax(original_preds, axis=1))
    adversarial_confusion_matrix = confusion_matrix(adversarial_labels, np.argmax(adversarial_preds, axis=1))

    # Create a new figure and plot the confusion matrices
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    plot_confusion_matrix(original_confusion_matrix, axs[0], f"Original Confusion Matrix (Model {i+1})", cmap="Greens", top_labels=5)
    plot_confusion_matrix(adversarial_confusion_matrix, axs[1], f"Adversarial Confusion Matrix (Model {i+1})", cmap="Blues", top_labels=5)

    # Add space between subplots and display plot
    plt.tight_layout()
    plt.savefig(f"conf_mat_Adversarial_Model {i+1}_SVHN.png", dpi=300)
    plt.show()


# # One pixel

# In[24]:


def one_pixel_method(numpy_array):
    # 80% chance of this happening
    if randint(0, 100) <= 60:
#         print("If condition passed: Applying one-pixel attack")
        (rows, columns, channels) = numpy_array.shape
        num_pixels = int(rows * columns * 0.2)  # Change up to 20% of the pixels
        for _ in range(num_pixels):
            random_row = randint(0, rows - 1)
            random_col = randint(0, columns - 1)
            for channel in range(channels):
                old_val = numpy_array[random_row, random_col, channel]
                new_val = numpy_array[:, :, channel].mean() + (numpy_array[:, :, channel].std() * 0.5)
                numpy_array[random_row, random_col, channel] = new_val
    else:
        pass
#         print("If condition failed: No attack applied")
    return numpy_array


# # Generate one pixel images on test set

# In[25]:


# Generate adversarial images
test_images_copy = test_images.copy()
num_images_to_generate = len(test_images_copy)
one_pixel_test_images = []
for image in tqdm(test_images_copy[:num_images_to_generate], desc="Generating adversarial images"):
    one_pixel_image = one_pixel_method(image)
    one_pixel_test_images.append(one_pixel_image)
one_pixel_test_images = np.array(one_pixel_test_images)


# In[26]:


# Create DataFrame to store accuracies
df = pd.DataFrame(columns=["Model", "One pixel Accuracy", "Test Accuracy"])

# Evaluate each model on original test images and adversarial images
for idx, model in enumerate(models):
    print(f"Evaluating model {idx+1} on original test images")
    original_predictions = model.predict(test_images)
    original_predicted_labels = np.argmax(original_predictions, axis=1)
    true_labels = np.array(test_labels)
    original_accuracy = np.mean(original_predicted_labels == true_labels)
    print(f"Accuracy on original test images for model {idx+1}: {original_accuracy:.4f}")

    # Evaluate each model on adversarial images
    print(f"Evaluating model {idx+1} on adversarial images")
    predictions = model.predict(one_pixel_test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == true_labels[:num_images_to_generate])
    print(f"Accuracy on one pixel images for model {idx+1}: {accuracy:.4f}")

    # Add accuracies to DataFrame
    df = df.append({"Model": f"Model {idx+1}", 
                    "One pixel Accuracy": accuracy, 
                    "Test Accuracy": original_accuracy}, 
                   ignore_index=True)

# Print DataFrame
print(df.to_string(index=False))


# # One pixel Confusion Matrices (Largest Frequency Labels)

# In[28]:


for i, model in enumerate(models):
    # Get the predictions of the model on the original test set and the adversarial images
    original_preds = model.predict(test_images)
    adversarial_preds = model.predict(one_pixel_test_images)

    # Get the actual labels of the original test set and the adversarial images
    original_labels = np.array(test_labels)
#     adversarial_labels = np.array(test_labels)

    # Get the confusion matrices for the original test set and the adversarial images
    original_confusion_matrix = confusion_matrix(original_labels, np.argmax(original_preds, axis=1))
    adversarial_confusion_matrix = confusion_matrix(original_labels, np.argmax(adversarial_preds, axis=1))

    # Create a new figure and plot the confusion matrices
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    plot_confusion_matrix(original_confusion_matrix, axs[0], f"Original Confusion Matrix (Model {i+1})", cmap="Greens", top_labels=5)
    plot_confusion_matrix(adversarial_confusion_matrix, axs[1], f"One-pixel Confusion Matrix (Model {i+1})", cmap="Purples", top_labels=5)

    # Add space between subplots and display plot
    plt.tight_layout()
    plt.savefig(f"conf_mat_OnePixel_Model {i+1}_SVHN.png", dpi=300)
    plt.show()

