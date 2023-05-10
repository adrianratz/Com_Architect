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


# # Training with Multi-processing 
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


# # Training with Multi-processing including time graph

import time
import concurrent.futures

hyperparams = [    {'reg_rate': 0.001, 'dropout_rate': 0.1},    {'reg_rate': 0.001, 'dropout_rate': 0.15},    {'reg_rate': 0.001, 'dropout_rate': 0.2},]

def train_model(params):
    start_time = time.time()
    model = build_model(params['dropout_rate'], params['reg_rate'])
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=5, verbose=1)
    end_time = time.time()
    return model, history, end_time - start_time

models = []
execution_times = []
with concurrent.futures.ThreadPoolExecutor(max_workers=len(hyperparams)) as executor:
    futures = [executor.submit(train_model, params) for params in hyperparams]
    for idx, future in enumerate(concurrent.futures.as_completed(futures)):
        model, history, execution_time = future.result()
        models.append(model)
        execution_times.append(execution_time)
        print(f"Finished training model {idx+1}")



# # Training with Multiprocessing, print CPU usage and time
import time
import concurrent.futures
import psutil

hyperparams = [
    {'reg_rate': 0.001, 'dropout_rate': 0.1},
    {'reg_rate': 0.001, 'dropout_rate': 0.15},
    {'reg_rate': 0.001, 'dropout_rate': 0.2},
]

def train_model(params):
    start_time = time.time()
    model = build_model(params['dropout_rate'], params['reg_rate'])
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=1, verbose=1)
    end_time = time.time()
    cpu_usage = psutil.cpu_percent()
    return model, history, end_time - start_time, cpu_usage

models = []
execution_times = []
cpu_usages = []
with concurrent.futures.ThreadPoolExecutor(max_workers=len(hyperparams)) as executor:
    futures = [executor.submit(train_model, params) for params in hyperparams]
    for idx, future in enumerate(concurrent.futures.as_completed(futures)):
        model, history, execution_time, cpu_usage = future.result()
        models.append(model)
        execution_times.append(execution_time)
        cpu_usages.append(cpu_usage)
        print(f"Finished training model {idx+1} with CPU usage: {cpu_usage:.2f}% and execution time: {execution_time:.2f} seconds")

# Print execution times and CPU usages for each model in a table
import pandas as pd
df = pd.DataFrame({
    "Model": [f"Model {idx+1}" for idx in range(len(hyperparams))],
    "Execution Time (s)": execution_times,
    "CPU Usage (%)": cpu_usages
})
print(df.to_string(index=False))



# Plot the execution time graph
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.bar(range(len(execution_times)), execution_times, color='#5DA5DA')
plt.xticks(range(len(execution_times)), ['Model {}'.format(i+1) for i in range(len(execution_times))], fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Execution Time (seconds)', fontsize=16)
plt.title('Execution Time per Model', fontsize=18)
plt.savefig('/Users/aratwatte2/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/Nebraska-Lincoln/Spring-2023/Com_Architect/Project/execution_times_with_multi.png', dpi=300, bbox_inches='tight')
plt.show()


# # Training without Multi-processing 
import time
import matplotlib.pyplot as plt

hyperparams = [    {'reg_rate': 0.001, 'dropout_rate': 0.1},    {'reg_rate': 0.001, 'dropout_rate': 0.15},    {'reg_rate': 0.001, 'dropout_rate': 0.2},]

models = []
training_times = []

for idx, params in enumerate(hyperparams):
    print(f"Training model {idx+1} with parameters:")
    print(params)
    print("--------------------------")
    start_time = time.time()
    model = build_model(params['dropout_rate'], params['reg_rate'])
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=5, verbose=1)
    end_time = time.time()
    training_time = end_time - start_time
    models.append(model)
    training_times.append(training_time)
    print("--------------------------")
    print(f"Finished training model {idx+1}")
    print(f"Training time: {training_time:.2f} seconds")
    print("--------------------------")



# # Training without multiprocessing print cpu usage, time
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt

hyperparams = [{'reg_rate': 0.001, 'dropout_rate': 0.1},{'reg_rate': 0.001, 'dropout_rate': 0.15},{'reg_rate': 0.001, 'dropout_rate': 0.2}]

models = []
training_times = []
cpu_usages = []
memory_usages = []
histories = []

for idx, params in enumerate(hyperparams):
    print(f"Training model {idx+1} with parameters:")
    print(params)
    print("--------------------------")
    start_time = time.time()
    model = build_model(params['dropout_rate'], params['reg_rate'])
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, verbose=1)
    end_time = time.time()
    training_time = end_time - start_time
    models.append(model)
    training_times.append(training_time)
    
    # track CPU usage and memory usage
    cpu_usage = []
    memory_usage = []
    for i, (_, logs) in enumerate(history.history.items()):
        # get current CPU and memory usage
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        cpu_usage.append(cpu)
        memory_usage.append(mem)
        
        # print progress
        print(f"Epoch {i+1}: CPU usage={cpu:.2f}%, Memory usage={mem:.2f}%")
    
    # store results
    cpu_usages.append(cpu_usage)
    memory_usages.append(memory_usage)
    histories.append(history)
    
    print("--------------------------")
    print(f"Finished training model {idx+1}")
    print(f"Training time: {training_time:.2f} seconds\t CPU usage: {cpu_usage[-1]:.2f}%\t Memory usage: {memory_usage[-1]:.2f}%")
    print("--------------------------")



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
for i in range(len(hyperparams)):
    ax1.plot(cpu_usages[i], label=f"Model {i+1}", linewidth=2, color='#FFA500')
ax1.legend()
# ax1.set_title("CPU Usage", fontsize=20)
ax1.set_xlabel("Epoch", fontsize=16)
ax1.set_ylabel("CPU Usage (%)", fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=14)

for i in range(len(hyperparams)):
    ax2.plot(memory_usages[i], label=f"Model {i+1}", linewidth=2, color= '#1f77b4')
ax2.legend()
# ax2.set_title("Memory Usage", fontsize=20)
ax2.set_xlabel("Epoch", fontsize=16)
ax2.set_ylabel("Usage (%)", fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.savefig("usage_graphs.png")
plt.show()



# Increase font size of ticks
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# Plot the training times
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(hyperparams)+1), training_times, 'bo-')
ax.set_title('Training Time vs Model')
ax.set_xlabel('Model')
ax.set_ylabel('Training Time (seconds)')
ax.set_xticks(range(1, len(hyperparams)+1))
plt.savefig('/Users/aratwatte2/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/Nebraska-Lincoln/Spring-2023/Com_Architect/Project/without_parallel_execution_times.png', dpi=300, bbox_inches='tight')
plt.show()


# # Train Val Accuracy and Loss

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


# # Evaluating on test set with multiprocessing 

import concurrent.futures
import time
import psutil

# Define a function to evaluate a model on test data
def evaluate_model(model, test_images, test_labels):
    # Get model name
    model_name = f"Model {models.index(model)+1}"
    
    # Record start time
    start_time = time.time()
    
    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, verbose=1)
    
    # Record end time
    end_time = time.time()
    
    # Calculate execution time
    exec_time = end_time - start_time
    
    # Get CPU usage
    cpu_usage = psutil.cpu_percent()
    
    # Print results
    print(f"{model_name}: Test accuracy = {test_acc:.4f}, Test loss = {test_loss:.4f}, Execution time = {exec_time:.2f} s, CPU usage = {cpu_usage}%")
    
    # Return results
    return (model_name, test_acc, test_loss, exec_time, cpu_usage)

# Evaluate each model on test data
print("Evaluating models on test data...")
with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
    futures = [executor.submit(evaluate_model, model, test_images, test_labels) for model in models]

# Get results
results = [future.result() for future in concurrent.futures.as_completed(futures)]

# Print results in a table
print("\nResults:")
print(f"{'Model':<10}{'Accuracy':<15}{'Loss':<15}{'Execution Time':<25}{'CPU Usage (%)':<15}")
for i in range(len(models)):
    model_name, test_acc, test_loss, exec_time, cpu_usage = results[i]
    print(f"{model_name:<10}{test_acc:<15.4f}{test_loss:<15.4f}{exec_time:<25.2f}{cpu_usage:<15.2f}")


# # Evaluation on test without multi process
import concurrent.futures
import time
import psutil

# Define lists to store results
model_names = []
test_losses = []
test_accs = []
exec_times = []
cpu_usages = []

# Evaluate each model on test data
print("Evaluating models on test data...")
for idx, model in enumerate(models):
    # Get model name
    model_name = f"Model {idx+1}"
    model_names.append(model_name)
    
    # Evaluate model on test data
    start_time = time.time()
    test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, verbose=1)
    end_time = time.time()
    exec_time = end_time - start_time
    exec_times.append(exec_time)
    
    # Get CPU usage
    cpu_usage = psutil.cpu_percent()
    cpu_usages.append(cpu_usage)
    
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # Print results
    print(f"{model_name}: Test accuracy = {test_acc:.4f}, Test loss = {test_loss:.4f}, Execution time = {exec_time:.4f}s, CPU usage = {cpu_usage:.2f}%")

# Print results in a table
print("\nResults:")
print(f"{'Model':<10}{'Accuracy':<15}{'Loss':<15}{'Execution Time':<20}{'CPU Usage':<15}")
for i in range(len(models)):
    print(f"{model_names[i]:<10}{test_accs[i]:<15.4f}{test_losses[i]:<15.4f}{exec_times[i]:<20.4f}s{cpu_usages[i]:<15.2f}%")


# # Adversarial Image Generator 
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
    plt.savefig(f"conf_mat_Adversarial_{model.name}_SVHN.png", dpi=300)
    plt.show()


# # One pixel
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
# Generate adversarial images
test_images_copy = test_images.copy()
num_images_to_generate = len(test_images_copy)
one_pixel_test_images = []
for image in tqdm(test_images_copy[:num_images_to_generate], desc="Generating adversarial images"):
    one_pixel_image = one_pixel_method(image)
    one_pixel_test_images.append(one_pixel_image)
one_pixel_test_images = np.array(one_pixel_test_images)


# # One pixel method without parallelism 
import psutil
import time

# Create DataFrame to store accuracies and execution times
df = pd.DataFrame(columns=["Model", "One pixel Accuracy", "Test Accuracy", "Execution Time", "CPU Usage"])

# Evaluate each model on original test images and adversarial images
for idx, model in enumerate(models):
    print(f"Evaluating model {idx+1} on original test images")
    start_time = time.time()
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
    
    # Add accuracies, execution time, and CPU usage to DataFrame
    end_time = time.time()
    execution_time = end_time - start_time
    cpu_usage = psutil.cpu_percent()
    df = df.append({"Model": f"Model {idx+1}", 
                    "One pixel Accuracy": accuracy, 
                    "Test Accuracy": original_accuracy, 
                    "Execution Time": execution_time, 
                    "CPU Usage": cpu_usage}, 
                   ignore_index=True)

# Print DataFrame
print(df.to_string(index=False))


# # One pixel method with parallelism 
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import time
import psutil

def evaluate_model(model, model_idx):
    # Evaluate model on original test images
    print(f"Evaluating model {model_idx+1} on original test images")
    start_time_orig = time.time()
    original_predictions = model.predict(test_images)
    original_predicted_labels = np.argmax(original_predictions, axis=1)
    true_labels = np.array(test_labels)
    original_accuracy = np.mean(original_predicted_labels == true_labels)
    end_time_orig = time.time()
    print(f"Accuracy on original test images for model {model_idx+1}: {original_accuracy:.4f}")

    # Evaluate model on adversarial images
    print(f"Evaluating model {model_idx+1} on adversarial images")
    start_time_adv = time.time()
    predictions = model.predict(one_pixel_test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == true_labels[:num_images_to_generate])
    end_time_adv = time.time()
    print(f"Accuracy on one pixel images for model {model_idx+1}: {accuracy:.4f}")

    # Get CPU usage
    cpu_usage = psutil.cpu_percent()

    # Return accuracies, model index, start/end times, and CPU usage
    return model_idx, accuracy, original_accuracy, start_time_orig, end_time_orig, start_time_adv, end_time_adv, cpu_usage

# Create DataFrame to store accuracies and execution times
df = pd.DataFrame(columns=["Model", "One pixel Accuracy", "Test Accuracy", "Orig Start Time", "Orig End Time", "Adv Start Time", "Adv End Time", "CPU Usage"])

# Evaluate models in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for i, model in enumerate(models):
        future = executor.submit(evaluate_model, model, i)
        futures.append(future)

        # Update progress using tqdm
        progress = f"Evaluating model {i+1}/{len(models)}"
        tqdm.write(progress)

    # Wait for all futures to finish with a timeout of 300 seconds
    concurrent.futures.wait(futures, timeout=300)

    # Get results from futures and add them to DataFrame
    for future in futures:
        model_idx, accuracy, original_accuracy, start_time_orig, end_time_orig, start_time_adv, end_time_adv, cpu_usage = future.result()
        df = df.append({"Model": f"Model {model_idx+1}", 
                        "One pixel Accuracy": accuracy, 
                        "Test Accuracy": original_accuracy,
                        "Orig Start Time": start_time_orig,
                        "Orig End Time": end_time_orig,
                        "Adv Start Time": start_time_adv,
                        "Adv End Time": end_time_adv,
                        "CPU Usage": cpu_usage}, 
                       ignore_index=True)

# Print DataFrame
print(df.to_string(index=False))


# # One pixel Confusion Matrices (Largest Frequency Labels)
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
    plot_confusion_matrix(adversarial_confusion_matrix, axs[1], f"Adversarial Confusion Matrix (Model {i+1})", cmap="Blues", top_labels=5)

    # Add space between subplots and display plot
    plt.tight_layout()
    plt.savefig(f"conf_mat_Adversarial_{model.name}_SVHN.png", dpi=300)
    plt.show()

