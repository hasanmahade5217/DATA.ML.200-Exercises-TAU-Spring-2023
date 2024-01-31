# Importing necessary libraries

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)



# Data Set location in the local machine
dataset_dir = 'E:/OneDrive - TUNI.fi/Tampere University (MSc in CS-DS)/Year 2/Period 4/DATA.ML.200 Pattern Recognition and Machine Learning/Exercise_2/Data/GTSRB_subset_2'


# Defining some properties for the image
batch_size = 32
img_height = 64
img_width = 64

####  Split data into two parts - 80% for training and 20% for testing.

# Training Data
train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

## Testing Data
test_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Checking the class names 
class_names = train_ds.class_names
print(class_names)

"""
# Visualization
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
"""
# Checking image properties
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# #### Normalizing

normalization_layer = tf.keras.layers.Rescaling(1./255)

# Normalizing training dataset
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
train_image_batch, train_labels_batch = next(iter(normalized_train_ds))
first_image = train_image_batch[0]

# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


# # Normalizing test dataset
normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
test_image_batch, test_labels_batch = next(iter(normalized_test_ds))
first_image = test_image_batch[0]

# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


# #### Model 1

# Simple Sequential structure
model = tf.keras.models.Sequential()

# Flatten 2D input image to a 1D vector
model.add(tf.keras.layers.Flatten(input_shape=(64,64,3)))
print(model.output_shape)

# Add 1st full connected layer (hidden layer in MLP terminology)
model.add(tf.keras.layers.Dense(10,activation='sigmoid'))

# Add 2nd full connected layer (hidden layer in MLP terminology)
model.add(tf.keras.layers.Dense(10,activation='sigmoid'))

# Add the output layer of 2 full-connected neurons (As output only has 2 class)
model.add(tf.keras.layers.Dense(2,activation='sigmoid'))

print(model.summary())


# Compile the network
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training the network
history = model.fit(normalized_train_ds,
                    epochs=10)

# inspecting loss
plt.plot(history.history['loss'], color='r', label='10 neurons in HL' )



# Reporting accuracy
result = model.evaluate(normalized_test_ds)
print("Test loss, Test accuracy : ", result)


# #### Model 2

# Simple Sequential structure
model = tf.keras.models.Sequential()

# Flatten 2D input image to a 1D vector
model.add(tf.keras.layers.Flatten(input_shape=(64,64,3)))
print(model.output_shape)

# Add 1st full connected layer (hidden layer in MLP terminology)
model.add(tf.keras.layers.Dense(100,activation='sigmoid'))

# Add 2nd full connected layer (hidden layer in MLP terminology)
model.add(tf.keras.layers.Dense(100,activation='sigmoid'))

# Add the output layer of 2 full-connected neurons (As output only has 2 class)
model.add(tf.keras.layers.Dense(2,activation='sigmoid'))

print(model.summary())


# Compile the network
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# training the network
history = model.fit(normalized_train_ds,
                    epochs=10)


# inspecting loss
plt.plot(history.history['loss'],  color='g', label='100 neurons in HL')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epoch vs Loss in different number of neurons")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

# Reporting accuracy
result = model.evaluate(normalized_test_ds)
print("Test loss, Test accuracy : ", result)


###### After increasing the neurons in the hidden layer it improves the accuracy




