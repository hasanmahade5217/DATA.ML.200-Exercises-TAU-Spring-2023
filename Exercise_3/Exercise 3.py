# Importing necessary libraries

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Data Set location in the local machine
dataset_dir = 'E:/OneDrive - TUNI.fi/Tampere University (MSc in CS-DS)/Year 2/Period 4/DATA.ML.200 Pattern Recognition and Machine Learning/Exercise_3/Data/GTSRB_subset_2'


# Defining some properties for the image
batch_size = 32
img_height = 64
img_width = 64


#Split data into two parts - 80% for training and 20% for testing.

# Training Data
train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode="categorical")

## Testing Data
test_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode="categorical")

# Checking the class names 
class_names = train_ds.class_names
print(class_names)


# Checking image properties
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# Normalization
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


# ### Model:

# declare input shape 
input = tf.keras.Input(shape=(64,64,3))

# Block 1 (convolution)
conv1 = tf.keras.layers.Conv2D(10, 3, strides=2, activation="relu")(input)

# Maxpooling 1
max_p1 = tf.keras.layers.MaxPooling2D(2)(conv1)

# Block 2 (convolution)
conv2 = tf.keras.layers.Conv2D(10, 3, strides=2, activation="relu")(max_p1)

# Maxpooling 2
max_p2 = tf.keras.layers.MaxPooling2D(2)(conv2)

# Block 3 (full connected)
fc = tf.keras.layers.Flatten()(max_p2)

# Finally, we add a classification layer.
output = tf.keras.layers.Dense(2, activation="sigmoid")(fc)

# bind all
cnn_model = tf.keras.Model(input, output)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

cnn_model.compile(loss=loss_fn, optimizer="SGD", metrics=["accuracy"])
cnn_model.summary()


# training
history = cnn_model.fit(normalized_train_ds, epochs=20)

# Ploting training loss
plt.plot(history.history['loss'])


plt.show()

# Testing and Accuracy
result = cnn_model.evaluate(normalized_test_ds)
print("Test loss, Test accuracy : ", result)




