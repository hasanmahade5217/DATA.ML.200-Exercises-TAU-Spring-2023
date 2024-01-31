#importing necessary libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

accuracies =[]
accuracies_labels = ['Clean', 'Noisy', 'Denoised','Noisy after Traning']


# Task 2-----------------------------------------------------------------------

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normalize data
train_images = train_images / 255.0
test_images = test_images / 255.0

# inspacting...
print (train_images.shape)
print (test_images.shape)


# adding noise to the original image...
noise_factor = 0.2
train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape) 
test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape) 

# Make sure values still in (0,1)
train_imagse_noisy = tf.clip_by_value(train_images_noisy, clip_value_min=0., clip_value_max=1.)
test_images_noisy = tf.clip_by_value(test_images_noisy, clip_value_min=0., clip_value_max=1.)


# show images 

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(test_images_noisy[i]))
    plt.gray()
plt.show()

# Task 3-----------------------------------------------------------------------

# declare input shape 
input = tf.keras.Input(shape=(28,28,1))

# Block 1 (convolution)
conv1 = tf.keras.layers.Conv2D(32, 3, strides=1, activation="relu")(input)


# Block 2 (convolution 2)
conv2 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(conv1)

# Block 3 (full connected9)
fc = tf.keras.layers.Flatten()(conv2)
fc = tf.keras.layers.Dense(10)(fc)

# Finally, we add a classification layer.
output = tf.keras.layers.Dense(10, activation="softmax")(fc)

# bind all
cnn_model = tf.keras.Model(input, output)


# This loss takes care of one-hot encoding (see https://keras.io/api/losses/)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

cnn_model.compile(loss=loss_fn, optimizer="adam", metrics=["accuracy"])
cnn_model.summary()

# training
history = cnn_model.fit(train_images, train_labels, epochs=1)

# Task 4-----------------------------------------------------------------------

test_loss, test_acc = cnn_model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy(Clean Images):', test_acc)
accuracies.append(test_acc)

# Task 5-----------------------------------------------------------------------
test_loss, test_acc = cnn_model.evaluate(test_images_noisy,  test_labels, verbose=2)
print('\nTest accuracy(Noisy Images):', test_acc)
accuracies.append(test_acc)


# Task 6-----------------------------------------------------------------------
# denoising autoencoder

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
          tf.keras.layers.Input(shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
          tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
          tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
          tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
          tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])
        
      
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Denoise()

autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

autoencoder.fit(train_images_noisy, train_images,
                epochs=1,
                shuffle=True,
                validation_data=(test_images_noisy, test_images))

#autoencoder.encoder.summary()
#autoencoder.decoder.summary()

# encoded and decoded images
encoded_imgs = autoencoder.encoder(test_images).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# show images after reconstruction
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(test_images_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()

# Task 7-----------------------------------------------------------------------
test_loss, test_acc = cnn_model.evaluate(decoded_imgs,  test_labels, verbose=2)
print('\nTest accuracy(Denoised Images):', test_acc)
accuracies.append(test_acc)


# Task 8-----------------------------------------------------------------------
#training with noisy images
history = cnn_model.fit(train_images_noisy, train_labels, epochs=1)

# testing with noisy image with the newly trained model
test_loss, test_acc = cnn_model.evaluate(test_images_noisy,  test_labels, verbose=2)
print('\nTest accuracy(Noisy Images):', test_acc)
accuracies.append(test_acc)

# ploting accuracies
plt.bar(accuracies_labels, accuracies)
plt.xlabel("Image Types")
plt.ylabel("Accuracy")
plt.title("Accuracy of the model in differnt settings")
plt.show()
