from keras.layers import Dense, Dropout, Input, ReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST

(x_train, y_train), (_, _) = mnist.load_data()

# Normalize pixel values to [-1, 1]
x_train = (x_train.astype(np.float32) - 127.5) / 127.5

# Flatten images (28x28 -> 784)
x_train = x_train.reshape(x_train.shape[0], -1)



# Create Generator

def create_generator():
    """
    Builds the generator network.
    Input: 100-dimensional random noise vector
    Output: 784-dimensional vector (28x28 image with tanh activation)
    """
    model = Sequential()
    model.add(Dense(512, input_dim=100))
    model.add(ReLU())
    model.add(Dense(512))
    model.add(ReLU())
    model.add(Dense(1024))
    model.add(ReLU())
    model.add(Dense(784, activation="tanh"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0001, beta_1=0.5)
    )
    return model


# Create Discriminator

def create_discriminator():
    """
    Builds the discriminator network.
    Input: 784-dimensional vector (flattened image)
    Output: 1 (probability of being real/fake)
    """
    model = Sequential()
    model.add(Dense(1024, input_dim=784))
    model.add(ReLU())
    model.add(Dropout(0.4))

    model.add(Dense(512))
    model.add(ReLU())
    model.add(Dropout(0.4))

    model.add(Dense(256))
    model.add(ReLU())

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0001, beta_1=0.5)
    )
    return model



# Create Combined GAN Model

def create_gan(discriminator, generator):
    """
    Builds the combined GAN model by stacking generator and discriminator.
    While training GAN, discriminator weights are frozen.
    """
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss="binary_crossentropy", optimizer="adam")
    return gan



# Instantiate Models

generator = create_generator()
discriminator = create_discriminator()
gan = create_gan(discriminator, generator)

gan.summary()



# Training Loop

epochs = 50
batch_size = 256

for epoch in range(epochs):
    for _ in range(batch_size):
        # Generate random noise
        noise = np.random.normal(0, 1, [batch_size, 100])

        # Generate fake images from noise
        generated_images = generator.predict(noise)

        # Sample a random batch of real images
        real_images = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

        # Concatenate real and fake images
        x = np.concatenate([real_images, generated_images])

        # Labels: 1 for real, 0 for fake
        y_dis = np.zeros(batch_size * 2)
        y_dis[:batch_size] = 1

        # Train the discriminator
        discriminator.trainable = True
        discriminator.train_on_batch(x, y_dis)

        # Train the generator via GAN model
        noise = np.random.normal(0, 1, [batch_size, 100])
        y_gen = np.ones(batch_size)

        discriminator.trainable = False
        gan.train_on_batch(noise, y_gen)

    print(f"Epoch {epoch + 1}/{epochs} completed")

# Save Generator Weights
generator.save_weights("generator_model.h5")
