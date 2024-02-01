import pandas as pd
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input

# Load Data
def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Preprocessing
def preprocess_data(data):
    df = pd.DataFrame(data)
    max_value = max([max(seq) for seq in df['norm'].tolist() if len(seq) > 0])
    min_value = min([min(seq) for seq in df['norm'].tolist() if len(seq) > 0])
    df['norm'] = df['norm'].apply(lambda x: [(i - min_value) / (max_value - min_value) for i in x])
    norms_padded = pad_sequences(df['norm'].tolist(), padding='post', dtype='float32')
    return df, norms_padded, min_value, max_value

# GAN Creation
def create_generator(norms_shape):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(100,)))
    model.add(Dense(np.prod(norms_shape), activation='relu'))
    model.add(Reshape(norms_shape))
    return model

def create_discriminator(norms_shape):
    model = Sequential()
    model.add(Flatten(input_shape=norms_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_gan(generator, discriminator, norms_padded, epochs=2000):
    discriminator.compile(optimizer='adam', loss='mean_squared_error')
    z = Input(shape=(100,))
    norm_generated = generator(z)
    discriminator.trainable = False
    validity = discriminator(norm_generated)
    combined = Model(z, validity)
    combined.compile(optimizer='adam', loss='mean_squared_error')
    batch_size = 32
    half_batch = int(batch_size / 2)
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, norms_padded.shape[0], half_batch)
        norms_real = norms_padded[idx]
        noise = np.random.normal(0, 1, (half_batch, 100))
        norms_generated = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(norms_real, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(norms_generated, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
        print(f"{epoch}/{epochs} [D loss: {d_loss}] [G loss: {g_loss}]")
    
    return generator, discriminator


# Main
data = load_data('match_1.json')
df, norms_padded, min_value, max_value = preprocess_data(data)
generator = create_generator(norms_padded.shape[1:])
discriminator = create_discriminator(norms_padded.shape[1:])
trained_generator, _ = train_gan(generator, discriminator, norms_padded)

# Save model
trained_generator.save('generator_model.h5')

print("Model saved!")
