from models.base_estimator import BaseEstimator

from features import lexical_per_sentence

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler


class GAN(BaseEstimator):

    def __init__(self):
        self.sent_features = 19
        self.max_sent = 15
        self.doc_shape = (self.max_sent * self.sent_features,)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        doc = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(doc)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.params = {
            'model_description': 'GAN',
            'predict': {
                'batch_size': 32,
                'verbose': 1
            }
        }

    def build_generator(self):
        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.doc_shape), activation='tanh'))
        model.add(Reshape(self.doc_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        doc = model(noise)

        return Model(noise, doc)

    def build_discriminator(self):
        model = Sequential()

        model.add(Dense(512, input_shape=self.doc_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        doc = Input(shape=self.doc_shape)
        validity = model(doc)

        return Model(doc, validity)

    def fit(self, train_x, train_y, train_positions):
        epochs = 30000
        batch_size = 32

        # Load the dataset

        train_x = np.array(lexical_per_sentence(train_x, self.max_sent))
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_x = self.scaler.fit_transform(train_x)

        train_x_true = np.array([train_x[index]
                                 for index, change in enumerate(train_y) if change])
        train_x_false = [train_x[index]
                         for index, change in enumerate(train_y) if not change]

        train_y = keras.utils.to_categorical(train_y, num_classes=2)

        half_batch = int(batch_size / 2)

        for epoch in range(1, epochs + 1):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            indices = np.random.randint(0, train_x_true.shape[0], half_batch)
            docs = train_x_true[indices]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_docs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(
                docs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(
                gen_docs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss))

    def predict(self, test_x):
        test_x = np.array(lexical_per_sentence(test_x, self.max_sent))
        test_x = self.scaler.transform(test_x)

        predictions = self.discriminator.predict(
            test_x, batch_size=32, verbose=1)

        return predictions.argmax(axis=-1)
