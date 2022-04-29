from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys

import numpy as np

class DCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # optimizer 정의
        optimizer = Adam(0.0002, 0.5)

        # 판별자 정의
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # 생성자 정의
        self.generator = self.build_generator()

        # 생성자의 입력(noise)과 출력(img)
        noise = Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        # 최종 모델(결합된 모델)의 경우 생성자만 학습
        self.discriminator.trainable = False

        # 판별자는 생성된 이미지를 입력으로 받아, 유효성을 검사
        validity = self.discriminator(img)

        # 최종 모델(결합된 모델)은 생성자와 판별자를 쌓아서 만든 모델임(GAN)
        self.model = Model(noise, validity)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # 생성자 정의 함수
    def build_generator(self):

        model = Sequential()
        ##입력 받는 부분
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))

      
        model.add(Conv2D(100, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(50, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(25, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

 
    def build_discriminator(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))

    

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

  
    def train(self, epochs, batch_size=128, sample_interval=50):

      
        (X_train, _), (_, _) = fashion_mnist.load_data()

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths

        D_loss_list = []
        G_loss_list = []
        for epoch in range(1,epochs+1):

          
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

          
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

         
            gen_imgs = self.generator.predict(noise)

       
            X_fake_and_real = tf.concat([gen_imgs, imgs], axis=0)
            fake_and_valid_label = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            self.discriminator.tainable = True
            d_loss = self.discriminator.train_on_batch(X_fake_and_real, fake_and_valid_label)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

        
            valid_label = tf.constant([[1.]] * batch_size)
            self.discriminator.tainable = False
            g_loss = self.model.train_on_batch(noise, valid_label)
            G_loss_list.append(g_loss)
            D_loss_list.append(d_loss[0])
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        self.plotLoss(G_loss_list, D_loss_list, epoch)

  
    def plotLoss(self, G_loss, D_loss, epoch):
        plt.figure(figsize=(10, 8))
        plt.plot(D_loss, label='Discriminitive loss')
        plt.plot(G_loss, label='Generative loss')
        plt.xlabel('BatchCount')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_graph/gan_loss_epoch_%d.png' % epoch)

  
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

   
    def load_model(self, model_path='saved_model/model.h5'):
        print('\nload model : \"{}\"'.format(model_path))
        self.model = tf.keras.models.load_model(model_path)

   
    def save_model(self, model_path='saved_model/model.h5'):
        print('\nsave model : \"{}\"'.format(model_path))
        self.model.save(model_path)

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=5000, batch_size=32, sample_interval=200)
    dcgan.save_model()
