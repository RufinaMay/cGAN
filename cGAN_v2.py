import numpy as np
from keras.optimizers import Adam
from keras.layers import Input, Flatten, LeakyReLU, Dense, BatchNormalization, Reshape
from keras.models import Model, Sequential
from keras.datasets import mnist
from matplotlib import pyplot as plt
import cv2
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU,ReLU, Dropout, Input
import pickle
from matplotlib import pyplot as plt

class GAN():
    def __init__(self):
        self.IM_SIZE = 256
        self.CHANNELS = 3
        self.IM_SHAPE = (self.IM_SIZE, self.IM_SIZE, self.CHANNELS)
#         self.IMAGE_TO_TEST = cv2.imread(r'C:\Users\Rufina\Desktop\thesis\cGAN\data\Image\00000850\day\20151101_165511.jpg')
#         self.IMAGE_TO_TEST = cv2.resize(self.IMAGE_TO_TEST, (self.IM_SIZE,self.IM_SIZE))
#         self.DATA_FOLDER = r'C:\Users\Rufina\Desktop\thesis\cGAN\data\Image\\'

        self.DATA_FOLDER = '/content/gdrive/My Drive/Colab Notebooks/THESIS/cGAN/data/Image/'
        self.IMAGE_TO_TEST = cv.imread(f'{DATA_FOLDER}00000850/day/20151101_165511.jpg')
        self.IMAGE_TO_TEST = cv.resize(IMAGE_TO_TEST, (IM_SIZE, IM_SIZE))
        
        with open('day_paths.pickle', 'rb') as f:
            self.DAY_PATH = pickle.load(f)
        with open('night_paths.pickle', 'rb') as f:
            self.NIGHT_PATH = pickle.load(f)
        
        
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.Discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.Generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        day = Input(shape=self.IM_SHAPE)
        night = self.generator(day)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(night)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(day, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def Generator(self):
        G = Sequential()
        #ENCODER PART
        G.add( Conv2D(filters=64, kernel_size=2, strides=(2,2), input_shape=(256,256,3)))
        G.add(LeakyReLU(0.2))
        
        G.add( Conv2D(filters=128, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(LeakyReLU(0.2))
        
        G.add( Conv2D(filters=256, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(LeakyReLU(0.2))
        
        G.add( Conv2D(filters=512, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(LeakyReLU(0.2))
        
        G.add( Conv2D(filters=512, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(LeakyReLU(0.2))   
        
        G.add( Conv2D(filters=512, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(LeakyReLU(0.2))
        
        G.add( Conv2D(filters=512, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(LeakyReLU(0.2))
        
        G.add( Conv2D(filters=512, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(LeakyReLU(0.2))
        
        
        #DECODER PART
        G.add(Conv2DTranspose(filters=512, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(ReLU())
        G.add(Dropout(0.5))
        
        G.add(Conv2DTranspose(filters=512, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(ReLU())
        G.add(Dropout(0.5))
        
        G.add(Conv2DTranspose(filters=512, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(ReLU())
        G.add(Dropout(0.5))
        
        G.add(Conv2DTranspose(filters=512, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(ReLU())
        G.add(Dropout(0.5))
        
        G.add(Conv2DTranspose(filters=256, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(ReLU())
        G.add(Dropout(0.5))
        
        G.add(Conv2DTranspose(filters=128, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(ReLU())
        G.add(Dropout(0.5))
        
        G.add(Conv2DTranspose(filters=64, kernel_size=2, strides=(2,2)))
        G.add(BatchNormalization())
        G.add(ReLU())
        G.add(Dropout(0.5))
        
        G.add(Conv2DTranspose(filters=3, kernel_size=2, strides=(2,2), activation='tanh'))
        
        day = Input(shape=self.IM_SHAPE)
        night = G(day)
    
        return Model(day, night)

    def Discriminator(self):
        D = Sequential()
        #ENCODER PART
        D.add( Conv2D(filters=64, kernel_size=4, strides=(2,2), input_shape=(256,256,3)))
        D.add(LeakyReLU(0.2))
        
        D.add( Conv2D(filters=128, kernel_size=4, strides=(2,2)))
        D.add(LeakyReLU(0.2))
        D.add(BatchNormalization())
        
        D.add( Conv2D(filters=256, kernel_size=4, strides=(2,2)))
        D.add(LeakyReLU(0.2))
        D.add(BatchNormalization())
        
        D.add( Conv2D(filters=512, kernel_size=4, strides=(2,2)))
        D.add(LeakyReLU(0.2))
        D.add(BatchNormalization())
        
        D.add( Conv2D(filters=512, kernel_size=4, strides=(2,2)))
        D.add(LeakyReLU(0.2))
        D.add(BatchNormalization())
        
        D.add( Conv2D(filters=512, kernel_size=4, strides=(2,2)))
        D.add(LeakyReLU(0.2))
        D.add(BatchNormalization())
        
        D.add( Conv2D(filters=1, kernel_size=2, strides=(2,2), activation='sigmoid')) 
        D.add(Reshape((-1,)))
        
        image = Input(shape=self.IM_SHAPE)
        validity = D(image)
    
        return Model(image, validity)

    def Batch(self, day_paths, night_paths):
         N = len(day_paths)
         for i in range(N):
             day = cv2.imread(f'{self.DATA_FOLDER}{day_paths[i]}')
             night = cv2.imread(f'{self.DATA_FOLDER}{night_paths[i]}')
             day = cv2.resize(day, (self.IM_SIZE,self.IM_SIZE))
             night = cv2.resize(night,(self.IM_SIZE,self.IM_SIZE))
             yield day, night

    def train(self, epochs):
        N = len(self.DAY_PATH)
        for epoch in range(epochs):
            D_LOSS, G_LOSS = 0.,0.
            #SGD
            for d, n in self.Batch(self.DAY_PATH, self.NIGHT_PATH):
                   #day, night = d/255, n/255
                   day, night = (d - 127.5) / 127.5, (n - 127.5) / 127.5
                   gen_imgs = self.generator.predict(day[np.newaxis,:])
                   d_loss_real = self.discriminator.train_on_batch(night[np.newaxis,:], np.ones((1,1)))
                   d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((1,1)))
                   d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                   g_loss = self.combined.train_on_batch(day[np.newaxis,:], np.ones((1, 1)))
                   
                   D_LOSS += d_loss
                   G_LOSS += g_loss
            print(f'epoch: {epoch}, D_LOSS: {D_LOSS/N}, G_LOSS: {G_LOSS} ')
            
#             if epoch%10== 0:
            img = self.generator.predict(self.IMAGE_TO_TEST[np.newaxis,:])
            cv2.imwrite(f'{epoch}.jpg',img[0]* 127.5+ 127.5)
            
        img = self.generator.predict(self.IMAGE_TO_TEST[np.newaxis,:])
        plt.imshow(img[0]* 127.5+ 127.5)
             
               


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=500)
