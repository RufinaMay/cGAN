import numpy as np
from keras.optimizers import Adam
from keras.layers import Input, LeakyReLU, BatchNormalization, Reshape
from keras.models import Model, Sequential
from matplotlib import pyplot as plt
import cv2
from keras.layers import Conv2D, Conv2DTranspose, ReLU, Dropout
import pickle

class GAN():
    def __init__(self):
        self.IM_SIZE = 256
        self.CHANNELS = 3
        self.IM_SHAPE = (self.IM_SIZE, self.IM_SIZE, self.CHANNELS)
#         self.IMAGE_TO_TEST = cv2.imread(r'C:\Users\Rufina\Desktop\thesis\cGAN\data\Image\00000850\day\20151101_165511.jpg')
#         self.IMAGE_TO_TEST = cv2.resize(self.IMAGE_TO_TEST, (self.IM_SIZE,self.IM_SIZE))
#         self.DATA_FOLDER = r'C:\Users\Rufina\Desktop\thesis\cGAN\data\Image\\'
       
        self.DATA_FOLDER = '/content/gdrive/My Drive/Colab Notebooks/THESIS/cGAN/data/Image/'
        self.IMAGE_TO_TEST = cv2.imread(f'{self.DATA_FOLDER}00000850/day/20151101_165511.jpg')
        self.IMAGE_TO_TEST = cv2.resize(self.IMAGE_TO_TEST, (self.IM_SIZE, self.IM_SIZE))
    
        with open('day_paths.pickle', 'rb') as f:
            self.DAY_PATH = pickle.load(f)
        with open('night_paths.pickle', 'rb') as f:
            self.NIGHT_PATH = pickle.load(f)
        
        
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.Discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.Generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        day = Input(shape=self.IM_SHAPE)
        night = self.generator(day)

        self.discriminator.trainable = False

        valid = self.discriminator(night)

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
        G.summary()
        
  
        day = Input(shape=self.IM_SHAPE)
        night = G(day)
    
        return Model(day, night)

    def Discriminator(self):
        D = Sequential()
        #ENCODER PART
        D.add( Conv2D(filters=64, kernel_size=4, strides=(2,2), input_shape=(self.IM_SIZE,self.IM_SIZE,3)))
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
        D.summary()
        image = Input(shape=self.IM_SHAPE)
        validity = D(image)
    
        return Model(image, validity)
    
    def image_normalization_mapping(self, image, from_min, from_max, to_min, to_max):
      """
      Map data from any interval [from_min, from_max] --> [to_min, to_max]
      Used to normalize and denormalize images
      """
      from_range = from_max - from_min
      to_range = to_max - to_min
      scaled = np.array((image - from_min) / float(from_range), dtype=float)
      return to_min + (scaled * to_range)
    
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
                   day = self.image_normalization_mapping(d, 0, 255, -1, 1)
                   night = self.image_normalization_mapping(n, 0, 255, -1, 1)
                   gen_imgs = self.generator.predict(day[np.newaxis,:])
                   d_loss_real = self.discriminator.train_on_batch(night[np.newaxis,:], np.ones((1,1)))
                   d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((1,1)))
                   d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                   g_loss = self.combined.train_on_batch(day[np.newaxis,:], np.ones((1, 1)))
                   
                   D_LOSS += d_loss
                   G_LOSS += g_loss
            print(f'epoch: {epoch}, D_LOSS: {D_LOSS/N}, G_LOSS: {G_LOSS} ')
            
            print("okay")
            img = self.image_normalization_mapping(self.IMAGE_TO_TEST, 0, 255, -1, 1)
            img = self.generator.predict(img[np.newaxis,:])
            img = self.image_normalization_mapping(img[0], -1, 1, 0, 255).astype('uint8')
            plt.imshow(img)
            plt.show()
            
        
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=500)
