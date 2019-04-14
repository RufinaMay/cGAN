# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:08:11 2019

@author: Rufina
"""
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU,ReLU, BatchNormalization, Dropout, Input
from keras.models import Sequential, Model
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop, Adam
import cv2 as cv
import numpy as np
import pickle
from matplotlib import pyplot as plt

IM_SIZE = 256
IMAGE_TO_TEST = cv.imread(r'C:\Users\Rufina\Desktop\thesis\cGAN\data\Image\00000850\day\20151101_165511.jpg')
IMAGE_TO_TEST = cv.resize(IMAGE_TO_TEST, (IM_SIZE, IM_SIZE))
DATA_FOLDER = r'C:\Users\Rufina\Desktop\thesis\cGAN\data\Image\\'
"""
IM_SIZE = 256
DATA_FOLDER = '/content/gdrive/My Drive/Colab Notebooks/THESIS/cGAN/data/Image/'
IMAGE_TO_TEST = cv.imread(f'{DATA_FOLDER}00000850/day/20151101_165511.jpg')
IMAGE_TO_TEST = cv.resize(IMAGE_TO_TEST, (IM_SIZE, IM_SIZE))
"""
with open('day_paths.pickle', 'rb') as f:
    DAY_PATH = pickle.load(f)
with open('night_paths.pickle', 'rb') as f:
    NIGHT_PATH = pickle.load(f)
def Generator():
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
    #G.add(Dropout(0.5))
    
    G.add(Conv2DTranspose(filters=512, kernel_size=2, strides=(2,2)))
    G.add(BatchNormalization())
    G.add(ReLU())
    #G.add(Dropout(0.5))
    
    G.add(Conv2DTranspose(filters=512, kernel_size=2, strides=(2,2)))
    G.add(BatchNormalization())
    G.add(ReLU())
    #G.add(Dropout(0.5))
    
    G.add(Conv2DTranspose(filters=512, kernel_size=2, strides=(2,2)))
    G.add(BatchNormalization())
    G.add(ReLU())
    G.add(Dropout(0.5))
    
    G.add(Conv2DTranspose(filters=256, kernel_size=2, strides=(2,2)))
    G.add(BatchNormalization())
    G.add(ReLU())
    #G.add(Dropout(0.5))
    
    G.add(Conv2DTranspose(filters=128, kernel_size=2, strides=(2,2)))
    G.add(BatchNormalization())
    G.add(ReLU())
    #G.add(Dropout(0.5))
    
    G.add(Conv2DTranspose(filters=64, kernel_size=2, strides=(2,2)))
    G.add(BatchNormalization())
    G.add(ReLU())
    #G.add(Dropout(0.5))
    
    G.add(Conv2DTranspose(filters=3, kernel_size=2, strides=(2,2), activation='tanh'))
    
    day = Input(shape=(IM_SIZE, IM_SIZE, 3))
    night = G(day)

    return Model(day, night)


def Discriminator():
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
    
    image = Input(shape=(IM_SIZE, IM_SIZE, 3))
    validity = D(image)

    return Model(image, validity)


def create_model():
    optimizer = Adam(0.0002, 0.5)
    #BUILD DISCRIMINATOR
    discriminator = Discriminator()
    discriminator.compile(loss='binary_crossentropy', 
        optimizer=optimizer,
        metrics=['accuracy'])

    # BUILD GENERATOR
    generator = Generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # The generator takes day as input and generates night
    day = Input(shape=(IM_SIZE, IM_SIZE, 3))
    night = generator(day)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The valid takes generated images as input and determines validity
    valid = discriminator(night)

    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity 
    combined = Model(day, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return discriminator, generator, combined

def Batch(day_paths, night_paths):
     N = len(day_paths)
     for i in range(N):
         day = cv.imread(f'{DATA_FOLDER}{day_paths[i]}')
         night = cv.imread(f'{DATA_FOLDER}{night_paths[i]}')
         day = cv.resize(day, (IM_SIZE, IM_SIZE))
         night = cv.resize(night, (IM_SIZE, IM_SIZE))
         yield day, night
   
def train(epochs=10, save_interval=50):
    discriminator, generator, combined = create_model()
    N = len(DAY_PATH)
    for epoch in range(epochs):
        D_LOSS, G_LOSS = 0.,0.
        for day, night in Batch(DAY_PATH, NIGHT_PATH):
            #TRAIN DISCRIMINATOR
            fake_night = generator(day)
            d_loss_real = discriminator.train_on_batch(night[np.newaxis,:], np.ones((1,1,1,1)))
            d_loss_fake = discriminator.train_on_batch(fake_night, np.zeros((1,1,1,1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            #TRAIN GENERATOR
            g_loss = combined.train_on_batch(day,  np.ones((1,1,1,1)))
            D_LOSS += d_loss
            G_LOSS += g_loss
        D_LOSS/=N
        G_LOSS/=N
        print(f'g loss {G_LOSS}, d loss {D_LOSS}')
        
        image = generator.predict(IMAGE_TO_TEST[np.newaxis,:])
        print(image[0])
        
        plt.imshow(image[0])
            

"""
#DISCRIMINATOR MODEL
def Discriminator_model():
    optimizer = RMSprop(lr=0.08)
    DiscriminatorModel = Sequential()
    DiscriminatorModel.add(Discriminator())
    DiscriminatorModel.compile(loss='binary_crossentropy', optimizer=optimizer)
    return DiscriminatorModel

#ADVERSARIAL MODEL
def Adversarial_model():
    optimizer = RMSprop(lr=0.08, clipvalue=1.0, decay=3e-8)
    AdversarialModel = Sequential()
    AdversarialModel.add(Generator())
    Dis = Discriminator()
    Dis.trainable = False
    AdversarialModel.add(Dis)
    AdversarialModel.compile(loss='binary_crossentropy', optimizer=optimizer)
    return AdversarialModel

def train_epoch(Day, Night, Discr, Gener, Adv):
    true_label = np.ones((1,1,1,1))
    fake_label = np.zeros((1,1,1,1))
    
    #TRAIN DISCRIMINATOR ON REAL DATA
    d_loss_real = Discr.train_on_batch(Night[np.newaxis,:], true_label)
    #TRAIN DISCRIMINATOR ON FAKE DATA
    fake_data = Gener.predict(Day[np.newaxis,:])
    d_loss_fake = Discr.train_on_batch(fake_data, fake_label)
    d_loss = 0.5 * (d_loss_real+ d_loss_fake)
    #TRAIN GENERATOR
    g_loss = Adv.train_on_batch(Day[np.newaxis,:], true_label)
    
    return d_loss, g_loss
    
def train(epochs = 10):
    #BATCH LENTH
    N = len(DAY_PATH)
    #MODELS
    Discr = Discriminator_model()
    Gener = Generator()
    Adv = Adversarial_model()
    for epoch in range (epochs):
        D_LOSS, G_LOSS = 0.,0.
        #SGD
        for day, night in Batch(DAY_PATH, NIGHT_PATH):
            d_loss, g_loss = train_epoch(day, night, Discr, Gener, Adv)
            D_LOSS += d_loss
            G_LOSS += g_loss
            print("ok")
        #PRINT LOSSES
        D_LOSS/=N
        G_LOSS/=N
        print(f'd_loss {D_LOSS}, g loss: {G_LOSS}')
        
        image = Gener.predict(IMAGE_TO_TEST[np.newaxis,:])
        print(image[0])
        cv.imshow('image',image[0])
        cv.waitKey(0)
        cv.destroyAllWindows()
"""        
        
train()




