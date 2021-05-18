import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
from tensorflow.keras import layers
import time
import sys
from tensorflow.keras.datasets import cifar10
from datetime import datetime
import cv2
from IPython import display

BUFFER_SIZE = 60000
BATCH_SIZE = 256

class DcGan:
    def __init__(self, colorDimension, imageWidth, imageHeight):
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 256
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4) 
        self.colorDimension = colorDimension   
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight 
        self.mod4 = int(self.imageWidth/4) 
        if self.imageWidth % 4 != 0:
            print("Error! 4 is not evenly divisible into image width: ", self.imageWidth) 

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.noise_dim = 100
        num_examples_to_generate = 16
        # You will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        self.seed = tf.random.normal([num_examples_to_generate, self.noise_dim])

        #make a directory with the results
        timeObj = datetime.now()
        timeNow = timeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
        self.resultsPath = "Results_" + str(timeNow)
        print("Making folder for results: ", self.resultsPath)
        os.mkdir(self.resultsPath)


    def makeModelAndTrain(self, train_dataset):
      self.generator = self.make_generator_model()

      noise = tf.random.normal([1, 100])
      generated_image = self.generator(noise, training=False)

      #plt.imshow(generated_image[0, :, :, 0], cmap='gray')

      self.discriminator = self.make_discriminator_model()
      decision = self.discriminator(generated_image)
      print (decision)

      checkpoint_dir = './training_checkpoints'
      self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
      self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                      discriminator_optimizer=self.discriminator_optimizer,
                                      generator=self.generator,
                                      discriminator=self.discriminator)

      EPOCHS = 50
      start = time.time()
      self.train(train_dataset, EPOCHS) 
      end = time.time()
      print("Total hub time: {:.1f}".format(end-start))   


    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(self.mod4*self.mod4*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((self.mod4, self.mod4, 256)))
        assert model.output_shape == (None, self.mod4, self.mod4, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, self.mod4, self.mod4, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, self.mod4 * 2, self.mod4 * 2, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.colorDimension, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.imageHeight, self.imageWidth, self.colorDimension)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[self.imageHeight, self.imageWidth, self.colorDimension]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output) 

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = self.generator(noise, training=True)

          real_output = self.discriminator(images, training=True)
          fake_output = self.discriminator(generated_images, training=True)

          gen_loss = self.generator_loss(fake_output)
          disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, epochs):
      #print("Training!")
      for epoch in range(epochs):
          start = time.time()

          #count = 1
          for image_batch in dataset:
              #print("About to train batch: ", count)
              self.train_step(image_batch)
              #print("Finished training batch: ", count)
              #count = count + 1

          # Produce images for the GIF as you go
          display.clear_output(wait=True)
          self.generate_and_save_images(self.generator,
                                  epoch + 1,
                                  self.seed)

          # Save the model every 15 epochs
          if (epoch + 1) % 15 == 0:
            self.checkpoint.save(file_prefix = self.checkpoint_prefix)

          print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

      # Generate after the final epoch
      display.clear_output(wait=True)
      self.generate_and_save_images(self.generator,
                              epochs,
                              self.seed)

    def generate_and_save_images(self, model, epoch, test_input):
      # Notice `training` is set to False.
      # This is so all layers run in inference mode (batchnorm).
      predictions = model(test_input, training=False)

      fig = plt.figure(figsize=(4, 4))

      for i in range(predictions.shape[0]):
          plt.subplot(4, 4, i+1)
          img = predictions[i, :, :, 0] * 127.5 + 127.5
          if self.colorDimension == 1:
                plt.imshow(img, cmap='gray')
          else:
                plt.imshow(img[...,::-1])
          plt.axis('off')

      fname = 'image_at_epoch_{:04d}.png'.format(epoch)
      fullpath = os.path.join(self.resultsPath, fname)
      plt.savefig(fullpath)
      #plt.show() 

    # Display a single image using the epoch number
    def display_image(epoch_no):
      return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

class Supporter:
    @staticmethod
    def getExampleImages():
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        return train_images, train_labels

    def batchAndShuffle(BATCH_SIZE, BUFFER_SIZE, train_images):
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return train_dataset


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    colorDimension = 1
    if len(sys.argv) < 2:
        print("Error, don't know which version you want to run!")
    elif len(sys.argv) < 3:
        pType = str(sys.argv[1])
        if pType == "example":
          print("Running example image version. To run a specific image, send in a path to the labels and training data!")
          train_images, train_labels = Supporter.getExampleImages()
          print("MNIST shape: ", train_images.shape)
          colorDimension = 1
        elif pType == "cifar10":
          (trainX, trainy), (testX, testy) = cifar10.load_data()
          print("CIFAR10 shape: ", trainX.shape)
          train_images = trainX
          train_labels = trainy
          colorDimension = 3
          #resize the images to be 28x28 to match with this model
          '''for i in range(len(train_images)):
              train_images[i] = np.resize(train_images[i], (28,28,3))
              #train_images[i] = cv2.resize(src=train_images[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
              print("After Shape: ", train_images[i].shape)'''

    imageHeight = train_images.shape[1]
    imageWidth = train_images.shape[2]
    dc = DcGan(colorDimension, imageWidth, imageHeight)
    train_images = Supporter.batchAndShuffle(dc.BATCH_SIZE, dc.BUFFER_SIZE, train_images)
    dc.makeModelAndTrain(train_images)

    '''else:
    train_images = []
    exampleMethod = False
    fileName = sys.argv[1] #labels
    folderName = sys.argv[2] #training data
    directory = os.fsencode(folderName)
    print("Running style transfer on folder: ", folderName)
    if (os.path.exists(directory) and os.path.isdir(directory)):
        count = 1
        for fileN in os.listdir(directory):
            if (os.path.exists(fileN) and os.path.isfile(fileN)):
                img = tf.keras.preprocessing.image.img_to_array(Image.open(fileN))
                train_images.append(img)
            else:
                print("Error! Unable to find content image sent in: ", fileName)

    #read in labels
    if (os.path.exists(fileName) and os.path.isfile(fileName)):
        #read in labels'''