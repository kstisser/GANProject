import os
import tensorflow as tf
import tensorflow_hub as hub
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import PIL.Image
from PIL import Image
import time
import functools
from datetime import datetime

import sys
import os.path

#This builds a model that returns the style and content sensors
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = self.vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                        outputs[self.num_style_layers:])

        style_outputs = [self.gram_matrix(style_output)
                        for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict} 

    def vgg_layers(self, layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model 

    #this computes the gram matrix for the style calculation
    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)            

class Supporter:
    @staticmethod
    def getExampleImages():
        content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
        style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

        content_image = Supporter.load_img(content_path)
        style_image = Supporter.load_img(style_path)
        return content_image, style_image

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    @staticmethod
    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img           

class NeuralStyleTransfer:
    def __init__(self, content_image, style_images, resultsFolder):
        self.content_image = content_image
        self.style_images = style_images
        self.resultsFolder = resultsFolder
        self.total_variation_weight=30

        # Load compressed models from tensorflow_hub
        os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
        print("Loading model from tensorflow hub")
        self.hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

        #test vgg model
        print("Testing VGG to get top 5 predictions for content image")
        self.testVGGOnContentImg()

        # to optimize use a weighted combination of the two losses to get the total loss
        #self.style_weight=1e-2
        #self.content_weight=1e4     
        self.style_weight=1e-10
        self.content_weight=1e4   

    def runBothModels(self):
        count = 1
        for styleImage in self.style_images:
            #run hub method
            hubImg = self.tensorflowHubStyle(styleImage)
            hubName = os.path.join(self.resultsFolder, (str(count) + "_hub.png"))
            hubImg.save(hubName)

            #run self trained method
            self.opt = None
            selfTrainedImg = self.applyStyle(styleImage)            
            selfTrainedName = os.path.join(self.resultsFolder, (str(count) + ".png"))
            selfTrainedImg.save(selfTrainedName)

            count = count + 1


    def tensorflowHubStyle(self, style_image):
        #have Tensorflow hub generate the image
        start = time.time()
        stylized_image = self.hub_model(tf.constant(self.content_image), tf.constant(style_image))[0]
        styledImg = Supporter.tensor_to_image(stylized_image)
        end = time.time()
        print("Total hub time: {:.1f}".format(end-start))
        return styledImg

    def applyStyle(self, style_image, removeHighFrequency=True, verbose=False):
        start = time.time()
        #choose intermediate layers to represent the content and style of the image
        content_layers = ['block5_conv2'] 

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1']

        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)

        #calls the model class to get the gram matrix of the style layers and the content layers
        extractor = StyleContentModel(style_layers, content_layers)
        self.style_targets = extractor(style_image)['style']
        self.content_targets = extractor(self.content_image)['content']

        if verbose:
            results = extractor(tf.constant(self.content_image))
            print('Styles:')
            for name, output in sorted(results['style'].items()):
                print("  ", name)
                print("    shape: ", output.numpy().shape)
                print("    min: ", output.numpy().min())
                print("    max: ", output.numpy().max())
                print("    mean: ", output.numpy().mean())
                print()

            print("Contents:")
            for name, output in sorted(results['content'].items()):
                print("  ", name)
                print("    shape: ", output.numpy().shape)
                print("    min: ", output.numpy().min())
                print("    max: ", output.numpy().max())
                print("    mean: ", output.numpy().mean())

        #make tf variable same shape as image to use to optimize the image
        image = tf.Variable(self.content_image)

        #regularization loss, remove high freq
        if removeHighFrequency:
            tf.image.total_variation(image).numpy()

        #make optimizer- Adam or LBFGS works
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        epochs = 5
        steps_per_epoch = 30

        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_step(image, extractor)
                print(".", end='')
            #display.clear_output(wait=True)
            #display.display(Supporter.tensor_to_image(image))
            print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end-start))

        return Supporter.tensor_to_image(image)

    def testVGGOnContentImg(self):
        #test VGG network
        x = tf.keras.applications.vgg19.preprocess_input(self.content_image*255)
        x = tf.image.resize(x, (224, 224))
        vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
        prediction_probabilities = vgg(x)

        #get predictions from VGG
        predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
        print([(class_name, prob) for (number, class_name, prob) in predicted_top_5])
            
    @tf.function()
    def train_step(self, image, extractor):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = self.style_content_loss(outputs)
            loss += self.total_variation_weight*tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(self.clip_0_1(image))
        return image

    #the regularization loss associated with variation loss is the sum of squares of the values
    def total_variation_loss(self, image):
        x_deltas, y_deltas = self.high_pass_x_y(image)
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

    #decrease the high frequency artifacts by using an explicit regularization term
    #on the high frequency components of an image
    #for style transfer this is called the variation loss, and is comparable to using the sobel edge detector
    def high_pass_x_y(self, image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

        return x_var, y_var

    # to optimize use a weighted combination of the two losses to get the total loss
    def style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.num_content_layers
        loss = style_loss + content_loss
        return loss

    #keep pixel values between 0 and 1
    def clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


#######################################################################################################
#generate two methods of running:
#1. run the example images
#2. send in an image and have it converted into each image in the style folder, and saved
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    exampleMethod = True
    styleImages = []
    if len(sys.argv) < 3:
        #must be running with example as no path is sent in
        print("Running example image version. To run a specific image, send in a path to the content image and a folder to the style images!")
        content_image, style_image = Supporter.getExampleImages()
        styleImages.append(style_image)
        print("content image type")
        print(type(content_image))
    else:
        exampleMethod = False
        fileName = sys.argv[1]
        folderName = sys.argv[2]
        directory = os.fsencode(folderName)
        print("Running style transfer on image: ", fileName)
        if (os.path.exists(fileName) and os.path.isfile(fileName)):
            content_image = tf.keras.preprocessing.image.img_to_array(Image.open(fileName))
            content_image = tf.expand_dims(content_image,0)
        else:
            print("Error! Unable to find content image sent in: ", fileName)

        if (os.path.exists(directory) and os.path.isdir(directory)):
            count = 1
            for fileN in os.listdir(directory):
                print(count, " Adding image: ", fileN)
                img = tf.keras.preprocessing.image.img_to_array(Image.open(os.path.join(directory, fileN)))
                
                if (len(img.shape) < 4):
                    img = tf.expand_dims(img, 0)
                #print(img.shape)

                styleImages.append(img)
                count = count + 1
        else:
            print("Error! Unable to find the folder to your style images!")

    #make a directory with the results
    timeObj = datetime.now()
    timeNow = timeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    resultsPath = "Results_" + str(timeNow)
    print("Making folder for results: ", resultsPath)
    os.mkdir(resultsPath)

    nst = NeuralStyleTransfer(content_image, styleImages, resultsPath)
    nst.runBothModels()

