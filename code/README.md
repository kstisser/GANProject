I have followed several tensorflow tutorials, implemented them, and this shows how you run them:

Neural Style Transfer
Referenced tutorial: https://www.tensorflow.org/tutorials/generative/style_transfer
Runs 2 different methods:
1. using the pictures from the tutorial
python neuralStyleTransfer.py
2. using pictures sent in
python neuralStyleTransfer.py <content image filename> <Folder that contains syle images>
example:
python neuralStyleTransfer.py bikes.jpg StyleImages

This runs through and performs style transfer using 2 methods:
1. using tensorflow hub model
2. using the self trained model based on VGG19

It is evident that method 2 needs a lot of tuning, and method 1, while very fast is suboptimal. 

