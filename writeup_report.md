# **Traffic Sign Recognition** 

## Writeup

### Deep Learning using Convolutional Neural Networks (CNN) and Tensorflow

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the training/validation/testing data sets from the supplied pickle files
* Explore, summarize and visualize these data sets
* Preprocess the images inside these data sets before using it inside CNN
* Design, train and test a model architecture based on LeNet
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Histogram"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/1.png "Traffic Sign 1"
[image4]: ./examples/2.png "Traffic Sign 2"
[image5]: ./examples/3.png "Traffic Sign 3"
[image6]: ./examples/4.png "Traffic Sign 4"
[image7]: ./examples/5.png "Traffic Sign 5"

## Rubric Points

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The writeup / README file provided is this file. Also, here is the link to my [project code](https://github.com/asilx/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The summary statistics of the traffic
signs data sets is as follows:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many instances the training data contains for each traffic sign type.

![The histogram depicting the training instances][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I believe additional color channels does not gain more information to CNN as the traffic signs are also distinguishable without color channels. 

Then, in order to compansate variances in brightness etc, I use openCV's histogram equalizer.

Thereafter, I normalized the image data because it is suggested in the course that CNNs perform better if the data lies around zero in every dimension.

I decided to augment the data by zooming in and out and rotating because the dataset size is limited and contains images from different angles and different distances. For this manner, I use keras' ImageDataGenerator with the parameters: width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10. 

Using this generator, I created 500 batches on-the-fly for each epoch.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 grayscale image   						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x80 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x80 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x80 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x80 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 8x8x40 	|
| RELU					|												|
| Max pooling           | 2x2 stride,  outputs 4x4x40   				|
| Flatten               | outputs 640                    				|
| Fully connected		| outputs 192  									|
| Fully connected		| outputs 120  with dropout (0.5)			    |
| Fully connected		| outputs 43  									|
| Softmax				| one hot encoded probabilities      			|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer.  My learning rate is 0.001. mu and sigma for convolutions are respectively.
Number of epochs is 30. Batch size is 128

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.985 
* test set accuracy of 0.962

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First, I have tried LeNet since it is a well-known architecture and we learnt it in lecture

* What were some problems with the initial architecture?
It was too simplistic for traffic sign recognition and it did not go beyond the validation accuracy of 0.87 in my case.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I increase the third dimension of each output in the convolutions in order to deduce filters for catching different characteristics of traffic signs. I used max pooling after 2 5x5 convolutions and 2 3x3 convolutions in order to reduce overfitting and to decrease the size of the output. In the second fully connected layer, I used dropout to reduce overfitting and enable learning redundant representations.


* Which parameters were tuned? How were they adjusted and why?
I increased the number of epochs to better train the network. Also, I augment the data to increase variance in the training set. 


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I used 4 different convolution layers with relatively high number of convolutions in order to catch the different characteristics of each sign. Between convolutions, I used max pooling layers to reduce overfitting and to decrease the size of the output.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Turn left ahead][image3] ![Road work][image4] ![Slippery Road][image5] 
![No entry][image6] ![Bumpy road][image7]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn left ahead 		| Turn left ahead								| 
| Road work    			| Road work										|
| Slippery Road			| Slippery Road									|
| No entry	      		| No entry  					 				|
| Bumpy Road			| Bumpy Road         							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This yields a perfect score.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all images, the model is almost completely sure (with a probabilty > 0.9) in its predictions and all of its predictions are correct!

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999999285			| Turn left ahead								| 
| .969719708   			| Road work										|
| .9049663				| Slippery Road									|
| .999881148			| No entry   					 				|
| .999085307		    | Bumpy Road      							    |




