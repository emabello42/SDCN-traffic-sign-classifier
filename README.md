## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I use what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. A model is trained and validated so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, this is evaluated on images of German traffic signs found on the web.


The Project
---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/example.png "Example"
[image3]: ./examples/gray.png "Grayscaling"
[image4]: ./examples/normalize.png "Normalization"
[image5]: ./examples/rotate_img.png "Rotation"
[image6]: ./examples/translate_img.png "Translation"
[image7]: ./examples/scale_img.png "Scaling"
[image8]: ./examples/chbrightness_img.png "Change in Brightness"
[image9]: ./examples/visualization_ext.png "Visualization Augmented training set"
[image10]: ./test_images/00764.ppm "Traffic Sign 1"
[image11]: ./test_images/00806.ppm "Traffic Sign 2"
[image12]: ./test_images/00781.ppm "Traffic Sign 3"
[image13]: ./test_images/01010.ppm "Traffic Sign 4"
[image14]: ./test_images/00450.ppm "Traffic Sign 5"

[image15]: ./examples/traffic_sign_featuremap1.png "Example 1 - Feature map 1"
[image16]: ./examples/traffic_sign_featuremap2.png "Example 1 - Feature map 2"
[image17]: ./examples/other_featuremap1.png "Example 2 - Feature map 1"
[image18]: ./examples/other_featuremap2.png "Example 2 - Feature map 2"
[image19]: ./examples/animal.jpeg "Example 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate the number of classes in the traffic signs data set. The other values can be obtained using python.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training, validation and test data sets. They are histograms showing the number of samples that there are for each traffic sign class. Although the training, validation and test data sets have similar distributions, there is a lack of samples for some traffic sign classes, e.g. the traffic sign "Speed limit (20km/h)" (class ID: 0).

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images (in the training, validation and test set) to grayscale because the training process turns out to be faster than using color information while the accuracy is better. After that, a  normalization process is applied, which consists in the following operation: image_data-128/128, which changes the range of values from [0,255] to [-1, 1).

Here is an example of a traffic sign image before and after grayscaling:
![alt text][image2]
![alt text][image3]

And after normalization:
![alt text][image4]

In order to reduce the overfitting in the model, the training set was augmented applying different kinds of transformations to the original images: translation, scaling, rotation and changes in the brightness. The transformations and their parameters were selected randomly with values within the following ranges:
* Translation:
	* Displacement along the x axis, **fx**:  [-2, 2]
	* Displacement along the y axis, **fy**: [-2, 2]
		![alt text][image6]
* Rotation in the range [-35°, 35°]  
		![alt text][image5]
* Scaling between [0.9, 1.1] of the original size
		![alt text][image7]
* Changes in brightness between [-100, 100]
		![alt text][image8]

On the other hand, the number of samples for each traffic sign class is balanced, applying the transformations at random until reaching the same amount of samples in each class (6030, which is three times the number of samples in the class with more samples in the original training set). The distribution of samples in the augmented training set is as follows:
![alt text][image9]

While the original training set has  34799 samples, the augmented one has 259290 samples.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 2x2 kernel size, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, 2x2 kernel size, outputs 5x5x16 	|
| Fully connected	1	| outputs 120        							|
| Dropout				| outputs 120        							|
| Fully connected	2	| outputs 84        							|
| Dropout				| outputs 84        							|
| Fully connected	3	| outputs 43        							|
| Softmax				| outputs 43        							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used as loss function the cross entropy of the outputs of the softmax layer in the model with the one hot encoding of the  correct label. The Adam optimizer *tf.train.AdamOptimizer* uses this loss function to train the model with the following hyperparameters:
* Learning rate = 0.0009
* Number of Epochs = 100
* Batch size = 128
* Dropout (the probability of keeping the output of the first two fully connected layers): 0.6

On the other hand, the normal distribution used to initialize the weight and bias arrays in the neural network architecture has a mean of 0 and standart deviaton of 0.1.



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 0.971
* Validation set accuracy of 0.970 
* Test set accuracy of 0.947
These results were calculated in the cell 13  of the IPython notebook with the function *evaluate(X, Y)* (in case of the training and validation set is taken the value obtained after the last epoch).

The LeNet neural network architecture was used as base for this model. I think this model is relevant for traffic sign classification because of its ability to automatically learn the invariant features necessary to recognize different elements in images that allow to determine the type of traffic sign.

The original LeNet architecture was changed in the following aspects:
* In this case, the output of the last layer is a softmax function which outputs 43 values corresponding to the probabilities of belonging to any of the 43 traffic sign classes.
* A dropout of 0.6 was applied to the outputs of the first two fully connected layers. This contributed to reducing the overfitting.

The dropout operation and the augmentation of the training set contributed to reducing considerably the overfitting.

The training process was executed over the augmented training set. After every epoch was reported the accuracy over the training and the validation set in order to figure out if there is overfitting (training accuracy better than validation accuracy) or underfitting (the training accuracy is very low). In this case, I think there is underfitting, because the training and validation accuracy are very close to each other, but the complexity of the network could be improved to obtain results closer to 0.99 in both cases.

After evaluating the accuracy in the training and validation sets with different hyperparameters and transformations to extend and balance the training set, we can see if the model is really working well after evaluating the accuracy of the model in the test set, which is 0.947.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14]

These images present some difficulties to be correctly classified:
1) The first image has a tree branch in front.
2) The second image is blurred and rotated.
3) The third image is very dark.
4) The fourth image is very blurred. Indeed is difficult for a human to distinguish between a speed limit of 100km/h and 120 km/h,
5) The fifth image is blurred and there is a variation in the light over the traffic sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road     	| Priority road   								| 
| Traffic signals     	| Wild animals crossing 						|
| Keep left				| Keep left										|
| Speed limit 120 km/h	| Speed limit 120 km/h					 		|
| No passing			| No passing      								|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.7%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the IPython notebook.

For the first image, the model is 100% sure that this is a *Priority road* sign (probability of 1) and it does contain that.

For the second image, the model is 23.71% sure that is this is a *Wild animals crossing* sign, but actually it contains a *traffic signal*.The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .2371         		| Wild animals crossing  						| 
| .2025     			| Keep left 									|
| .1687					| Traffic signals								|
| .1308	      			| Dangerous curve to the right					|
| .0974				    | Road work     								|

For the third image, the model is 99.93% sure that is this is a *Keep left* sign, and it does contain a **Keep left* sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9993         		| Keep left  									| 
| .0005     			| Stop 											|
| .0001					| Wild animals crossing 						|
| .0000	      			| Vehicles over 3.5 metric tons prohibited		|
| .0000				    | Speed limit 70km/h    						|

For the fourth image, the model is 100% sure that this is a *Speed limit 120 km/h* sign (probability of 1) and it does contain that.

For the fifth image, the model is 90.21% sure that is this is a *No passing* sign, and it does contain a **No passing* sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9021         		| No passing  									| 
| .0639     			| No passing for vehicles over 3.5 metric tons	|
| .0155					| Vehicles over 3.5 metric tons prohibited 		|
| .0092	      			| End of no passing								|
| .0058				    | End of all speed and passing limits			|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The following  images show the RELU activations after the two convolutional layers when the input image is a traffic sign (Speed limit 80km/h):
Input image:
![alt text][image2]

Activations:
![alt text][image15]
![alt text][image16]

The next images show the same corresponding activations, but when the input image does not contain a traffic sign:
Input image:
![alt text][image19]

Activations:
![alt text][image17]
![alt text][image18]

From these two examples can be seen that the model distinguishes the borders and the number in the traffic sign, but in the other example image is not possible to extract any relevant feature.