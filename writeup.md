# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Link to my [project code](https://github.com/cpierceworld/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image_distro]: ./examples/image_distro.png "Test Image Distribution"
[test_image1]: ./examples/stop_sign_200.png "Stop Sign"
[test_image2]: ./examples/thirty_200.png "30kph Sign"
[test_image3]: ./examples/slippery_200.png "Slippery Sign"
[test_image4]: ./examples/left_turn_200.png "Left Turn Sign"
[test_image5]: ./examples/construction_200.png "Construction Sign"
[test_image6]: ./examples/right_of_way_200.png "Right Of Way Sign"
[test_image7]: ./examples/hundred_thirty_200.png "130kph Sign"
[before_after_image1]: ./examples/before_after_1.png "Before/After Processing Training Data"
[before_after_image2]: ./examples/before_after_2.png "Before/After Processing Training Data"
[transform_image]: ./examples/transformed.png "Before/After Transformation"
[loss_image]: ./examples/loss_plot.png "Loss over EPOCHs"
[accuracy_image]: ./examples/accuracy_plot.png "Accuracy over EPOCHs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the python and nupmy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image_distro]

It is clear that there is not a uniform distribution of samples per image label.   I attempted to generate more data for the under-represented lables (with random transformations of the existing samples), but I found that only made the training perform worse.

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth, seventh, and eigth code cells of the IPython notebook.

From "Data Set Exploration and Visualization", I could see that man of the images were dark and with low contrast.  I rand the training data through a histogram equalization process using OpenCV's implementation of "Contrast Limited Adaptive Histogram Equalization" (CLACHE)]

The result was the training data being all of relatively equal brightness and contrast:

![alt text][before_after_image1]

![alt text][before_after_image2]

I kept the images as 3 channel color since color is an important part of identifying traffic signs.

I normalized the image pixel values from "0 to 255" to "-0.5 to 0.5". 

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data set was already split into Training/Validation/Test and I just used that existing split:
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630

As mentioned earlier, there is not a uniform distribution of samples per image label.

The sixth and seventh code cells of the IPython notebook contains the code for augmenting the data set. I initially generated more training data to try and increase the number of under-represented image labels (adding x3 for labels with 0-100 samples, x2 for 100-200 samples, and one extra for 200-300 samples).   This made the training data set much more uniform.

To do the aumenting, I took each image and used OpenCV to do a random -5 to 5 pixel horizonal and vertical translation, a random -12 to 12 degree rotation  I also did a random brightness shift.

Here is an example of an original image and an augmented image:

![alt text][transform_image]


However, in the end, I found that adding the augmented data to the trainig set actually _decreased_ accuraacy in the end, so I ended up not using the generated images.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer                 | Description                                   | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                             | 
| Convolution 5x5       | 1x1 stride, same padding, outputs 32x32x16    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 16x16x16                  |
| Convolution 3x3       | 1x1 stride, same padding, outputs 16x16x32    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 8x8x32                    |
| Fully connected       | input = 2048, output = 512                    |
| Fully connected       | input = 512, output = 128                     |
| Fully connected       | input = 128, output = 128                     |
| Softmax               | input = 128, output = 43                      |

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 
The code for training the model is located in the eleventh, twelfth, and thirteenth cells of the Ipython notebook.

My training included:
* 50% dropout between fully connected layers
* Atom Optimizer
* learning rate of 0.00025
* 100 EPOCHs
* batch size of 128.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100.0%
* validation set accuracy of 96.4%
* test set accuracy of 95.9%

Here is the graphs of the "loss" and the "accuracy" during training:
![alt text][loss_image] ![alt text][accuracy_image]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][test_image1] ![alt text][test_image2] ![alt text][test_image3] 
![alt text][test_image4] ![alt text][test_image5] ![alt text][test_image6] 
![alt text][test_image7] 

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
