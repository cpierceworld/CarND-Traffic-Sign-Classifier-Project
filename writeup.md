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
[test_image4]: ./examples/left_turn_200.png "Right Turn Sign"
[test_image5]: ./examples/construction_200.png "Road Work Sign"
[test_image6]: ./examples/right_of_way_200.png "Right Of Way Sign"
[test_image7]: ./examples/hundred_thirty_200.png "130kph Sign"
[before_after_image1]: ./examples/before_after_1.png "Before/After Processing Training Data"
[before_after_image2]: ./examples/before_after_2.png "Before/After Processing Training Data"
[transform_image]: ./examples/transformed.png "Before/After Transformation"
[loss_image]: ./examples/loss_plot.png "Loss over EPOCHs"
[accuracy_image]: ./examples/accuracy_plot.png "Accuracy over EPOCHs"
[predict_image1]: ./examples/predict_stop_sign.png "Prediction Stop Sign"
[predict_image2]: ./examples/predict_30kph.png "Prediction 30kph Sign"
[predict_image3]: ./examples/predict_sippery.png "Prediction Slippery Sign"
[predict_image4]: ./examples/predict_right_turn.png "Prediction Right Turn Sign"
[predict_image5]: ./examples/predict_road_work.png "Prediction Road Work Sign"
[predict_image6]: ./examples/predict_right_of_way.png "Prediction Right Of Way"
[predict_image7]: ./examples/predict_130kph.png "Prediction 130kph Sign"


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

I started with the LeNet architecture which is a well known architecture for classification problems and that I had readily available.

The LetNet architecture was able to get an accuracy in the 80's (percent).  My assumption was that it was not robust enough for the road sign images vs. digit images (from 1 chanel black and white to 3 channel color, as well going from 10 labels to 43)

I next increased the depth of the ouput nodes in the convolution layers (for the extra color channels) and increased the number of nodes in the fully connected layers, including adding a layer (for the extra labels).  With this I was able to "overfit" so I knew the model was capable of modeling the data.

I then added dropout between the fully connected layers to counteract the overfitting.

From that point it was a lot of trial and error:
* Raising/Lower the dropout
* Adding/Removing output depth of the convolution layers
* Addding/Removing nodes from the the fully connected layers
* Adding/Removing fully connected layers
* Increasing/Decreasing the learning rate
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][test_image1] ![alt text][test_image2] ![alt text][test_image3] 
![alt text][test_image4] ![alt text][test_image5] ![alt text][test_image6] 
![alt text][test_image7] 

The last image (130 kph speed limit) is _not_ part othe trainig set, so it will be interesting to see what the model classifies it as.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

Here are the results of the prediction:

| Image                 | Prediction             | 
|:---------------------:|:----------------------:| 
| Stop Sign             | Stop sign              | 
| 30 km/h               | 30 km/h                |
| Slippery Road         | Slippery Road          |
| Right Turn            | Right Turn             |
| Road Construction     | Road Construction      |
| Right of Way          | Right of Way           |


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.   This better than the accuracy of the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.  Below is  a plot of the top 5 predictions for each sign.

![alt text][predict_image1]
![alt text][predict_image2]
![alt text][predict_image3]
![alt text][predict_image4]
![alt text][predict_image5]
![alt text][predict_image6]
![alt text][predict_image7]

The top predictions for each sign:

| Probability      | Prediction             | 
|:----------------:|:----------------------:| 
| 1.0000           | Stop sign              | 
| 0.8859           | 30 km/h                |
| 0.9909           | Slippery Road          |
| 1.0000           | Right Turn             |
| 1.0000           | Road Construction      |
| 0.9959           | Right of Way           |
| 0.9403           | 30 km/h                |

For the 30 km/h sign, it only got 88.6% probability, the next higest also being a speed sign (11.4% for 50 km/h).   So it seems good at recognizing speed signs, not quite as confident with the digits.

The interesting one is the last one, the 130 km/h sign.  There was no 130 km/h sign in the training set nor is there a classifcation label for it.   Note the top 3 predictions for this sign are all speed signs, and the one if settled on at 94% probability is 30 km/h, and the sign actually contains "30" in the digits.

