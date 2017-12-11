## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/sliding_windows.jpg
[image6]: ./examples/sliding_window.jpg
[image7]: ./examples/sliding_windows.jpg
[image8]: ./examples/sliding_window.jpg
[image9]: ./examples/bboxes_and_heat.png
[image10]: ./examples/bboxes_and_heat.png
[image11]: ./examples/bboxes_and_heat.png
[image12]: ./examples/bboxes_and_heat.png
[image13]: ./examples/bboxes_and_heat.png
[image14]: ./examples/bboxes_and_heat.png
[image15]: ./examples/labels_map.png
[image16]: ./examples/labels_map.png
[image17]: ./examples/labels_map.png
[image18]: ./examples/labels_map.png
[image19]: ./examples/labels_map.png
[image20]: ./examples/labels_map.png
[image21]: ./examples/output_bboxes.png
[image22]: ./examples/output_bboxes.png
[image23]: ./examples/output_bboxes.png
[image24]: ./examples/output_bboxes.png
[image25]: ./examples/output_bboxes.png
[image26]: ./examples/output_bboxes.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for the code for this step is contained in cell 2 & 3 & 4 of the IPython notebook "Vehicle_Detection.ipynb".

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `get_hog_features()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `get_hog_features()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Picking the right value for all parameters was challenging but it was a matter of try and error. I tried many combinations but only a few parameters seem to have a greater effect on how well the classifier learn. 

I settled on the HOG parameters based on the performance of the SVM classifier. The HOG parameters provided the best SVM classifier performance

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM the default classifier parameters using the HOG features. I was able to achieve a test accuracy of 99.3%. The code for the code for this step is contained in cell 6 of the IPython notebook "Vehicle_Detection.ipynb".

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the `find_cars` function to implement the sliding window search. I used different sliding window size based on the object (vehicle) location so that bigger windows at the bottom of the image (close vehicles) and smaller windows for the center of the image (far objects). 

Here are some examples of classified detections using the sliding window search.

![alt text][image3]
![alt text][image4]
![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately, I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]
![alt text][image7]
![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The code for the code for this step is contained in cell 10 of the IPython notebook "Vehicle_Detection.ipynb".

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]
![alt text][image26]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of my biggest challenge was getting rid of false positives and I was able to achieve that using a heatmap with the appropriate threshold. It was challenging to pick the right value to the heatmap threshold sine detecting the white vehicle was a lot harder compared to other vehicles, so I was hard to come up with a value that gets rid of the false positives without dropping too many detections for the white vehicle. My pipeline detects vehicles with a very high accuracy. One of the thing that can be improved is to average the detection boxes over few frames, so they don't keep on changing in size from frame to another for the same vehicle.
