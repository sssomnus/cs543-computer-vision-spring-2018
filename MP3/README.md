********

# Spring 2018 CS543:  Assignment 3 
**@author**: Ziyu Zhou  

********



# Part 1: Stitching pairs of images

## 1 Implementation & Solution

### 1.1 Overview

#### 1.1.1 Usage
To run the program, simply `cd` to the `part_1` folder and run `python3 main.py`. The overall pipeline of the program is:

```
Set up parameters
Read images and convert to grayscale
Perform feature detection
Perform feature matching
Use RANSAC to match inliers
Load original color images and normalize
Stitch images
```

#### 1.1.2 Parameters
Final parameters for thresholding:

```python
t_match = 7000  # Threshold for matching features.
t_ransac = 0.5  # Threshold for RANSAC.
```
See the corresponding sections for detailed explanations.


### 1.2 Load and Preprocess
This is implemented in `utils/io_data_tools.py`. 

Note that funciton `load_and_normalize()` is to load the image, covert it to double and normalize the pixel values so that the smallest pixel values become 0 and the largest ones become 1. This is necessary when stitching the images since we need to average the pixel values of the two images and without this, the pixel values will overflow, causing unnatural colors. However, for the other parts of the program, the images should not be normalized, otherwise OpenCV will be very likely to complain because of incorrect image types.

### 1.3 Feature Detection
I use SIFT detector to detect keypoints and generate descriptors, which is implemented in the function named `sift_descriptors()` in file `feature_matching.py`. 

The image array passed to this function must be grayscale and in CV_8U type, because the `sift.detectAndCompute()` and `cv2.xfeatures2d.SIFT_create()` will not accept double precision.

### 1.4 Feature Matching

#### 1.4.1 Basic Idea
This is implemented in function `get_matched_pixels()` in file `feature_matching.py`, which uses Euclidean distance to find matching descriptors.

The basic steps are:

```
Compute pair distance between descriptors of two images
Get matched descriptors with distances smaller than some threshold
Find the corresponding keypoint coordinates
```

#### 1.4.2 Parameter
The threshold for the Euclidean distance is specified in `main.py` as `t_match`.

The best value of it is `t_match = 7000`. I tested a set of values including 2000, 4000, 6000, 8000, and it was shown that the number of matched pairs are more reasonable when `t_match` was in range `[4000, 6000]`. I performed more tests there and finally chose `7000`.


### 1.5 RANSAC

#### 1.5.1 Basic Idea

The main fucntion to perform RANSAC to fit homography is `ransac_fitting()`, which is file `ransac.py`.

The basic procedure of RANSAC is:

```
While not reach max iterations
	Randomly select 4 unique matched pairs
	Fit a homography H
	Jump to next loop if H is degenerate
	Find and add inliers according to some threshold
	Save current solution and compute residual if it's the best
```

Note that in the first step the pairs selected should not contain duplicates so as to avoid degenerate `H`.

#### 1.5.2 Compute Homography
A helper function `fit_homography()` is used to compute homography `H` during each iteratoin of the RANSAC procedure.

```
Construct A matrix
Perform SVD and find singular vector of the smallest singular value
Get H
Normalize H
```

#### 1.5.3 Compute Residuals
Helper function `get_errors()` is used to get residual between original points and points transformed by H. It Return an array of errors for all points. 

It can also be used to compute average residual for inliers by simplying sum the returned array up.

#### 1.5.4 Parameters
`max_ite = 1000` defines the number of times that RANSAC runs. Setting it to such a large value can make sure the result is the best one.

The threshold for distance between matching pairs in the RANSAC loop is `t_ransac = 0.5`. Using a smaller threshold can make sure that we won't add some "fake" pairs into our inliers.

### 1.6 Stitching

#### 1.6.1 Basic Idea
The main function to perform stitching is `stitch_img()` in file `stitching.py`.

The basic steps are:

```
Warp the left image
Move the right image
For every pixel in the image frame
	If left pixel is not all black while the right one is, \
	set final pixel value as the left one and vise versa
	If both are not all black, take average of two pixels
```
The returning size of the stitched image is defined as

```python
return warped_l[:moved_r.shape[0], :moved_r.shape[1], :]
```
In this way the unnecessary black area can be cut off.

#### 1.6.2 Warp Images
Assuming the right image remains unwarped which only has translation, I warp the left image onto the right one.

Function `warp_left()` in `stitching.py` is used to warp the left image and make sure no pixel is out of image frame, which is kind of a tricky part.

The key to do this is to find the maximum (or minimum) translation of the four corner points of the image after being transformed by the homography `H`, and adjust the output image frame size according to that translation. This is conducted using `cv2.warpPerspective()`, who has a parameter `dsize` for us to change the size of the image frame. 

Another trick is to multiply `H` with the translation matrix to make sure the ouput image is at the correct location of the image frame.

Function `move_right()` is used to move the right image according to the translation of the left one which has been warpped. We don't actually need to "warp" this image.



## 2 Outputs and Results

### 2.1 Inliers
For inliners founded after RANSAC,

```
Number of inliers: 165, 
Average residual: 0.1605191409353521
```
See the following image for the locations of inlier matches in both images. 

<center>
![img](part_1/outputs/inlier_matches.svg)
</center>

### 2.2 Final Result for Stitching

The result is relatively good, except for some bulrry regions on top the tree at the middle part of the image.

![img](part_1/outputs/uttower_stitched.jpg)


### 2.3 Other Results

See below for the keypoints founded by SIFT detector.

|              `sift_left`             |              `sift_right`             |
|:------------------------------------:|:-------------------------------------:|
| ![img](part_1/outputs/sift_left.jpg) | ![img](part_1/outputs/sift_right.jpg) |


<div style="page-break-after: always;"></div>

## 3 Extra Credits for Part 1

Based on the first possible extra credit: _Extend your homography estimation to work on multiple images_.

### 3.1 Basic Idea

Codes for extra credits are implemented in `main_extra_credit.py`. To run the program, `cd` to `part_1` and run `python3 main_extra_credit.py`.

The idea to stitch 3 images is first stitching images 1 and 2, and using that stitched image to stitch with image 3.

### 3.2 Outputs

#### 3.2.1 `hill`

![img](part_1/outputs/extra_credit/hill/stitched_123.jpg)

#### 3.2.2 `ledge`
Some part of the image has been cropped incorrectly. Time allowed, I might further improve the stitching method.

![img](part_1/outputs/extra_credit/ledge/stitched_123.jpg)

#### 3.2.3 `pier`
![img](part_1/outputs/extra_credit/pier/stitched_123.jpg)


<div style="page-break-after: always;"></div>


# Part 2: Fundamental Matrix Estimation and Triangulation


**Note for how to run the program under different purposes:**

```
To run the program, type in command line:
    1. If you want to fit fundamental matrix:
        1)  with ground truth labels (normalized & unnormalized)
            "python3 main.py fit_fundamental gt"
        2)  no ground truth labels (using RANSAC)
            "python3 main.py fit_fundamental no_gt"
    2. If you want to perform triangulation:
        "python3 main.py triangulation"  
```

## 1 Outputs using Ground Truth Matches

Note that all plots (excpet **section 1.3**) are using homogeneous system setup to solve the fundamental matrix `F`. See **section 1.3** for comparisons over homogeneous and non-homogeneous setups.

### 1.1 `house`

**Ground truth matches:**

![img](part_2/outputs/house_matches.svg)


<div style="page-break-after: always;"></div>

#### 1.1.1 Unnormalized Results

Note that `Avg dist1` and `Avg dist2` means average residuals for image 1 and image 2 separately.


![img](part_2/outputs/house_epipolar_unnorm.svg)



**Residuals:**

```
Unnormalized. 
Avg dist1: 0.03515121198779122, 
Avg dist2: 0.04263472012864928
```


#### 1.1.2 Normalized Results


![img](part_2/outputs/house_epipolar_norm.svg)



**Residuals:**

```
Normalized. 
Avg dist1: 0.003125666448257006, 
Avg dist2: 0.0038395935080640137
```

<div style="page-break-after: always;"></div>

### 1.2 `library`

**Ground truth matches:**


![img](part_2/outputs/library_matches.svg)



#### 1.2.1 Unnormalized Results


![img](part_2/outputs/library_epipolar_unnorm.svg)



**Residuals:**

```
Unnormalized. 
Avg dist1: 0.06041061439397669, 
Avg dist2: 0.05687674020796551
```


#### 1.2.2 Normalized Results


![img](part_2/outputs/library_epipolar_norm.svg) 


**Residuals:**

```
Normalized. 
Avg dist1: 0.006814739720254728, 
Avg dist2: 0.005789292241227336
```


### 1.3 Comparisons over Homogeneous and Non-homogeneous Setups

No matter using normalized or unnormalized method, homogeneous setup seems to outperform non-homogeneous setup with smaller residuals, see the table below (the results are generated from `house` image):

|                 |             Unnormalized             |                Normalized               |
|-----------------|:------------------------------------:|:---------------------------------------:|
| Homogeneous     | Avg dist1: 0.0351512, Avg dist2: 0.0426347 | Avg dist1: 0.0031256, Avg dist2: 0.0038395 |
| Non-homogeneous | Avg dist1: 0.0423628, Avg dist2: 0.0365413   | Avg dist1: 0.1013664, Avg dist2: 0.0649582 |


Non-homogeneous setup is not quite stable, sometimes producing very bad results (actually most times), as shown below:

|                Unnormalized, Non-homogeneous               |                Normalized, Non-homogeneous               |
|:----------------------------------------------------------:|:--------------------------------------------------------:|
| ![img](part_2/outputs/house_epipolar_unnorm_nonhomo.svg) | ![img](part_2/outputs/house_epipolar_norm_nonhomo.svg) |

Thus, I decided to use homogeneous for all cases.


## 2 Outputs without Ground Truth Matches


### 2.1 `house`

#### 2.1.1 Output


![img](part_2/outputs/house_epipolar_ransac.svg)


#### 2.1.2 Inliers
**Plot for inlier matches using RANSAC:**


![img](part_2/outputs/house_inlier_matches.svg)


**Inliers:**

```
Number of inliers: 106, 
Average residual: 0.006313754734804022
```


### 2.2 `library`

#### 2.2.1 Output


![img](part_2/outputs/library_epipolar_ransac.svg)


#### 2.2.2 Inliers
**Plot for inlier matches using RANSAC:**


![img](part_2/outputs/library_inlier_matches.svg)


**Inliers:**

```
Number of inliers: 360, 
Average residual: 0.002426856898789952
```


## 3 Outputs for Triangulation

### 3.1 `house`

**3D plots:**

See the plots for different viewpoints of the 3D triangulation result. Also find the `house_3d.gif` file in folder `outputs/triangulation` for better visualization.

| ![img](part_2/outputs/triangulation/house_3d_1.png) | ![img](part_2/outputs/triangulation/house_3d_2.png) |
|:---------------------------------------------------:|:---------------------------------------------------:|
| ![img](part_2/outputs/triangulation/house_3d_3.png) | ![img](part_2/outputs/triangulation/house_3d_4.png) |

**Residuals:**

```
Res_1 = 0.002522109350186388,
Res_2 = 0.15655240898888137
```


### 3.2 `library`

**3D plots:**

See the plots for different viewpoints of the 3D triangulation result. Also find the `library_3d.gif` file in folder `outputs/triangulation` for better visualization.

| ![img](part_2/outputs/triangulation/library_3d_1.png) | ![img](part_2/outputs/triangulation/library_3d_2.png) |
|:---------------------------------------------------:|:---------------------------------------------------:|
| ![img](part_2/outputs/triangulation/library_3d_3.png) | ![img](part_2/outputs/triangulation/library_3d_4.png) |

**Residuals:**

```
Res_1 = 0.07312796424275397, 
Res_2 = 0.26767951261775025
```


<div style="page-break-after: always;"></div>


# Part 3: Single-View Geometry

## 1 Find Vanishing Points and Horizon Line

### 1.1 Vanishing Points Results

#### 1.1.1 Plots of the VPs

<center>

|             |                                     VPs & Lines used to estimate them                                    |
|:-----------:|:--------------------------------------------------------------------------:|
| 1 | <img src="part_3/outputs/vp1.png" height="250"> </br>  <img src="part_3/outputs/vp1_lines.png" height="250"> |
|     2     |  <img src="part_3/outputs/vp2.png" height="250"> </br> <img src="part_3/outputs/vp2_lines.png" height="250"> |
|     3     |  <img src="part_3/outputs/vp3.png" height="250"> </br> <img src="part_3/outputs/vp3_lines.png" height="250">   |

</center>

#### 1.1.2 VP pixel coordinates

Note that each column represents for a vanishing point.

```
Coordiantes of vanishing points:
[[ -7.66111124e+01   1.32730024e+03   4.96046500e+02]
 [  2.01637495e+02   2.28109402e+02   7.32284515e+03]
 [  1.00000000e+00   1.00000000e+00   1.00000000e+00]]
```

### 1.2 Ground Horizon Line

#### 1.2.1 Plot for Ground Horizon Line

<center>
![img](part_3/outputs/horizon_line.png)
</center>


#### 1.2.2 Parameters
Parameters are in the form of `a * x + b * y + c = 0`, where `a^2 + b^2 = 1`.

```
Horizon line:
[ -1.88524737e-02   9.99822276e-01  -2.03045968e+02]
```

## 2 Focal Length and Optical Center

### 2.1 Basic Idea

Implemented in `get_camera_parameters()`. Use `SymPy` to solve symbols `f, px, py`. An example:

```python
f, px, py = solve([eq1[0], eq2[0], eq3[0]], (f, px, py))[0]
```

### 2.2 Results

```
Focal length: 698.632679394766, 
Optical center: (628.764400220098, 284.282720211123)
```

## 3 Rotation Matrix

See the matrix below. Each column corresponds to the rotation related to `X, Y, Z` axis.

```
Rotation Matrix:
[[ 0.70591761 -0.01876032 -0.70804546]
 [-0.05676693  0.99493491 -0.08295805]
 [ 0.70601547  0.09875512  0.70127711]]
```

## 4 Estimate Heights

### 4.1 CSL Building

**Lines and measurements used to perform the calculation:**

<center>

![img](part_3/outputs/CSL.png)

</center>

**Height:**

```
Height of CSL building is 27.823072050153215 meters
```

### 4.2 Spike Statue

**Lines and measurements used to perform the calculation:**

<center>

![img](part_3/outputs/spike.png)

</center>

**Height:**

```
Height of the spike statue is 9.250992828270343 meters
```

### 4.3 Lamp Posts

#### (1) Lamp Post 0

For the first lamp post.

**Lines and measurements used to perform the calculation:**

<center>

![img](part_3/outputs/lamp0.png)

</center>

**Height:**

```
Height of the lamp posts is 4.698483945609883 meters
```

#### (2) Lamp Post 1

For the second lamp post.

**Lines and measurements used to perform the calculation:**

<center>

![img](part_3/outputs/lamp1.png)

</center>

**Height:**

```
Height of lamp post 1 is 5.30974128870862 meters
```

#### (3) Lamp Post 2

For the third lamp post.

**Lines and measurements used to perform the calculation:**

<center>

![img](part_3/outputs/lamp2.png)

</center>

**Height:**

```
Height of lamp post 2 is 4.618452824075178 meters
```

### 4.4 Assume the person is 6ft tall
Then all the heights of the objects will increase. As the result shows here:

```
Assume 6ft person:
Height of CSL building is 30.35244223653078 meters
Height of the spike statue is 10.09199217629492 meters
Height of the lamp posts is 5.125618849756236 meters
```


## 5 Extra Credit for Part 3

Based on the first possibole extra credit: _Perform additional measurements on the image: which of the people visible are the tallest?_

The conclusion is that **person 1** is the tallest (see the following height measurements and plots, as well as **section 5.2** for details).

### 5.1 Estimate Heights of All Persons

#### (1) Person 1

**Lines and measurements used to perform the calculation:**

<center>

![img](part_3/outputs/person1.png)

</center>

**Height:**

```
Height of person 1 is 1.742537347394483 meters
```

#### (2) Person 2

**Lines and measurements used to perform the calculation:**

<center>

![img](part_3/outputs/person2.png)

</center>

**Height:**

```
Height of person 2 is 1.5959521784783823 meters
```

#### (3) Person 3

**Lines and measurements used to perform the calculation:**

<center>

![img](part_3/outputs/person3.png)

</center>

**Height:**

```
Height of person 3 is 1.609119572631246 meters
```

### 5.2 Find the Tallest Person

From **section 5.1**, we know that `H1 > Height of reference person (1.6764) > H3 > H2`. Thus, person 1 is the tallest.