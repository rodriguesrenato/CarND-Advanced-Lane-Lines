# **Advanced Lane Finding Project**

![][video_gif1]
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[img2.1.1]: ./output_images/2_1_camera_calibration_undistorted.png "Chessboard Image Undistorted"
[img2.1.2]: ./output_images/2_1_test_image_undistorted.png "Test Image Undistorted"
[img2.2]: ./output_images/2_2_thresholds_calibration_sc.png "Threshold Calibration"
[img2.3]: ./output_images/2_3_distortion_comparisson.png "Distortion Comparisson"
[img2.4]: ./output_images/2_4_perspective_transform_calibration.png "Perspective Transform Calibration"
[img2.5.1]: ./output_images/2_5_warped_histogram.png "Histogram"
[img2.5.2]: ./output_images/2_5_sliding_window.png "Sliding Window Method"
[img2.5.3]: ./output_images/2_5_search_around.png "Search Around Method"
[img2.6]: ./output_images/2_6_lane_lines.png "Draw Lane Lines"

[video_gif1]: ./output_videos/project_video_output_1.gif "Video Result Output"
[video_gif2]: ./output_videos/project_video_output_2.gif "Video Result Output"
[video_composition_gif]: ./output_videos/project_video_output_composition.gif "Video Composition Output"

[video_lanes]: ./project_video.mp4 "Video"
[video_composition]: ./output_videos/project_video_composition_output.avi "Video Composition"

&nbsp;

---

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

&nbsp;

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

&nbsp;

---

## Writeup / README

&nbsp;

>#1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

&nbsp;

---

## Project Structure Reference
&nbsp;

All functions and code mentioned bellow are contained in the IPython notebook located in `./adv_lane_finding.ipynb`. It is divided in 2 main sections, as detailed below:

1. Full Project Implementation
    - 1.1. Imports
    - 1.2. Global variables
    - 1.3. Functions
        - [Line 2  ] `cal_camera_calibration()`
        - [Line 51 ] `cal_undistort()`
        - [Line 57 ] `cal_threshold()`
        - [Line 92 ] `cal_perspective_matrix()`
        - [Line 115] `cal_perspective()`
        - [Line 135] `cal_sliding_window()`
        - [Line 227] `cal_search_around_poly()`
        - [Line 303] `cal_fit_polynomial()`
        - [Line 365] `draw_lane_area()`
        - [Line 397] `cal_lane_curvature()`
        - [Line 421] `reset_ma()`
        - [Line 428] `get_fit_avg_arr_filtered()`
        - [Line 435] `compose_image_arr()`
    - 1.4. Advance Lane Finding implementation

&nbsp;

2. Individual Calibrations and Demonstrations
    - 2.1. Camera calibration with visualization
    - 2.2. Color and Gradient threshold dynamic calibration
    - 2.3. Comparisson between original and undistorted image
    - 2.4. Warp perspective calibration with given straight lane lines images
    - 2.5. Polynomial fit methods

&nbsp;

---

## Camera Calibration

&nbsp;

>#1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I have built a function called `cal_camera_calibration()` that returns the the camera calibration matrix and distortion coefficients.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration matrix `mtx` and distortion coefficients `dist` using the `cv2.calibrateCamera()` function and return them.

I applied this distortion correction to the test image using the `cal_undistort()` function, passing the calculated camera calibration params, and obtained this result: 

![camera calibration][img2.1.1]

The code for this step is contained in the `2.1.` section on `./adv_lane_finding.ipynb`.

&nbsp;

---

## Pipeline (single images)

&nbsp;

>#1. Provide an example of a distortion-corrected image.

The original image is distortion-corrected by function `cal_undistort()`, which receives an image `img` and camera calibration params `mtx` and `dist` as arguments, performs `cv2.undistort()` function with them, and finally returns the undistorted image.

And example to demonstrate this step is at section `2.1`, by make the following adjust to change image input:

```py
# Read an chessboard image
# img = mpimg.imread("camera_cal/calibration1.jpg")
img = mpimg.imread("test_images/straight_lines1.jpg")

```

that return the image below:

![camera calibration][img2.1.2]

Another comaparisson was made between original and undistorted image on section `2.3`, where we can see the differences from them in a thresholded binay image, where the red pixels represents pixels only from original image, in green pixels only from undistorted image and in yellow the pixels that is common for both. 

![undistortion comparisson][img2.3]

&nbsp;

>#2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The function `cal_threshold()` is responsible for the generation of a threshold binary of the image passed as argument and returns the thresholded images.

this function first convert the image to HLS colorspace accordingly to input image colorspace specified in `color_space` argument. The S channel of hsl image is used as input to a Sobel Operator in x direction by function `cv2.Sobel()`. 
A threshold is applied to the sobel output and S channel image, with the respective `sobel_threshold` and `hls_threshold` values, and the two binary results are combined in a single binary image `combined_binary`, which is return by `cal_threshold()` with others binary images generated.

This step is demonstrated at section `2.2`. The image below is the dynamic calibration process only for thresholds. This method was used to help testing different configuration and find the best values for `sobel_threshold` and `hls_threshold`.

![dynamic calibration][img2.2]

&nbsp;

>#3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This step was divided in two functions: `cal_perspective_matrix()` and `cal_perspective()`.

The `cal_perspective_matrix()` is responsible to calculate and return the transformation matrix `M`, source and destination vertices arrays (`src_vertices` and `dst_vertices`). It takes the current image size in x and y as arguments.

This function might be called one time and these returned values is stored in global variables to avoid being calculated every frame iteration. Only when one of the global variables presented bellow are changed, this function needs to run again to update calculated outputs. The global variables created to help build those vertices and are listed bellow:

```
dst_horizontal_offset = 360
src_roi_upper = 450
src_horizontal_offset_upper = 47
src_horizontal_offset_lower = 457
src_horizontal_drift_upper_l = 0
src_horizontal_drift_upper_r = 0
src_horizontal_drift_lower_l = 0
src_horizontal_drift_lower_r = 0
```

The center o image in x is taken as base to build the vertices and offset values to dist points from center and drift to make minor adjustments. The `src_roi_upper` was chosen to be close as the one used in class to get the right conversion values to meters. This is how `src_vertices` and `dst_vertices` are calulated:

```py
def cal_perspective_matrix(img_size_x,img_size_y):
    # Define warp source vertices 
    src_vertices = np.array(
    [[img_size_x/2 - src_horizontal_offset_lower + src_horizontal_drift_lower_l , img_size_y],
        [img_size_x/2 - src_horizontal_offset_upper + src_horizontal_drift_upper_l , src_roi_upper], 
        [img_size_x/2 + src_horizontal_offset_upper + src_horizontal_drift_upper_r , src_roi_upper], 
        [img_size_x/2 + src_horizontal_offset_lower + src_horizontal_drift_lower_r , img_size_y]],
        np.float32)

    # Define warp destination vertices 
    dst_vertices = np.array(
    [[img_size_x/2 - dst_horizontal_offset, img_size_y],
        [img_size_x/2 - dst_horizontal_offset, 0],
        [img_size_x/2 + dst_horizontal_offset, 0], 
        [img_size_x/2 + dst_horizontal_offset, img_size_y]],
        np.float32)

    # Calculate the warp transformation matrix M
    M = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
    
    return src_vertices, dst_vertices, M
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 183, 720      | 280, 720      | 
| 593, 450      | 280, 0        |
| 687, 450      | 1000, 0       |
| 1097, 720     | 1000, 720     |

After get the `M` value, the function `cal_perspective()` is called. It takes and image, `M`, and a boolean flag `inverted` as arguments. It uses `cv2.warpPerspective()` to compute a new warped image by `M` and with the same size of input image. `cal_perspective()` permits to have binary and colored image as input, it checks the input image shape to get it's 2D size and convert binary image to 0-255 range values before use `cv2.warpPerspective()`.

If `inverted` flag is set `True`, then the flag `flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR` is added to `cv2.warpPerspective()` to internally invert `M` matrix, so the `cal_perspective()` function can be used later to unwarp previously warped images. 

This step is demonstrated at section `2.4` and the output is shown below:

![alt text][img2.4]

* Note that during the src vertices calibration, it wasn't possible to get perfect vertical lane lines, so I have prioritize to get many lines aligned as possible, so only the right line from first image wasn't perfect straight and other lines was pretty close to a straight line.

&nbsp;

>#4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This step was divided in three main functions: `cal_sliding_window()`, `cal_search_around_poly()` and `cal_polynomial_fit()`. To explain this step, the section `1.4. Advance Lane Finding Implementation` will be used as reference and all mentioned functions are located on the section `1.3. Functions`. 

These steps outputs can be visualized by the code in section `1.4.`, after run this cell, in the opencv window, just press `a` to change to image mode and then change between test images with `w` and `e`. To reset moving average to force `cal_sliding_window()`, just press `r`.

The code snippet below is on the section `1.4.` (Lines 139 - 147) and shows the following steps after `cal_perspective()` to calculate the polynomial fit in the warped image:

```py
    # Apply perspective transform to the undistorted thresholded image
    img_warped = cal_perspective(img_und_thresholded, M)

    # Get moving average arrays without empty values
    left_fit_avg_arr_filtered, right_fit_avg_arr_filtered = get_fit_avg_arr_filtered()

    # If both moving average arrays aren't empty
    if len(left_fit_avg_arr_filtered) > 0 and len(right_fit_avg_arr_filtered) > 0 :
        # Calculate the Search Around method to find lane lines 
        left_fit, right_fit, left_fitx, right_fitx, fity, img_warped_poly_fit = cal_search_around_poly(img_warped, margin_sa=margin_sa)

    else:
        # Calculate the Sliding window method to find lane lines 
        left_fit, right_fit, left_fitx, right_fitx, fity, img_warped_poly_fit = cal_sliding_window(img_warped,nwindows=nwindows,margin=margin,minpix=minpix)
```

To improve this step I have implemented a moving average on the calculated polynomial fit coefficients for each side. Each lane side has it's own moving average array (`left_fit_avg_arr` and `right_fit_avg_arr`) with a fixed number of elements defined by global variable `moving_avg_max_count`. The moving average arrays are initialized with empty numpy arrays and updated on each polynomial fit calculations. This step will be described later.

The function `get_fit_avg_arr_filtered()` returns the moving average arrays without empty polynomial fit coefficients. If both moving average filtered arrays are not empty (both have at least one valid polynomial fit coefficients array), it will call `cal_search_around_poly()`, otherwise it will call `cal_sliding_window()`.

In the function **`cal_sliding_window()`**, it is implemented the Sliding Window method to find lane lines pixels. 

This method starts without previous lane lines position references, so the histogram of the bottom half of the binary warped is calculated and the x argument of maximum values from mid left and mid right are taken as left and right starting points. 

From this start points, a rectangular region of interest `window`, defined by `margin` and `window_height`, in the binary warp is used toto seach for non zero pixels and append they to an left and right array. This window start on x position calculated by the histogram step and then iterate the number of times defined by `nwindows`. After each window iteration, if more than `minpix` non zero pixel were found, than the x position is updated by the average of x position of the non zero pixels located on the current window. Finally, it results in x and y arrays of position from left and right sides, which will be passed as arguments to the function `cal_fit_polynomial()` that will be explained later.

In the function **`cal_search_around_poly()`**, it is implemented the Search Around method to find lane lines pixels.

At this point, this function gets the curent moving average arrays with empty values filtered to calculate the average of them. With left and right averaged polynomial coefficients, a search for nonzero values in the binary warped image is made in a area +- `margin_sa` values in x direction of the averaged polynomial fit, for each side. Finally, it results in x and y arrays of position from left and right sides, which will be passed as arguments to the function `cal_fit_polynomial()`

In the function **`cal_fit_polynomial()`**, it is implemented the polynomial fit calculation from x and y arrays of position from left and right lane lines found, given as arguments.

It calculates new left and right fit polynomial values from given arrays, appends them to their respective moving average arrays, removes the oldest values and then calculates for each side a new average from moving average arrays and the x positions array with this new averaged polynomial coefficients. Theses calculated values are returned to the caller, which will be used to draw the polynomial fits in the warped image.

After `cal_sliding_window()` and `cal_search_around_poly()` draw the polynomial fit and other references in the binary warped imagem, both functions returns the results from `cal_polynomial_fit()` and the drawn binary warped image.

To show the results of `cal_sliding_window()` and `cal_search_around_poly()` individually, check section `2.5` to get the results shown below:

- Histogram of the lower half binary warped image
![alt text][img2.5.1]

- `cal_sliding_window()` output with windows drawn in green
![alt text][img2.5.2]

- `cal_search_around_poly()` output with search area drawn in green
![alt text][img2.5.3]

&nbsp;

>#5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function `cal_lane_curvature()` is responsible for calculating the lane lines curvatures. It takes y positions array, the left and right averaged polynomial coefficients previously calculated and the pixels to meters conversion factors, and this function return the left and right curvatures in meters.

There were approached two ways of get the polynomial coefficients converted to meters: 
1. by directly apply x and y convertion factors to coefficients
   
   ```py
    left_fit = np.array([left_fit[0]*factor_x/(factor_y**2), left_fit[1]*factor_x/factor_y, left_fit[2]*factor_x])

    right_fit = np.array([right_fit[0]*factor_x/(factor_y**2), right_fit[1]*factor_x/factor_y, right_fit[2]*factor_x])
   ```

2. By calculate a new polynomial fit with x and y arrays multiplied by respectives convertion factors

    ```py
    # Build an array of x values with given polynomial fit coefficients
    left_fitx = left_fit_cr[0]*fity**2 + left_fit_cr[1]*fity + left_fit_cr[2]
    right_fitx = right_fit_cr[0]*fity**2 + right_fit_cr[1]*fity + right_fit_cr[2]

    # Multply x and y arrays for the respective factors, and then calculate the new polynomial fit coefficients
    left_fit_cr_m = np.polyfit(fity*factor_y, left_fitx*factor_x, 2)
    right_fit_cr_m = np.polyfit(fity*factor_y, right_fitx*factor_x, 2)
    ```

The radius of curvature is calculated on the closest y values from the vehicle and it uses the new polynomial coefficients calculated.

```py
# Calculation of R_curve (radius of curvature)
left_curverad = ((1 + (2*left_fit_cr_m[0]*y_eval*factor_y + left_fit_cr_m[1])**2)**1.5) / np.absolute(2*left_fit_cr_m[0])
right_curverad = ((1 + (2*right_fit_cr_m[0]*y_eval*factor_y + right_fit_cr_m[1])**2)**1.5) / np.absolute(2*right_fit_cr_m[0])
```

The position of the vehicle is calculated in the section `1.4.` (Line 165), by the following approach:

```py
lane_center_diff_x = (float(right_fitx[-1] + left_fitx[-1])/2 - float(img_warped_poly_fit.shape[1])/2)*xm_per_px

lane_center_diff_x_str = "{:0.2f}m {}".format(np.absolute(lane_center_diff_x),"Left" if lane_center_diff_x > 0.1 else ("Right" if lane_center_diff_x < 0.1 else "Center"))
```

The lane center is calculate by subtracting the last right x position and the last left x position from the averaged polynomial fit arrays of x positions. The difference between lane center and the image center is multiplied by `xm_per_px` to get this value in meters. If the result is positive, the car is near the left side, otherwise is near the right side.


&nbsp;
>#6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This step was implemented in function `cal_draw_lane_area()`. In section `1.4.` (Lines 177-179), the curvature and distance from center informations are written to final image (`img_lanes`). Here is an example of my result on a test image:

![alt text][img2.6]

The function `cal_draw_lane_area()` receives the undistorted image, warped image output from polynomial fit functions, perspective transform matrix `M` and the polynomial fit arrays calculated in previous steps.

As the resulted warped image from polynomial fit comes with green pixels, then the green and white pixels are changed to black and saved on a new image `img_warped_filtered`

A new black image of same size of warped image is created and a polygon will be drawn on it, filled in green color, with vertices made from polynomial fit arrays, so it fill the area between the left and right polynomial fit. 

This image will be merged with `img_warped_filtered` and unwarped with the function `cal_perspective()` with argument `inverted=True`, to internally use the invert `M` matrix in `cv2.warpPerspective()` by adding the argument `flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR`. 

The unwarped image is directly merged with the undistorted image to result the `img_lanes`, but it results a weak colored highlights in the lane. To achieve a better result, after merging them in `img_lanes_aux`, a copy of undistorted image is made as `img_lanes` and all colored pixels positions in the unwarped image are calculated and used to copy the respective pixels colors from `img_lanes_aux` to `img_lanes`. 

&nbsp;

---

## Pipeline (video)

&nbsp;


> #1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to my video results: 

1. Result output: [project_video_output.avi](video_output)
![alt text][video_gif2]
2. Video composition output of all steps: [project_video_composition_output.avi](video_compostion)
![alt text][video_composition_gif]
&nbsp;

---

## Discussion

&nbsp;

>#1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I started developing this project in the jupyter notebook, step by step on multiple cells. As the project started getting longer I had to repeat code from previous steps, so because of that I decided to write a single cell code that runs both images and videos, automatically handle variables calibration by trackbars and join all created images on a single image to show all together. This code was deloped incrementing step by step, to make easier the developing and learning process.
    
I created the `compose_image_arr()` function to receive an array of image and an array of image titles to join images by a fixed number of images horizontaly and a resize factor to resize all images before arrange them.

By using opencv imshow and trackbars, it gets much tangible to see how each variable and configuration affects on each step and helped a lot to understand my approaches.

I've tried many combinations on threshold step, but the result wasn't precise as I expect, but resonable. The challenge videos didn't work well, even changing the parameters I used in the project video, so I suppose that it has to be used other technics/functions to make them work (combine more color channel thresholds, other image filters, reduce the warping area to avoid sharp curves)

I find a way to avoid outliers and when lane lines aren't found. I implemented a simple moving average with fixed length of the results of polynomial fit. After each polynomial fit calculation, these new values are appended to the moving average and a mean of values are calculated and returned as the current polinomial fit values. If moving average gets only empty values, then it does Sliding Window method until gets a valid values to the moving average arrays.

The curvature calculation was tricky, but I got resonable curvature values. As the threshold step didn't got perfectly the pixels farther than the car, and it seems that even on calibration on straight lines the results are not 100% straight, then it has oscillated between values even though I have implemented the moving average for polynomial fit. 

The distance from center of lane calculation worked pretty well with expected results. 

Errors might have some contribuition from camera calibration process because in the calibration images given, some chessboard papers weren't sticked perfectly flat on the wall and had some wrinkles on it.

The implementation made to draw the lane lines back on the undistorted image might not be the best approach, but it was the way I found to get stronger colors merged to the it.

When I watched the output video I noticed that it is much faster than the one shown when the code is running. The time to process each frame and show on cv2.imshow() window is longer than the time between frames in the video file. 

Some code snippets from Udacity course class and from opencv, numpy, matplotlib official docs were used as references to develop this project.