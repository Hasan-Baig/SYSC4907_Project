# SYSC4907_Project

## Process:

![process](https://github.com/Hasan-Baig/SYSC4907_Project/blob/hasan/pics/process.png?raw=true)

## Segment the Hand region

Before hand gesture recognition can occur, need to find the hand region by eliminating all the other unwanted portions in the video sequence. We do this by:
1) Background Subtraction
2) Motion Detection and Thresholding
3) Contour Extraction

### Background Subtraction
- generates a foreground mask (binary image containing the pixels belonging to moving objects) by using static cameras
- calculates the foreground mask performing subtraction between current frame and background model (static part of the scene)

![back_sub](https://github.com/Hasan-Baig/SYSC4907_Project/blob/hasan/pics/back_sub.png?raw=true)

After figuring out the background model using running averages, the current frame which holds the foreground object is used in addition to the background. We calculate the absolute difference between the background model (updated over time) and the current frame (which has our hand) to obtain a difference image that holds the newly added foreground object (which is our hand).

### Motion Detection and Thresholding
- To detect the hand region from this difference image, we need to threshold the difference image
- This allows the hand region to become visible and all the other unwanted regions are painted as black.
- Thresholding is the assigment of pixel intensities to 0's and 1's based a particular threshold level so that our object of interest alone is captured from an image.

### Contour Extraction
- After thresholding the difference image, we find contours (outline) in the resulting image. The contour with the largest area is assumed to be our hand.

## Legend:

![legend1](https://github.com/Hasan-Baig/SYSC4907_Project/blob/hasan/pics/legend1.png?raw=true)
![legend2](https://github.com/Hasan-Baig/SYSC4907_Project/blob/hasan/pics/legend2.png?raw=true)
