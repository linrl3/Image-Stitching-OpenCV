# Image-Stitching-OpenCV
Simple image stitching algorithm using SIFT, homography, KNN and Ransac in Python.
For full details and explanations, you're welcome to read `image_stitching.pdf`. 	

The project is to implement a featured based automatic image stitching algorithm. When we input two images with overlapped fields, we expect to obtain a wide seamless panorama.

We use scale invariant features transform(SIFT) to extract local features of the input images, K nearest neighbors algorithms to match these features and Random sample consensus(Ransac) to calculate the homograph matrix, which will be used for image warping. Finally we apply a weighted matrix as a mask for image blending.

## Dependency
- Python 2 or 3 
- OpenCV 3

## Usage
`python Image_Stitching [/PATH/img1] [/PATH/img2]`

## Sample 

## Input images
<img src="https://github.com/linrl3/Image-Stitching-OpenCV/blob/master/images/q11.jpg" width=300 height=400 >   <img src="https://github.com/linrl3/Image-Stitching-OpenCV/blob/master/images/q22.jpg" width=300 height=400 >

## Matching
![matching](https://github.com/linrl3/Image-Stitching-OpenCV/blob/master/images/matching.jpg)

## Output image
![pano](https://github.com/linrl3/Image-Stitching-OpenCV/blob/master/images/panorama.jpg)

## other examples

Room:
![room](https://github.com/linrl3/Image-Stitching-OpenCV/blob/master/images/room.jpg)

Another building: This is a stitching of three related images. It doesn't work so well. Trying to improve.
![gym](https://github.com/linrl3/Image-Stitching-OpenCV/blob/master/images/gym.jpg)
