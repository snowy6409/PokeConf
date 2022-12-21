import cv2
import numpy as np
import imutils

# Load image, grayscale, blur, Otsu's threshold
image = cv2.imread('rn_image_picker_lib_temp_e48345e3-656e-40d5-9994-7763d43c9794.jpg')
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Find contours and filter for cards using contour area
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
threshold_min_area = 60000
number_of_contours = 0
d_min = 100

image_center = (image.shape[0]/2, image.shape[1]/2)
for c in cnts:
    area = cv2.contourArea(c)
    if area > threshold_min_area:
        cv2.drawContours(image, [c], 0, (36,255,12), 3)
        number_of_contours += 1
        # finding bounding rect
        rect = cv2.boundingRect(c)
        # skipping the outliers
        if rect[3] > image.shape[1]/2 and rect[2] > image.shape[0]/2:
            continue
        pt1 = (rect[0], rect[1])
        # finding the center of bounding rect-digit
        c = (rect[0]+rect[2]*1/2, rect[1]+rect[3]*1/2)
        d = np.sqrt((c[0] - image_center[0])**2 + (c[1]-image_center[1])**2)
        # finding the minimum distance from the center
        if d < d_min:
            d_min = d
            rect_min = [pt1, (rect[2],rect[3])]
        pad = 5
        result = image[rect_min[0][1]-pad:rect_min[0][1]+rect_min[1][1]+pad, rect_min[0][0]-pad:rect_min[0][0]+rect_min[1][0]+pad]
print("Contours detected:", number_of_contours)

cv2.imshow("result",result)
cv2.imshow('thresh', thresh)
cv2.imshow('image', image)

cv2.waitKey()