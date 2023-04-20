import numpy as np
import matplotlib.pyplot as plt
import cv2
def resize(img, line):
    
    scale_percent = 100 # percent of original size
    if (img.shape[0]>=500 or img.shape[1]>=600):
       scale_percent = 40
      
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim)
    if line ==1:
        plt.imsave("static/images/resized.jpg",resized,cmap='gray')
        img = cv2.imread("static/images/resized.jpg",0)

    else:
        cv2.imwrite("static/images/resized.jpg",resized)
        img = cv2.imread("static/images/resized.jpg",cv2.IMREAD_COLOR)

    # resized = imutils.resize(img,width=200)
        
    
    return img


def convolve(img, kernel):
    size = img.shape[0]
    k = kernel.shape[0]
    
    # Initiate an array of zeros for the resulting convolved image
    convolved_img = np.zeros(shape=(size, img.shape[1]))
    
    # Loop over the rows
    for i in range(img.shape[0]-2):
        # Loop over the columns
        for j in range(img.shape[1]-2):
            mat = img[i:i+k, j:j+k]
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))
            
    return convolved_img

def round_angle(angle):
    angle = np.rad2deg(angle) % 180

    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle


