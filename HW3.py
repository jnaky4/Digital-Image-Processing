import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

'''

takes in an image, returns an array for the magnitude and Direction
convoles either the sobel or prewitt kernels (1) for sobel
    one in the X and one in the Y direction
compute the hypotenuse to get magnitude for each pixel
compute arctan to get Direction of each pixel
'''

def FilteredGradient(img, choosekernel):
    # create the Sobel & Prewitt kernels X & Y
    sobelKernelX = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], np.float32)
    sobelKenrelY = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]], np.float32)
    prewittKernelX = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]], np.float32)
    prewittKernelY = np.array([[-1, -1, -1],
                               [0, 0, 0],
                               [1, 1, 1]], np.float32)
    if choosekernel == 1:
        '''filter2D(source image, depth, kernel)'''
        gradX = cv2.filter2D(img, -1, sobelKernelX)
        gradY = cv2.filter2D(img, -1, sobelKenrelY)
    else:
        gradX = cv2.filter2D(img, -1, prewittKernelX)
        gradY = cv2.filter2D(img, -1, prewittKernelY)

    magnitudeF = np. hypot(gradX, gradY)
    magnitudeF = magnitudeF / magnitudeF.max() * 255
    magnitudeF = magnitudeF.astype(np.uint8)
    directionOrientation = np.arctan2(gradY, gradX)
    return (magnitudeF, directionOrientation)




"""
Takes a image and an orientation array
    iterate through each pixel and check left and right neighbors
    neighbors are divided into 4 quadrants even though 8 adjacent neighbors
    this is because we grab 2 neighbors and only need 180 degreess to get both
    
    example
            135 90 45
              \ | /
           180 - - 0
                   
    000                                         0X0
    010  D = 90 grab top and bottom neighbor    010
    000                                         0x0
    
    
    if pixel is less than their intensity, set to 0, otherwise set to pixel value
"""
def non_max_suppression(img, D):
    row, col = img.shape

    # visualizes pixels magnitude and direction understanding function
    # printDisplay(img)

    # change the degrees into radians
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            try:
                right = 255
                left = 255
                # for each pixel check the angle array for the angle
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    right = img[i, j + 1]
                    left = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    right = img[i + 1, j - 1]
                    left = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    right = img[i + 1, j]
                    left = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    right = img[i - 1, j - 1]
                    left = img[i + 1, j + 1]

                # check if the pixel is stronger than neighbors
                # if not set index in return array to 0
                if (img[i, j] < right) or (img[i, j] < left):
                    img[i, j] = 0
            except IndexError as e:
                pass
    return img



# useful visualization of image, shows pixel value at pixel location
def printDisplay(img):
    width, height = img.shape
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            if (img[i][j] != 0):
                print('{:<3}'.format(img[i][j]), end=" ")
            else:
                print("   ", end=" ")
        print()






"""
    Take and image, ThresholdRatio for low and high, returns a hyserisis thresholded image
    
    
"""
def threshold(img, lowThresholdRatio, highThresholdRatio, weak, strong):

    row, col = img.shape
    threshedImg = np.zeros((row, col), dtype=np.int32)
    # calculate the low and high thresholds
    highThresh = img.max() * highThresholdRatio
    lowThresh = highThresh * lowThresholdRatio

    """
    Loops through all img pixels
    sets pixels to 3 values:
        255: for above high threashold
        25: for pixels in between
        0: for pixels below low threshold
    """
    for i in range(1, int(img.shape[0] - 1)):
        for j in range(1, int(img.shape[1] - 1)):
            if(img[i,j] > highThresh) :
                threshedImg[i, j] = strong
            elif(img[i,j] < lowThresh) :
                threshedImg[i, j] = 0
            #Intermediate pixels
            else :
                threshedImg[i, j] = weak

    return threshedImg



"""
    Last Step, Takes the image, weak and strong values to filter, returns the image
    for each pixel, looks at next neighbors, if any neighbor is strong then its a connected edge
    if there are no connected edges, change the value to 0
"""
def hysteresis(img, weak, strong):
    # printDisplay(img)
    # print(weak)
    # print(strong)
    row, col = img.shape
    for i in range(1, row-1):
        for j in range(1, col-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def load():
    # Load the image as greyscale
    img = cv2.imread('moon.tif', 0)
    return img

def main():

    # Load the image
    img = load()
    f = plt.figure()
    # display original image
    plt.imshow(img, cmap="gray")
    plt.title('Original')

    # apply GaussianBlur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    f = plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title('GaussianBlur')

    # # Executing the Filtered Gradient Method
    img, filtOrient = FilteredGradient(img,1)
    f = plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title('filtered gradient image')

    # # Executing the Non maximum Suppression Method
    img = non_max_suppression(img, filtOrient)
    f = plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title('nmax suppressed')

    # Executing the Hysteresis Thresholding Method
    weak = np.int32(45)
    strong = np.int32(255)
    lowThreshRatio = .02
    highThreshRatio = 0.17
    img = threshold(img, lowThreshRatio, highThreshRatio, weak, strong)

    f = plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title('apply threshold')


     # Executing the Hysteresis Method
    img = hysteresis(img, weak, strong)
    f = plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title('after hysteresis')
    printDisplay(img)
    # once complete will show all images at once
    plt.show()

if __name__ == '__main__':
    main()
