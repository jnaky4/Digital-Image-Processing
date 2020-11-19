import cv2
import numpy as np

# read in the image
"""imread argument 2, changes the value """
img = cv2.imread('moon.tif', 0)
rows, cols = img.shape

"""Used to test imgsharpen for overflow"""
def testsharpen():
    img2 = np.zeros([rows, cols], dtype=np.uint8)
    print(img[8][10])
    print(img[8][10] * -1)
    print(img[10][9])
    print(img[10][9] * -1)
    print(img[10][10])
    print(img[10][10] * 5)
    print(img[10][11])
    print(img[10][11] * -1)
    print(img[11][10])
    print(img[11][10] * -1)
    test = (img[8][10] * -2) + (img[10][9] * -1) + (img[10][10] * 5) + (img[10][11] * -1) + (img[11][10] * -1)
    if test < 0:
        test = 0
    if test > 255:
        test = 255
    img2[10][10] = test
    print(img2[10][10])


"""@args: display: if 1 will display the image"""
def imgsharpen(display):
    calculate = 0
    img2 = np.zeros([rows, cols], dtype=np.uint8)
    output = np.zeros((rows, cols), dtype=np.uint8)
    difference = np.zeros((rows-2, cols-2), int)
    mask = [[0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]]
    # testsharpen()

    """iterate through the image ignoring the border indexes
            this choice was for simplicity and not introducing errors
    """
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            """the mask above have 4 index that take the product of the index
                    to improve efficiency those values are not calculated since the would return 0"""
            calculate = ((img[i-1][j] * mask[0][1]) +
                        (img[i][j-1] * mask[1][0]) +
                        (img[i][j] * mask[1][1]) +
                        (img[i][j+1] * mask[1][2]) +
                        (img[i+1][j] * mask[2][1]))
            """checking for overflow, if adjacent pixels are opposite values, 
            the mask value product will become negative/positive and overflow"""
            if calculate < 0:
                calculate = 0
            if calculate > 255:
                calculate = 255
            img2[i][j] = calculate
    """@arg display == 1: displays the results, used for method chaining"""
    if display:
        try:
            print("done")
            cv2.imshow('result', img2)
            cv2.imshow('original', img)
            cv2.waitKey(0)
        except Exception as e:
            print(e)
    return output


"""@args: display: if 1 will display the image"""
def meanblur(display):
    weight = 0
    img2 = np.zeros([rows, cols], dtype=np.uint8)
    mask = np.array([[0, 1, 0],
                     [1, 4, 1],
                     [0, 1, 0]])
    """we take the product of the mask index with the img index / sum of values"""
    for i in range(3):
        for j in range (3):
            weight += mask[i][j]

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            """cast type to uint8 and divide by mask size 9 to prevent an overflow
                // is integer division, floors decimal value from division
                mask indexes that == 0 are ignored to increase efficiency... this is python"""
            img2[i][j] = (img[i-1][j] * mask[0][1]//weight +
                          img[i][j-1] * mask[1][0]//weight +
                          img[i][j] * mask[1][1]//weight +
                          img[i][j+1] * mask[1][2]//weight +
                          img[i+1][j] * mask[2][1]//weight)
    """@arg display == 1: displays the results, used for method chaining"""
    if display:
        try:
            cv2.imshow('result', img2)
            cv2.imshow('original', img)
            cv2.waitKey(0)
            print("done")
        except Exception as e:
            print(e)
    return img2


"""@args: display: if 1 will display the image
this method grabs the 3x3 mask, stores it in the mask array, 
where it is sorted and takes the median value to replace the old index value"""
def medianblur(display):
    img2 = img2 = np.zeros((rows, cols), dtype=np.uint8)
    """create an empty mask to be filled with intensity values around pixel index"""
    mask = np.zeros((3, 3), np.uint8)
    """iterate through the image ignoring the border indexes
            this choice was for simplicity and not introducing errors
        we ignore padding for simplicity:
            1 because kernels typically pad the mask anyway
            2 for simplicity of complicated nested if else, potentially introducing errors"""
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            mask[0][0] = img[i-1][j-1]
            mask[0][1] = img[i-1][j]
            mask[0][2] = img[i-1][j+1]
            mask[1][0] = img[i][j-1]
            mask[1][1] = img[i][j]
            mask[1][2] = img[i][j+1]
            mask[2][0] = img[i+1][j-1]
            mask[2][1] = img[i+1][j]
            mask[2][2] = img[i][j+1]
            """sort the values so we can pull the median value an assign it to the pixel"""
            mask.sort()
            img2[i][j] = mask[1][1]
    """@arg display == 1: displays the results, used for method chaining"""
    if display:
        try:
            cv2.imshow('result', img2)
            cv2.imshow('original', img)
            cv2.waitKey(0)
            print("done")
        except Exception as e:
            print(e)
    return img2

"""@arg img: takes an image to apply the smoothing"""
def GaussianSmooth(img):
    """returns the GausKernel"""
    kernel = gkern(21, 1.5)
    """applies the GausKernel img with the Cv2 filter"""
    return cv2.filter2D(img, -1, kernel)

"""@arg kernlen : length of the kernel
@smegma means sigma, but dont google it
returns a 2d array of floats, from gaus smoothing algorithm"""
def gkern(kernlen, smegma):
    g_kernel = cv2.getGaussianKernel(kernlen, smegma)
    g_kernel = g_kernel * g_kernel.transpose()
    return g_kernel


def GaussianPyramid(display):
    img = cv2.imread('stanfordbunny.jpg', 0)
    height, width = img.shape
    nwidth = int(3 / 2 * width + 1)
    """creates a array for next image"""
    nextImage = np.zeros((height, nwidth), dtype=np.uint8)
    gaussImg = GaussianSmooth(img)
    offsetX = 0
    offsetY = 0
    iterations = 7
    """intial image"""
    nextImage[0:gaussImg.shape[0], :gaussImg.shape[1]] = gaussImg
    gaussImg = cv2.resize(gaussImg, None, fx=.5, fy=.5, interpolation=cv2.INTER_LINEAR)
    gaussImg = GaussianSmooth(gaussImg)
    """second image placed to the right of intial image"""
    nextImage[:gaussImg.shape[1], width:] = gaussImg
    offsetY += gaussImg.shape[1]
    gaussImg = cv2.resize(gaussImg, None, fx=.5, fy=.5, interpolation=cv2.INTER_LINEAR)
    gaussImg = GaussianSmooth(gaussImg)
    offsetX = width
    """does the last number of iterations, cant do the first easily due to shifting right initally"""
    for x in range(2, iterations):
        nextImage[offsetY: offsetY + gaussImg.shape[1], offsetX: offsetX + gaussImg.shape[0]] = gaussImg
        offsetY += gaussImg.shape[1]
        gaussImg = cv2.resize(gaussImg, None, fx=.5, fy=.5, interpolation= cv2.INTER_LINEAR)
        gaussImg = GaussianSmooth(gaussImg)
    """@arg display == 1: displays the results, used for method chaining"""
    if display:
        try:
            cv2.imshow('Gaussian Pyramid Image', nextImage)
            cv2.waitKey(0)
            print("done")
        except Exception as e:
            print(e)
    return nextImage


"""if you pass a 1 it will display the results, 0 to not display in case of method chaining"""
# meanblur(1)
# medianblur(1)
# imgsharpen(1)
GaussianPyramid(1)

