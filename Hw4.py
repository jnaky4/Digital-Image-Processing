import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math



# useful visualization of image, shows pixel value at pixel location
def print_save_display(img):
    saveFile = open("display.txt", "w+")

    width, height = img.shape
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            if (img[i][j] != 0):
                print('{:<3}'.format(img[i][j]), end=" ")
                saveFile.write('{:<4}'.format(img[i][j]))
            else:
                print("   ", end=" ")
                saveFile.write("    ")
        print()
        saveFile.write("\n")

    saveFile.close()


"""
    same as binaryThreshold method, except it is designed in a recursive manner
"""
def binaryThreshRecursion(img, oldThres, newThres, diffThres):
    print("applying Recursion")
    displayThresh(img, newThres)
    row, col = img.shape
    if abs(oldThres - newThres) < diffThres:
        return newThres
    low = np.zeros((row, col), dtype=np.int32)
    high = np.zeros((row, col), dtype=np.int32)
    for i in range(1, int(img.shape[0] - 1)):
        for j in range(1, int(img.shape[1] - 1)):
            if img[i][j] > newThres:
                high[i][j] = img[i][j]
            else:
                low[i][j] = img[i][j]
    highCount = cv2.countNonZero(high)
    highSum = np.sum(high)
    highAvg = highSum//highCount
    lowCount = cv2.countNonZero(low)
    lowSum = np.sum(low)

    # print(np.sum(low))
    # print(cv2.countNonZero(low))

    if np.sum(low) != 0 and cv2.countNonZero(low) != 0:
        lowAvg = lowSum // lowCount
        result = (lowAvg + highAvg) // 2
    else:
        result = highAvg

    return binaryThreshRecursion(img, newThres, result, diffThres)


"""
    Initial step for finding a good binary threshold in the image
    takes an image, a initial threshold set at the max pixel value in the image/2
    and a difference tolerance
    
    this method splits the image into two arrays, one above and another below the threshold given
    once complete, calculates the average pixel value in each array
    if the difference is greater than diffThres, it calls the binaryThresRecursion
    until the average difference is less than
"""
def binaryThreshold(img, threshold, diffThres):
    print("step 1")
    displayThresh(img,threshold)
    row, col = img.shape
    low = np.zeros((row, col), dtype=np.int32)
    high = np.zeros((row, col), dtype=np.int32)
    for i in range(1, int(img.shape[0] - 1)):
        for j in range(1, int(img.shape[1] - 1)):
            if img[i][j] > threshold:
                high[i][j] = img[i][j]
            else:
                low[i][j] = img[i][j]
    highCount = cv2.countNonZero(high)
    highSum = np.sum(high)
    highAvg = highSum//highCount
    lowCount = cv2.countNonZero(low)
    lowSum = np.sum(low)
    # ensure not dividing by 0
    if np.sum(low) != 0 and cv2.countNonZero(low) != 0:
        lowAvg = lowSum // lowCount
        newThres = (lowAvg + highAvg) // 2
    else:
        newThres = highAvg
    # lowAvg = lowSum//lowCount
    # newThres = (lowAvg + highAvg) // 2
    return binaryThreshRecursion(img, threshold, newThres, diffThres)

"""
    takes 2 pixel values, and an image array
    checks which is lower
    iterates through the image and sets the higher pixel value to the lower one
    returns the new img array
"""
def joinCluster(value1, value2, img):
    apply = 0
    if value1 == value2:
        return img
    if value1 > value2:
        apply = value2
    else:
        apply = value1

    for i in range(0, int(img.shape[0] - 1)):
        for j in range(0, int(img.shape[1] - 1)):
            if (img[i][j] == value1 or img[i][j] == value2) and img[i][j] != apply:
                img[i][j] = apply

    return img


#
"""
    need a binary image of values 255 and 0
    does a row by row walk of the image
    at each pixel check if the adjacent previously checked pixels are already part of a cluster
    
    initally the first pixel found that is white gets assigned the value 1, as the algorithm
    walks it either finds an adjacent pixel that has a value and assigns it to itself, 
    or it doesnt and assigns a new value to the pixel and increments the counter
    
    if the pixel has 2 neighbors that have been assigned a value ie already apart of a cluster
    then Join cluster is called, which grabs the lower adjacent pixel value and
    replaces all values of the higher pixel value in the image. 
    
    Returns an array the size of the image with the cluster values assigned to each pixel
        - only pixels of 255, the others previously should have been set to zero and are ignored
    

"""
def connected_components(img):
    row, col = img.shape
    cluster_map = np.zeros((row, col), dtype=np.int32)
    counter = 0
    topLeft = False
    topCenter = False
    topRight = False
    left = False

    # iterate through all the pixels
    for i in range(1, int(img.shape[0] - 1)):
        for j in range(1, int(img.shape[1] - 1)):
            # if the pixel is white
            if img[i][j] == 255:
                # check if top left neighbor is assigned to a cluster
                if cluster_map[i-1][j-1] != 0:
                    topLeft = True
                    cluster_map[i][j] = cluster_map[i-1][j-1]
                # check if top center neighbor is assigned to a cluster
                if cluster_map[i-1][j] != 0:
                    topCenter = True
                    if topLeft == True:
                        cluster_map = joinCluster(cluster_map[i-1][j-1], cluster_map[i-1][j], cluster_map)
                    else:
                        cluster_map[i][j] = cluster_map[i-1][j]
                # check if top right neighbor is assigned to a cluster
                if cluster_map[i-1][j+1] != 0:
                    topRight = True
                    if topLeft == True:

                        cluster_map = joinCluster(cluster_map[i-1][j-1], cluster_map[i-1][j+1], cluster_map)
                    elif topCenter == True:
                        cluster_map = joinCluster(cluster_map[i-1][j], cluster_map[i-1][j+1], cluster_map)
                    else:
                        cluster_map[i][j] = cluster_map[i-1][j+1]
                # check if left neighbor is assigned to a cluster
                if cluster_map[i][j-1] != 0:
                    left = True
                    if topLeft == True:
                        cluster_map = joinCluster(cluster_map[i - 1][j - 1], cluster_map[i][j-1], cluster_map)
                    elif topCenter == True:
                        cluster_map = joinCluster(cluster_map[i-1][j], cluster_map[i][j-1], cluster_map)
                    elif topRight == True:
                        cluster_map = joinCluster(cluster_map[i-1][j+1], cluster_map[i][j-1], cluster_map)
                    else:
                        cluster_map[i][j] = cluster_map[i][j-1]
                if topLeft == False and topCenter == False and topRight == False and left == False:
                    cluster_map[i][j] = counter
                    counter += 1
                topLeft = False
                topCenter = False
                topRight = False
                left = False

    return cluster_map


"""
    Method to display the image threshold
    takes the threshold and an image array
    iterates through the image, and sets values below threshold to 0
    and values above to 255
    then displays the result once plt.show is called
    also returns the altered image if desired
"""
def displayThresh(img, threshold):
    temp = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    for i in range(1, int(img.shape[0] - 1)):
        for j in range(1, int(img.shape[1] - 1)):
            if img[i][j] <= threshold:
                temp[i][j] = 0
            else:
                temp[i][j] = 255
    f = plt.figure()
    plt.imshow(temp, cmap="gray")
    plt.title('Threshold applied')
    return temp

"""
    computes "distance" between two values, using Euclids formula
"""
def compute_euclidean_distance(color1, color2):
    return math.sqrt(math.pow((color1 - color2), 2))

"""
    Takes a image processed by connected components and colors the pixels based on
    the value assigned to each cluster
    only uses 8 colors by getting the value % number of colors
"""
def color_components(img):
    colors = [(255,0,0), (255,128,0), (255,255,0), (0,255,0), (0,0,255), (127,0,255), (255,0,255), (255,0,127)]
    colored_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(0, int(img.shape[0] - 1)):
        for j in range(0, int(img.shape[1] - 1)):
            if img[i][j] != 0:
                # print(colors[(img[i][j]%8)])
                colored_image[i][j] = colors[((img[i][j] -1)%len(colors))]

    f = plt.figure()
    plt.imshow(colored_image)
    plt.title('colorized connected components')
    return colored_image


def kMeans(img, k):
    width, height = img.shape
    colorVals = generatePoints(k)
    distFromCentroids = np.zeros(k, dtype=float)
    belongToCluster = 0
    editImg = img

    # Getting the euclidian distances from pixel color to each assigned cluster centroid color
    for i in range(0, width - 1):
        for j in range(0, height - 1):
            # For each k value
            min = 255
            # Use this to evalute distance from centroids
            for kval in range(0, k):
                distFromCentroids[kval] = compute_euclidean_distance(editImg[i][j], colorVals[kval])
                if(distFromCentroids[kval] < min):
                    min = distFromCentroids[kval]
                    belongToCluster = kval
            editImg[i][j] = colorVals[belongToCluster]

    f1 = plt.figure()
    plt.imshow(editImg, cmap="gray")
    plt.title('Values based off random')

    count = 1
    oldColorVals = colorVals
    newCluster, colorVals = recurseAverage(editImg, k, colorVals)
    while( not((oldColorVals[0] > colorVals[0] - 5) and (oldColorVals[0] < colorVals[0] + 5)) ):
        print("Recursive iteration: " + str(count))
        print("Old centroid val: " + str(oldColorVals[0]) + " New centroid val: " + str(colorVals[0]))

        count += 1
        oldColorVals = colorVals
        newCluster, colorVals = recurseAverage(newCluster, k)

        f2 = plt.figure()
        plt.imshow(newCluster, cmap="gray")
        plt.title('Values based off consecutive iterations of kmeans')
    print("Done")

    f3 = plt.figure()
    plt.imshow(newCluster, cmap="gray")
    plt.title('Refined values')
    # print_save_display(newCluster)
    # print(type(newCluster))
    return newCluster


def recurseAverage(clusters, k, colorVals):
    sumColors = np.zeros(k, dtype=int) # The sum in order to find the centroid with realtion to the k values
    timesHit = np.zeros(k, dtype=int)
    distFromCentroids = np.zeros(k, dtype=float)
    newClusters = clusters
    width, height = clusters.shape

    for i in range(0, width - 1):
        for j in range(0, height - 1):
            for kval in range(0, k):
                if(newClusters[i][j] == colorVals[kval]):
                    sumColors[kval] += newClusters[i][j]
                    timesHit[kval] += 1

    for kval in range(0, k):
        if(timesHit[kval] != 0):
            colorVals[kval] = int(sumColors[kval] / timesHit[kval])
        else:
            colorVals[kval] = 0

    for i in range(0, width - 1):
        for j in range(0, height - 1):
            # For each k value
            min = 1000
            # Use this to evalute distance from centroids
            for kval in range(0, k):
                distFromCentroids[kval] = compute_euclidean_distance(newClusters[i][j], colorVals[kval])
                if (distFromCentroids[kval] < min):
                    min = distFromCentroids[kval]
                    belongToCluster = kval
            newClusters[i][j] = colorVals[belongToCluster]

    return newClusters, colorVals


def generatePoints(k):
    randColors = np.zeros(k, dtype=int)
    for i in range(0, k):
        randColors[i] = random.randint(0, 255)  # Not truely random but wanted better points in order to check
        print("Random Color " + str(i + 1) + ": " + str(randColors[i]))

    return randColors

"""
    takes an image and a list of already used pixel values
    returns the first value found in the image that is not zero and isnt in the list

"""
def getfirstNonZero(img, alreadyDone):
    temp_set = set(alreadyDone)
    width, height = img.shape
    for i in range(0, width):
        for j in range(0, height):
            if img[i][j] != 0 and (img[i][j] not in temp_set):
                return img[i][j]

"""
    takes an image and a desired pixel value,
    iterates through the image and sets all values == to pixel value 'index'
    and all not equal to zero
    returns the new array
"""
def clean_background(img, index):
    row, col = img.shape

    # print("row", row)
    # print("col", col)
    cleaned_image = np.zeros([row, col], dtype=np.int32)
    for i in range(0, row):
        for j in range(0, col):
                if img[i][j] == index:
                    cleaned_image[i][j] = 255
                else:
                    cleaned_image[i][j] = 0
    # print("cleaned img", cleaned_image)
    # print_save_display(cleaned_image)
    return cleaned_image

"""
    takes an image and a pixel value, and sets all pixels == to value to zero
    returns the new array
"""
def remove_pixel_value(img, value):
    # print(value)
    # print_save_display(img)
    row, col = img.shape
    cleaned_image = np.zeros([row, col], dtype=np.int32)

    for i in range(0, row):
        for j in range(0, col):
                if img[i][j] != value:
                    cleaned_image[i][j] = img[i][j]
    # print_save_display(cleaned_image)
    return cleaned_image

def load():
    # Load the image as greyscale
    img = cv2.imread('img/3.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def main():
    """
    Part 1
    """
    img = load()
    f = plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title('Original')

    thresh = binaryThreshold(img, (img.max()/2), 10)
    print("Threshold Set: ", thresh)
    plt.show()


    """
    Part 2
    """
    # ultimately sets the number of colors you want to divide the image into
    numcomponents = 4
    # temporary arrays used for image processing
    maskArray = []
    tempimgArray = []
    filterArray = []
    layersArray = []
    componentsdone = []

    img = load()




    clusters = kMeans(img, numcomponents)
    # create new image so original does not get overrwritten
    temp_img = clusters
    # create an array that will be the size of numcompnents
    tempimgArray.append(temp_img)
    print("starting connected components process, this takes awhile")

    # process for connected components, does this for each color
    for i in range(numcomponents):
        # find fist non zero pixel in image
        component = getfirstNonZero(tempimgArray[i], componentsdone)
        # add that to the already done so getfistNonZero wont return the same pixel
        componentsdone.append(component)
        print("pixel value getting isolated: ", component)

        # remove every pixel except component pixel
        # append the image of only that pixel value to the maskArray
        maskArray.append(clean_background(tempimgArray[i], component))
        # # remove that pixel value from the img
        tempimgArray.append(remove_pixel_value(clusters, component))




    # take the isolated layers and applies connected components algorithm on each one
    # then colors the pixels in the image
    for i in range(numcomponents):
        filterArray.append(connected_components(maskArray[i]))
        layersArray.append(color_components(filterArray[i]))

    plt.show()


if __name__ == '__main__':
    main()
