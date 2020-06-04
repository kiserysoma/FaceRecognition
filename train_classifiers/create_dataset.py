import os
import cv2
import pickle
import random
from collections import Counter
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import imutils

IMAGE_PATH = '/media/Projects/shared/face_dataset/'
DST_PATH = '/media/Projects/shared/face_dataset_v2/'

def copy_images():
    imgs = pickle.load( open ('img_names.p', 'rb') )
    IDs = []
    indexes = []
    buffer = []

    for i in range(len(imgs)):
        index = imgs[i][0:5].find('_')
        ID = imgs[i][0:index]
        if ID not in IDs:
            indexes.append(i)
            IDs.append(ID)
        buffer.append(ID)

    for i in range(len(IDs)):
        for j in range(10):
            if i != len(IDs)-1:
                r1 = indexes[i]
                r2 = indexes[i+1]
            else:
                r1 = indexes[i]
                r2 = (len(imgs)-1)
            randn = random.randint(r1, r2)

            chosen_img = imgs[randn]
            img = cv2.imread(IMAGE_PATH + chosen_img)
            cv2.imwrite(DST_PATH + chosen_img, img)

copy_images()

def random_rotation(image):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-10, 10)
    return imutils.rotate(image, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

buffer = []
IDs = []
indexes = []
original_images = sorted(os.listdir(DST_PATH))
print(original_images)

for i in range(len(original_images)):
    index = original_images[i][0:5].find('_')
    ID = original_images[i][0:index]
    if ID not in IDs:
        indexes.append(i)
        IDs.append(ID)
    buffer.append(ID)
most_common = [item for item in Counter(buffer).most_common()]
print(most_common)

available_transformations = {
    'rotate': random_rotation,
    #'noise': random_noise,
}

for i in range(len(IDs)):
    if i == len(IDs)-1:
        current_images = original_images[indexes[len(indexes)-1]:]
    else:
        current_images = original_images[indexes[i]:indexes[i+1]]
    to200 = 200-len(current_images)
    print(to200)
    for j in range(to200):
        img_name = random.choice(current_images)
        #print(img_name)
        img = cv2.imread(DST_PATH + img_name)
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        while num_transformations <= num_transformations_to_apply:
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](img)
            num_transformations += 1
        dst = DST_PATH + img_name[:len(img_name)-4] + "_" + str(j) + '.jpg'
        print(dst)
        cv2.imwrite(dst, transformed_image)
