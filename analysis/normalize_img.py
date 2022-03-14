from inspect import formatannotationrelativeto
from typing import final
from xxlimited import new
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from skimage import data
from skimage import color
from skimage import img_as_float

def get_min_max():
    pass

if __name__ == '__main__':
    bc_results_path = "BC_gymcloth"
    # light_brown_multiplier = [0.5, 0.55, 0.85] #BGR
    # dark_brown_multiplier = [0.5, 0.5, 0.6] #BGR

    # img = cv2.imread(img_folder_path)
    # tint_cloth(img, light_brown_multiplier, dark_brown_multiplier)

    minimum = 20
    maximum = 155
    newRange = 255
    oldRange = maximum - minimum

    i = 0
    for folder1 in np.sort(os.listdir(bc_results_path)):
        sub_dir = os.path.join(bc_results_path, folder1)
        for folder2 in np.sort(os.listdir(sub_dir)):
            sub_sub_dir = os.path.join(sub_dir, folder2)
            data = []
            pk_file = sub_sub_dir + "/25x25_real.pkl"
            with open(pk_file, 'rb') as f:
                data = pickle.load(f)
            observations = data['obs']
            for i in range(len(observations)):
                img = np.array(observations[i])
                img = np.where(img > minimum, img, minimum)
                img = np.where(img < maximum, img, maximum)
                var = newRange/oldRange
                img = (img - minimum) * var
                img = img.astype(np.uint8)
                cv2.imwrite("normalized_imgs/%05d.png"%i , img)
                i += 1