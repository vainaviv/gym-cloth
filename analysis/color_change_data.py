import os
import argparse
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import  cv2
from skimage import data
from skimage import color
from skimage import img_as_float
from tint_cloth import tint_cloth

def tint_data(observation):
    light_brown_multiplier = [1.0, 1.0, 1.0] #BGR
    dark_brown_multiplier = [0.5, 0.5, 0.6] #BGR
    background_multiplier = [0.9, 0.9, 0.9]
    new_obs = []
    for i in range(len(observation)):
        tinted_step = tint_cloth(observation[i], light_brown_multiplier, dark_brown_multiplier, background_multiplier)
        new_obs.append(tinted_step)
    return new_obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='logs/data_25x25_2000_iters.pkl')
    args = parser.parse_args()

    new_data_loc = 'logs/data_25x25_match_real.pkl'

    combined = []
    with open(args.file, 'rb') as f:
        combined = pickle.load(f)

    for i in range(0, len(combined)):
        print("Processing iteration %d" % (i))
        observation = combined[i]['obs']
        tinted_obs = tint_data(observation)
        combined[i]['obs'] = tinted_obs

    with open(new_data_loc, 'wb') as f:
        pickle.dump(combined, f)
