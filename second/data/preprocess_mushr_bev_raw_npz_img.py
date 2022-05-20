import pathlib
import pickle
import time
from collections import defaultdict
from functools import partial
import os
import glob
import sys

import cv2
import numpy as np
from numpy.core.fromnumeric import shape
from skimage import io as imgio

from PIL import Image
import asyncio

from joblib import Parallel, delayed, parallel
import natsort

# imports
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..', '..')
sys.path.insert(0, import_path)
# from second.core import box_np_ops
# from second.core import preprocess as prep
# from second.core.geometry import points_in_convex_polygon_3d_jit
# from second.data import kitti_common as kitti
# from second.utils import simplevis
# from second.utils.timer import simple_timer

from second.utils.mapping import mapping
# from second.utils.raycast import raycast

import seaborn as sns
import matplotlib.pyplot as plt

import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


# @background
def process_episode(input):
    episode_name = input[0]
    episode_folder = input[1]
    print("Processing episode: " + episode_name)
    data = np.load(episode_name)
    times = data['ts']
    actions = data['angles']
    images = data['images']
    num_images = times.shape[0]
    for i in range(num_images):
        bgr_img = np.zeros(shape=images[i,:].shape, dtype=np.int8)
        bgr_img[:, :, 0] = images[i,:,:,2]
        bgr_img[:, :, 1] = images[i,:,:,1]
        bgr_img[:, :, 2] = images[i,:,:,0]
        bgr_img = Image.fromarray(np.uint8(bgr_img))
        img_filename = os.path.join(episode_folder, str(i)+'.png')
        bgr_img.save(img_filename)


def process_folder(folder, processed_dataset_path, output_folder_name):
    episodes_list = natsort.natsorted(glob.glob(os.path.join(folder, 'processed_withpose2/*.npz')))
    total_n_episodes = len(episodes_list)
    print("Total number of episodes to be processed = {}".format(total_n_episodes))
    parallel_processing = True
    # parallel_processing = False
    if parallel_processing is True:
        inputs_list = []
        for i, episode_name in enumerate(episodes_list, 0):
            episode_folder = os.path.join(folder, output_folder_name, str(i))
            if not os.path.isdir(episode_folder):
                os.makedirs(episode_folder)
                # move on to process that bag file inside the folder
                inputs_list.append((episode_name, episode_folder))
            else:
                print('Warning: episode path already exists. Skipping this episode...')
        Parallel(n_jobs=36)(delayed(process_episode)(input) for input in inputs_list)
    else:
        for i, episode_name in enumerate(episodes_list, 0):
            print("Beginning episode number = {} out of {}: {}%".format(i, total_n_episodes, int(100.0*i/total_n_episodes)))
            episode_folder = os.path.join(folder, output_folder_name, str(i))
            if not os.path.isdir(episode_folder):
                os.makedirs(episode_folder)
                # move on to process that bag file inside the folder
                process_episode((episode_name, episode_folder))
            else:
                print('Warning: episode path already exists. Skipping this episode...')

# define script parameters
base_folder = '/home/azureuser/hackathon_data/hackathon_data_2p5_nonoise2'
output_folder_name = 'processed_fpvimages'
# folders_list = sorted(glob.glob(os.path.join(base_folder, '*')))
folders_list = sorted([os.path.join(base_folder,x) for x in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, x))])
total_n_folders = len(folders_list)
print("Total number of folders to be processed = {}".format(total_n_folders))

for i, folder in enumerate(folders_list, 1):
    print("Beginning folder number = {} out of {}: {}%".format(i, total_n_folders, int(100.0*i/total_n_folders)))
    # create processed folder
    processed_dataset_path = os.path.join(folder, output_folder_name)
    if not os.path.isdir(processed_dataset_path):
        os.makedirs(processed_dataset_path)
        # move on to process that bag file inside the folder
        process_folder(folder, processed_dataset_path, output_folder_name)
    else:
        print('Warning: path already exists. Skipping this folder...')
