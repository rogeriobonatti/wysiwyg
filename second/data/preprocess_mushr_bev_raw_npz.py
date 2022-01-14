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

def background(f):
    async def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

def load_params(data_col):
    condition = (data_col[2:]<12.0) & (data_col[2:]>0.5) & (~np.isnan(data_col[2:]))
    ok_R = np.extract(condition, data_col[2:])
    num_points = ok_R.shape[0]
    # angles = np.linspace(0, 2*np.pi, 720)*-1.0 + np.pi # aligned in car coordinate frame (because ydlidar points backwards)
    angles = np.linspace(0, 2*np.pi, 720)
    ok_angles = np.extract(condition, angles)
    original_points = np.zeros(shape=(num_points, 3), dtype=float) # leave z as zero always, just change X and Y next
    # car coord points x forward, y right. car front points up in the picture
    # ydlidar has zero deg pointing backwards in the car, and angle grows clock-wise
    # original_points[:,0] = -np.cos(ok_angles)*ok_R # X
    # original_points[:,1] = -np.sin(ok_angles)*ok_R # Y
    original_points[:,0] = -np.cos(ok_angles)*ok_R # X
    original_points[:,1] = np.sin(ok_angles)*ok_R # Y
    voxel_size = 0.1
    range_x = 10.0
    range_y = 10.0
    range_z = voxel_size/2.0
    pc_range = np.array([-range_x, -range_y, -range_z, range_x, range_y, range_z], dtype=float)
    lo_occupied = np.log(0.7 / (1 - 0.7))
    lo_free = np.log(0.4 / (1 - 0.4))
    sensor_origins = np.tile(np.array([range_x, range_y, range_z]) , (num_points, 1)).astype(float)
    # original_points = original_points + sensor_origins # add sensor origin offset in the map 
    time_stamps = np.repeat(data_col[0], num_points).astype(float)
    return original_points, sensor_origins, time_stamps, pc_range, voxel_size, lo_occupied, lo_free


def grid2lin_idx(grid_idx, sx, sy, sz):
    return int(grid_idx[0]+sx*grid_idx[1]+sy*sx*grid_idx[2])

def lin2grid_idx(lin_idx, sx, sy, sz):
    grid_idx = np.zeros(shape=(3,))
    grid_idx[0] = lin_idx%sx
    lin_idx = int(lin_idx/sx)
    grid_idx[1] = lin_idx%sy
    lin_idx = int(lin_idx/sy)
    grid_idx[2] = lin_idx
    return grid_idx.astype(int)

def convert_logodds2img(logodds, sx, sy, sz):
    img = np.zeros(shape=(sx,sy))
    for i in range(logodds.size):
        grid_idx = lin2grid_idx(i, sx, sy, sz)
        img[grid_idx[0]][grid_idx[1]] = logodds[i]
    return img

def compute_bev_image(original_points, sensor_origins, time_stamps, pc_range, voxel_size):
    nx = int(np.floor((pc_range[3]-pc_range[0])/voxel_size))
    ny = int(np.floor((pc_range[4]-pc_range[1])/voxel_size))
    vis_mat = -1.0* np.ones(shape=(ny, nx))
    original_points_idx = np.floor(original_points / voxel_size).astype(int) # becomes relative indexes instead of meters
    # transform from car-centered pixels towards standard image reference frame on upper left corner
    # Y is rows, down and X is cols, to the right
    points_vis_idx = np.zeros(shape=original_points_idx.shape, dtype=int)
    points_vis_idx[:,0] = int(ny/2)-original_points_idx[:,0] # y dir in image
    points_vis_idx[:,1] = int(nx/2)+original_points_idx[:,1] # x dir in image
    # remove indexes out of bounds for the image
    filtered_points_idx = points_vis_idx[(points_vis_idx[:,0]>0) & \
                                         (points_vis_idx[:,0]<=(ny-1)) & \
                                         (points_vis_idx[:,1]>0) & \
                                         (points_vis_idx[:,1]<=(nx-1))] 
    for p in filtered_points_idx:
        vis_mat[p[0], p[1]] = 1.0
    return vis_mat


# @background
def process_episode(input):
    episode_name = input[0]
    episode_folder = input[1]
    print("Processing episode: " + episode_name)
    data = np.load(episode_name)
    times = data['ts']
    actions = data['angles']
    lidars = data['lidars']
    # poses = data['poses']
    # goal = data['goal']
    num_images = times.shape[0]
    data = np.concatenate((times, actions, lidars), axis=1)
    for i in range(num_images):
        original_points, sensor_origins, time_stamps, pc_range, voxel_size, lo_occupied, lo_free = load_params(data[i, :])
        
        # logodds = mapping.compute_logodds(original_points, sensor_origins, time_stamps, pc_range, voxel_size, lo_occupied, lo_free)
        # logodds_mat = convert_logodds2img(logodds, 200, 200, 1)
        # plt.imshow(logodds_mat)
        # plt.savefig('bla.png')

        # compute visibility
        vis_mat = compute_bev_image(original_points, sensor_origins, time_stamps, pc_range, voxel_size)
        vis_mat = vis_mat*127+127
        im = Image.fromarray(np.uint8(vis_mat), 'L')
        img_filename = os.path.join(episode_folder, str(i)+'.png')
        im.save(img_filename)
        # print(img_filename)
        # plt.imshow(vis_mat)
        # plt.savefig('bla.png')


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
base_folder = '/home/azureuser/hackathon_data/hackathon_data'
output_folder_name = 'processed_images_bev_fixed6'
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
