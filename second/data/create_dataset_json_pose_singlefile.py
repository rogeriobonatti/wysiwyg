import sys
import os
import json
import matplotlib.pyplot as plt
import natsort
import numpy as np
import glob
import math


def gen_ann(dir_names, root_dir):
    ann = {'type': 'mushr_sim_pretrain', 'ann': {}}

    total_frame_count = 0
    
    for dir_name in dir_names:
        total_frame_dir = 0
        print(f'Processing {dir_name}...')
        # go inside every single episode from this directory
        base_numpy = os.path.join(dir_name, 'processed_withpose2')
        numpy_list = natsort.natsorted(glob.glob(os.path.join(root_dir, base_numpy, '*.npz')))

        # if dir_name == '1634699222.269483':
        #     bla=1

        for i, numpy_file in enumerate(numpy_list):
            episode_name = dir_name + '_' + str(i)
            ann['ann'][episode_name] = {}  # initialize empty dict
            
            # add episode metadata:
            all_images_path = os.path.join(dir_name, 'processed_images_bev_fixed_8m', str(i), 'all_images.npy')
            all_pcls_path = os.path.join(dir_name, 'processed_images_bev_fixed_8m', str(i), 'all_pcls.npy')
            meta_data = {
                'all_images_path' : all_images_path,
                'all_pcls_path' : all_pcls_path
            }
            ann['ann'][episode_name]['metadata'] = meta_data
            
            # add the data about each step of the episode
            ann['ann'][episode_name]['data'] = []
            data = np.load(numpy_file)
            times = data['ts']
            actions = data['angles']
            # lidars = data['lidars']
            poses = data['poses']
            goal = data['goal']
            num_images = times.shape[0]
            for j in range(num_images):
                # add each individual image in the dataset dict
                # img_occupancy_rel_path = os.path.join(dir_name, 'processed_images_occupancy2', str(i), str(j)+'.png')
                # img_bev_rel_path = os.path.join(dir_name, 'processed_images_bev_fixed10', str(i), str(j)+'.png')
                # pcl_rel_path = os.path.join(dir_name, 'processed_images_bev_fixed10', str(i), str(j)+'.npy')
                # change the action in case there is a None value
                action = actions[j,0]
                if np.isnan(action) and j>0:
                    action = actions[j-1,0]
                elif np.isnan(action) and j==0:
                    action = 0.0
                pose = poses[j].tolist()
                if np.isnan(pose).any() and j>0:
                    pose = poses[j-1].tolist()
                elif np.isnan(pose).any() and j==0: 
                    pose = [0.0, 0.0, 0.0]
                frame_ann = {
                    'dir_name' : dir_name,
                    'episode_number': i,
                    'img_number': j,
                    'goal': goal.tolist(),
                    'video_len' : num_images,
                    'time' : times[j,0],
                    'action' : action,
                    'pose' : pose,
                    # 'img_occupancy_rel_path' : img_occupancy_rel_path,
                    # 'img_bev_rel_path' : img_bev_rel_path,
                    # 'pcl_rel_path' : pcl_rel_path
                }
                ann['ann'][episode_name]['data'].append(frame_ann)
                total_frame_count += 1
                total_frame_dir += 1
            # to make super small dataset
            # if i==5:
            #     break
        print(f'Processed {total_frame_dir} frames in this directory...')
    print(f'Data loading complete. Processed {total_frame_count} frames in total...')
    return ann


# root_dir = '/home/azureuser/hackathon_data_premium/hackathon_data_2p5_withfullnoise0'
root_dir = '/home/azureuser/hackathon_data_premium/hackathon_data_2p5_withpartialnoise0'
# root_dir = '/home/azureuser/hackathon_data/real'
dir_names = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]

# supersmall
# ann = gen_ann(dir_names[:1], root_dir)
# with open(os.path.join(root_dir, 'singlefile_train_ann_pose_supersmall8m.json'), 'w') as f:
#     json.dump(ann, f, indent=4)

# A small one for debugging and validation
ann = gen_ann(dir_names[:4], root_dir)
with open(os.path.join(root_dir, 'singlefile_train_ann_pose_debug8m.json'), 'w') as f:
    json.dump(ann, f, indent=4)

# Full training set.
ann = gen_ann(dir_names[4:], root_dir)
with open(os.path.join(root_dir, 'singlefile_train_ann_pose8m.json'), 'w') as f:
    json.dump(ann, f, indent=4)

