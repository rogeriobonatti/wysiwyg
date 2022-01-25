import sys
import os
import json
import matplotlib.pyplot as plt
import natsort
import numpy as np
import glob
import math

# Format: 
# {
#     'type': 'mushr_sim_pretrain', 
#     'ann': [
#             {'dir_name': dir_name, 
#              'episode_number': episode_number,
#              'img_number': img_number,
#              'video_len': video_len, 
#              'time': time,
#              'action': action,
#              'img_occupancy_rel_path' : img_occupancy_rel_path,
#              'img_bev_rel_path' : img_bev_rel_path}
#             },
#             ...
#            ]
#     }
# }


def gen_ann(dir_names, root_dir):
    ann = {'type': 'mushr_sim_pretrain', 'ann': {}}

    total_frame_count = 0
    
    for dir_name in dir_names:
        total_frame_dir = 0
        print(f'Processing {dir_name}...')
        # go inside every single episode from this directory
        base_numpy = os.path.join(dir_name, 'processed_withpose2')
        numpy_list = natsort.natsorted(glob.glob(os.path.join(root_dir, base_numpy, '*.npz')))

        if dir_name == '1634699222.269483':
            bla=1

        for i, numpy_file in enumerate(numpy_list):
            episode_name = dir_name + '_' + str(i)
            ann['ann'][episode_name] = []  # initialize empty list
            data = np.load(numpy_file)
            times = data['ts']
            actions = data['angles']
            # lidars = data['lidars']
            poses = data['poses']
            goal = data['goal']
            num_images = times.shape[0]
            for j in range(num_images):
                # add each individual image in the dataset dict
                img_occupancy_rel_path = os.path.join(dir_name, 'processed_images_occupancy2', str(i), str(j)+'.png')
                img_bev_rel_path = os.path.join(dir_name, 'processed_images_bev_fixed6', str(i), str(j)+'.png')
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
                    'img_occupancy_rel_path' : img_occupancy_rel_path,
                    'img_bev_rel_path' : img_bev_rel_path
                }
                ann['ann'][episode_name].append(frame_ann)
                total_frame_count += 1
                total_frame_dir += 1
        print(f'Processed {total_frame_dir} frames in this directory...')
    print(f'Data loading complete. Processed {total_frame_count} frames in total...')
    return ann


root_dir = '/home/azureuser/hackathon_data/hackathon_data_slow_nonoise'
# root_dir = '/home/azureuser/hackathon_data/real'
dir_names = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]

# A small one for debugging and validation
ann = gen_ann(dir_names[:2], root_dir)
with open(os.path.join(root_dir, 'train_ann_pose_debug.json'), 'w') as f:
    json.dump(ann, f, indent=4)

# Full training set.
ann = gen_ann(dir_names[2:], root_dir)
with open(os.path.join(root_dir, 'train_ann_pose.json'), 'w') as f:
    json.dump(ann, f, indent=4)

