import sys
import os
import json
import matplotlib.pyplot as plt
import natsort
import numpy as np
import glob

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
        base_numpy = os.path.join(dir_name, 'processed')
        numpy_list = natsort.natsorted(glob.glob(os.path.join(root_dir, base_numpy, '*.npy')))

        for i, numpy_file in enumerate(numpy_list):
            episode_name = dir_name + '_' + str(i)
            ann['ann'][episode_name] = []  # initialize empty list
            data = np.load(numpy_file)
            num_images = data.shape[0]
            times = data[:, 0]
            actions = data[:, 1]
            for j in range(num_images):
                # add each individual image in the dataset dict
                img_occupancy_rel_path = os.path.join(dir_name, 'processed_images', str(i), str(j)+'.png')
                img_bev_rel_path = os.path.join(dir_name, 'processed_images_bev', str(i), str(j)+'.png')
                frame_ann = {
                    'dir_name' : dir_name,
                    'episode_number': i,
                    'img_number': j,
                    'video_len' : num_images,
                    'time' : times[j],
                    'action' : actions[j],
                    'img_occupancy_rel_path' : img_occupancy_rel_path,
                    'img_bev_rel_path' : img_bev_rel_path
                }
                if np.isnan(frame_ann['action']):
                    frame_ann['action'] = 0.0
                ann['ann'][episode_name].append(frame_ann)
                total_frame_count += 1
                total_frame_dir += 1
        print(f'Processed {total_frame_dir} frames in this directory...')
    print(f'Data loading complete. Processed {total_frame_count} frames in total...')
    return ann


root_dir = '/home/azureuser/hackathon_data/hackathon_data'
dir_names = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]

# A small one for debugging.
ann = gen_ann(dir_names[:1], root_dir)
with open(os.path.join(root_dir, 'train_ann_debug.json'), 'w') as f:
    json.dump(ann, f, indent=4)

# Full training set.
ann = gen_ann(dir_names, root_dir)
with open(os.path.join(root_dir, 'train_ann.json'), 'w') as f:
    json.dump(ann, f, indent=4)

