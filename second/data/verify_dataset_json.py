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
#              'time': time,
#              'action': action,
#              'img_rel_path': img_rel_path}
#             },
#             ...
#            ]
#     }
# }

root_dir = '/home/azureuser/hackathon_data/hackathon_data'
file_name = 'train_ann.json'

f = open(os.path.join(root_dir, file_name))
ann = json.load(f)

count = 0
for frame_ann in ann['ann']:
    if frame_ann['dir_name'] == '1634698996.767572' and frame_ann['episode_number'] == 0:
        count += 1

a = np.load('/home/azureuser/hackathon_data/hackathon_data/1634698996.767572/processed/ep0.npy')
print(count)