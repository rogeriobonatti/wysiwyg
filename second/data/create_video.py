import glob
import os
import cv2
import natsort

# define script parameters
# files_folder = '/home/azureuser/hackathon_data/hackathon_data/1634698996.767572/processed_images/0'
# out_path = '/home/azureuser/hackathon_data/hackathon_data/1634698996.767572/processed_images/0/bla/movie.avi'

files_folder = '/home/azureuser/hackathon_data/model_eval/1642212421.0042143'
out_path = '/home/azureuser/hackathon_data/model_eval/1642212421.avi'

img_list = natsort.natsorted(glob.glob(os.path.join(files_folder, '*.png')))
total_n_imgs = len(img_list)
print("Total number of images to be processed = {}".format(total_n_imgs))

img_array = []
for filename in img_list:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
