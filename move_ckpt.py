import os
import shutil

processed_folder="./preprocessed"
ckpt_folder="./pretrained"

for elem in os.listdir(ckpt_folder):
    ckpt_elem=os.path.join(ckpt_folder,elem,"gaussian_scene_fea_dev")
    shutil.move(ckpt_elem,os.path.join(processed_folder,elem,"gaussian","gaussian_scene_fea_dev"))