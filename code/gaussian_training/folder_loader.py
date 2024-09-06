import os
import os.path
import cv2
import torch
import numpy as np
from natsort import natsorted
import json
from torch.utils.data import Dataset
import ipdb

def remod(x, mod=64):
    return ((x-1)//mod + 1)*mod


# add split to select key frames 
class Folder_Dataset(Dataset):
    def __init__(self, data_dir, start_rate = 0., end_rate = 1., skip_length=0):
        self.data_dir = data_dir
        seg_mask_dir=os.path.join(data_dir,'seg_masks')
        parsing_mask_dir=os.path.join(data_dir,'parsing')
        ori_img_dir=os.path.join(data_dir,'ori_imgs')
        train_img_dir=os.path.join(data_dir,'train_imgs')
        landmark_dir=os.path.join(data_dir,'landmarks')

        self.seg_mask_paths = natsorted([os.path.join(seg_mask_dir, f) for f in os.listdir(seg_mask_dir) if f.endswith('.png')])
        self.parsing_mask_paths = natsorted([os.path.join(parsing_mask_dir, f) for f in os.listdir(parsing_mask_dir) if f.endswith('.png')])
        self.ori_img_paths = natsorted([os.path.join(ori_img_dir, f) for f in os.listdir(ori_img_dir) if f.endswith('.jpg')])
        self.train_img_paths = natsorted([os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.png')])
        self.landmark_paths = natsorted([os.path.join(landmark_dir, f) for f in os.listdir(landmark_dir) if f.endswith('.lms')])

        # ipdb.set_trace()
        tmp_img = cv2.imread(self.ori_img_paths[0])
        self.img_size = tmp_img.shape[:2]

        start_id = int(len(self.ori_img_paths) * start_rate + .5)
        end_id = int(len(self.ori_img_paths) * end_rate + .5)

        if skip_length < 1:
            self.load_ids = np.arange(start_id, end_id)
        else:
            self.load_ids = np.arange(start_id, end_id)[::skip_length]

        print("Frame len: {}".format(self.load_ids.shape[0]))

        self.preload_to_cpu()


    def preload_to_cpu(self):
        self.image_list = []

        for i in range(len(self.load_ids)):
            load_idx = self.load_ids[i]
            img_gt = cv2.imread(self.train_img_paths[load_idx], cv2.IMREAD_UNCHANGED)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGRA2RGBA)[:, :, :3]

            self.image_list.append(img_gt)
        self.image_list = torch.tensor(np.array(self.image_list))

    def __len__(self):
        return self.load_ids.shape[0]

    def __getitem__(self, idx):
        load_idx = self.load_ids[idx]
        img_gt = cv2.imread(self.train_img_paths[load_idx], cv2.IMREAD_UNCHANGED)
        ori_img_gt_path = self.ori_img_paths[load_idx]
        seg_mask = cv2.imread(self.seg_mask_paths[load_idx], cv2.IMREAD_UNCHANGED)
        parsing_mask = cv2.imread(self.parsing_mask_paths[load_idx], cv2.IMREAD_UNCHANGED)
        lms = np.loadtxt(self.landmark_paths[load_idx], dtype=np.float32)

        img_gt = torch.from_numpy(cv2.cvtColor(img_gt, cv2.COLOR_BGRA2RGBA))
        seg_mask = torch.from_numpy(cv2.cvtColor(seg_mask, cv2.COLOR_BGR2RGB))
        parsing_mask = cv2.cvtColor(parsing_mask, cv2.COLOR_BGR2RGB)

        # 'label_names':[ 'background','neck','face','cloth','rr','Ir','rb','lb','re', 'le','nose','imouth','llip','ulip','hair', 'eyeg','hat','earr','neck_1']

        head_ids = np.array(( 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), dtype=np.uint8)
        head_mask = np.isin(parsing_mask, head_ids).astype(np.float32)
        head_mask = torch.from_numpy(head_mask).to(dtype=torch.float32)*255

        hair_ids = np.array(( 14), dtype=np.uint8)
        hair_mask = np.isin(parsing_mask, hair_ids).astype(np.float32)
        hair_mask = torch.from_numpy(hair_mask).to(dtype=torch.float32)*255

        face_ids = np.array(( 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), dtype=np.uint8)
        face_mask = np.isin(parsing_mask, face_ids).astype(np.float32)
        face_mask = torch.from_numpy(face_mask).to(dtype=torch.float32)*255

        head_eye_g_ids = np.array(( 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), dtype=np.uint8)
        head_eye_g_mask = np.isin(parsing_mask, head_eye_g_ids).astype(np.float32)
        neck_ids = np.array(( 1), dtype=np.uint8)
        neck_mask = np.isin(parsing_mask, neck_ids).astype(np.float32)
        kernel = np.ones((5,5),np.uint8)
        head_eye_g_mask = cv2.dilate(head_eye_g_mask,kernel,iterations=5)
        head_eye_g_mask = np.maximum(head_eye_g_mask,neck_mask)
        head_eye_g_mask = torch.from_numpy(head_eye_g_mask).to(dtype=torch.float32)*255
        
        nonzero_head=torch.nonzero(head_mask[:,:,0])
        left = torch.min(nonzero_head[:, 1]).numpy()
        right = torch.max(nonzero_head[:, 1]).numpy()
        top=torch.min(nonzero_head[:,0]).numpy()
        down=torch.max(nonzero_head[:,0]).numpy()

        # a head box
        old_size = np.ceil((right - left + down - top) / 2 * 1.3)
        center_x = right - (right - left) / 2.0
        center_y =  down - (down - top) / 2.0

        left_face = np.ceil(max(center_x-old_size/2, 0))
        top_face = np.ceil(max(center_y-old_size/2, 0))
        box_l = min(old_size, img_gt.shape[0]-left_face, img_gt.shape[1]-top_face)

        # a bigger head box
        head_old_size = np.ceil(max(right - left, down - top) * 1.7)
        head_center_x = right - (right - left) / 2.0
        head_center_y =  max(down - (down - top) / 2.0,0)

        left_head = np.ceil(max(head_center_x-head_old_size/2, 0))
        top_head = np.ceil(max(head_center_y-head_old_size/2, 0))
        head_box_l = min(head_old_size, img_gt.shape[0]-left_head, img_gt.shape[1]-top_head)

        
        data_dict = {
            "img_gt": img_gt,
            "load_idx": load_idx,
            "bbox_tensor": torch.tensor([left_face, top_face, box_l, box_l], dtype=torch.int32),
            "head_bbox_tensor": torch.tensor([left_head, top_head, head_box_l, head_box_l], dtype=torch.int32),
            "head_mask": head_mask,
            "idx": idx,
            "lmss": torch.from_numpy(lms),
            "hair_mask": hair_mask,
            "ori_img_gt_path": ori_img_gt_path,
            'face_mask': face_mask, 
            "seg_mask": seg_mask,
            "head_eye_g_mask":head_eye_g_mask,
        }

        return data_dict