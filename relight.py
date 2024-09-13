from PIL import Image
import cv2
from code.ReLight.gradio_demo import process_relight
import torch
import numpy as np
import os
import ipdb

@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


q_tuple_list =[
    ('neon light, city'),
    ('magic lit, sci-fi RGB glowing, studio lighting'),
    ('neon light, city'),
    ('evil, gothic, Yharnam')
]

for elem in q_tuple_list:
    text_des = elem
    text = text_des
    save_path = os.path.join('./bkps_right', text_des.replace(' ', '_'))
    os.makedirs(save_path, exist_ok = True)
    for i in range(2):
        # ipdb.set_trace()
        img = np.ones([512,512,3])*255
        text = 'only background, ' + text_des
        input_fg, results = process_relight(img.astype(np.uint8), text,512,512, seed=np.random.randint(10000,99999), bg_source='right')
        results = pytorch2numpy(results)
        cv2.imwrite(os.path.join(save_path,f"bk_{i}.png"), results[0][...,[2,1,0]])

    save_path = os.path.join('./bkps_left', text_des.replace(' ', '_'))
    os.makedirs(save_path, exist_ok = True)
    for i in range(2):
        img = np.ones([512,512,3])*255
        text = 'only background, ' + text_des
        input_fg, results = process_relight(img.astype(np.uint8), text,512,512, seed=np.random.randint(10000,99999), bg_source='left')
        results = pytorch2numpy(results)
        cv2.imwrite(os.path.join(save_path,f"bk_{i}.png"), results[0][...,[2,1,0]])
