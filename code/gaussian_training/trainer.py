import torch
from .folder_loader import Folder_Dataset
from .scene_adteeth import GaussianScene
from .emoca_recog import Emoca_Recog
from .config import pipeline_config, optim_config, model_config, diffu_config
from tqdm import tqdm
import cv2
from .gaussian_renderer import  render_fea
from ..render_utils.loss_utils import l1_loss, ssim, l2_loss
from ..utils_funcs.video_writer import VideoWriter
from ..finetune_module.finetune_model import InstructPix2Pix
import lpips
import time
import os
from torchvision.transforms import Resize
import torch.nn.functional as F
from glob import glob
import ipdb
import numpy as np

from .styletransfer import Style_transfer


class Trainer():
    def __init__(self,
                 source_dir,
                 save_dir,
                 device = 'cuda:0',
                 dtype = torch.float16,
                 prompt = None,
                 train_skip = 0, # decide percentage of the dataset keyframes  
                 ref_image_path = None,
                 exp_name = ""
                 ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.source_dir = source_dir
        self.folder_dataset = Folder_Dataset(self.source_dir, 0., .6, skip_length = train_skip)
        self.folder_dataset_fix = Folder_Dataset(self.source_dir, 0., 0.01, skip_length = 0)
        self.eval_dataset = Folder_Dataset(self.source_dir, .6, .98)
        # self.eval_dataset = Folder_Dataset(self.source_dir, 0., .6)

        self.save_dir = os.path.join(save_dir, 'gaussian_scene_fea_dev'+exp_name)
        self.log_dir = os.path.join(save_dir, 'gaussian_scene_fea_dev'+exp_name, 'logs')
        # self.use_predef_style = use_predef_style
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.img_size = self.folder_dataset.img_size
        self.gaussian_scene = GaussianScene(self.source_dir, self.img_size, device=self.device)
        self.gaussian_scene.training_setup(optim_config)
        self.resize = Resize((224,224))
        self.resize512 = Resize((512,512))

        self.emoca_recog = Emoca_Recog(self.device)
        
        self.prompt = prompt
        self.ref_image_path = ref_image_path

        self.image_guide = diffu_config.image_guide
        self.text_guide = diffu_config.text_guide
        self.diffuse_step = diffu_config.diffuse_step

        self.gaussian_prepared = False
        self.recon_dir = os.path.join(self.source_dir, 'gaussian_scene_fea_dev')
        
        # ipdb.set_trace()
        if len(glob(os.path.join(self.save_dir, '*recon.pt')))>2:
            model_path = sorted(glob(os.path.join(self.save_dir, '*recon.pt')))
            print(f">>>>>>>>> Load latest model in {model_path[-1]}")
            self.gaussian_scene.restore(torch.load(model_path[-1]))
            self.gaussian_prepared = True
        elif os.path.isfile(os.path.join(self.recon_dir, 'recon.pt')):
            print(os.path.isfile(os.path.join(self.recon_dir, 'recon.pt')))
            self.gaussian_scene.restore(torch.load(os.path.join(self.recon_dir, 'recon.pt')))
            self.gaussian_prepared = True
           
        
        bg_color = [1, 1, 1]
        self.background = torch.tensor(bg_color, dtype=self.dtype, device=self.device)
        self.percep_module = lpips.LPIPS(net='vgg').to(self.device).eval()
        
    def train_gaussian_fea(self, train_dataset = None):
        if train_dataset is None:
            train_dataset = self.folder_dataset
        self.gaussian_scene.training_setup(optim_config)
        epoch_num = min(int((optim_config.iterations-1)/len(train_dataset)) + 1, 50)

        train_batch_size = 10
        folder_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
        save_epoch_freq = 10

        for epoch in tqdm(range(epoch_num)):
            if epoch % save_epoch_freq == 0:
                self.eval_fea(epoch)


            for data_dict in tqdm(folder_loader, total=len(train_dataset)//train_batch_size, desc='training'):
                img_gt = data_dict['img_gt']
                frame_ids = data_dict['load_idx']
                
                img_gt = img_gt.to(self.device)
                batch_size = img_gt.shape[0]
                loss_l1, loss_ssim, loss_G, loss_mask = 0., 0., 0., 0.
                batch_xyz, batch_rot = self.gaussian_scene.get_xyz_rot_batch(frame_ids)
                for i in range(batch_size):
                    frame_id = int(frame_ids[i])
                    self.gaussian_scene.register_xyz_rotation(batch_xyz[i], batch_rot[i])
                    viewpoint_cam = self.gaussian_scene.cameras[frame_id]
                    render_pkg = render_fea(viewpoint_cam, self.gaussian_scene, pipeline_config, self.background)
                    render_fea_map, render_mask = render_pkg["render"], render_pkg['mask']
                    render_img = self.gaussian_scene.neural_renderer(render_fea_map.unsqueeze(0))[0]
                    gt_img = img_gt[i, :, :, :3].permute(2,0,1).float()/255.
                    gt_mask = img_gt[i, :, :, 3].float()/255.
                    l1_dis = l1_loss(render_img, gt_img)
                    l1_dis_fea = l1_loss(render_fea_map[0:3,:,:], gt_img)
                    l1_mask = l1_loss(render_mask.squeeze(0), gt_mask)
                    loss_l1 += (l1_dis*0.5+l1_dis_fea)
                    loss_mask += l1_mask
                    ssim_dis = 1.0 - ssim(render_img, gt_img)
                    loss_ssim += ssim_dis

                    percep_dis = torch.zeros_like(loss_l1)
                    if epoch > 10:
                        percep_dis = torch.mean(self.percep_module(render_img.unsqueeze(0)*2.-1., gt_img.unsqueeze(0)*2.-1.))
                    loss_G += 1e-2 * percep_dis
                    
                    
                loss_lap = self.gaussian_scene.cal_lap_loss_neural()
                loss_scale = self.gaussian_scene.cal_scale_loss()
                loss_dis = self.gaussian_scene.cal_dis_loss()
                loss_l1 /= batch_size
                loss_ssim /= batch_size
                loss_G /= batch_size
                loss_mask /= batch_size
                loss = (1.0 - optim_config.lambda_dssim) * loss_l1 + optim_config.lambda_dssim * loss_ssim + .5 * loss_mask + loss_lap * 1e2 + loss_dis*1e2  + loss_scale * 1e-0 

                loss += loss_G * 1e-1
                loss.backward()
                with torch.no_grad():
                    self.gaussian_scene.optimizer.step()
                    self.gaussian_scene.optimizer.zero_grad(set_to_none=True)
                print(loss_l1.item(),loss_ssim.item(),loss_mask.item(),loss_lap.item(),loss_dis.item(),loss_scale.item(),"\n")

            img_gt_view = img_gt[-1, :, :, :3]
            img_debug = torch.cat((torch.clamp(render_img.permute(1,2,0).detach()*255, 0, 255).byte(), img_gt_view), dim=1)[..., [2,1,0]]
            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + '.jpg'), img_debug.cpu().numpy())
            if (epoch+1) % save_epoch_freq == 0:
                torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, f'{epoch}_recon.pt'))
                torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, f'recon.pt'))
        
        torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, 'recon.pt'))

    def eval_fea(self, epoch, filename = None, use_fix_opacity=False, use_render_mask=True):
        
        eval_dataset = self.eval_dataset
        test_folder_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=10, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)
        if filename is None:
            vid_out = VideoWriter(os.path.join(self.log_dir, str(epoch).zfill(4) + '_eval.mp4'))
            vid_out_gt = VideoWriter(os.path.join(self.log_dir,'gt_eval.mp4'))
            vid_out_render = VideoWriter(os.path.join(self.log_dir, 'render_eval.mp4'))
        else:
            vid_out = VideoWriter(filename)
            vid_out_gt = None 
            vid_out_render = None
        eval_time = 0
        for data_dict in test_folder_loader:
            img_gt = data_dict["img_gt"]
            frame_ids = data_dict["load_idx"]
            img_gt = img_gt.to(self.device)
            batch_size = img_gt.shape[0]
            batch_xyz, batch_rot = self.gaussian_scene.get_xyz_rot_batch(data_dict["load_idx"])

            with torch.no_grad():
                for i in range(batch_size):
                    frame_id = int(frame_ids[i])

                    self.gaussian_scene.register_xyz_rotation(batch_xyz[i], batch_rot[i])
                    viewpoint_cam = self.gaussian_scene.cameras[frame_id]


                    render_pkg = render_fea(viewpoint_cam, self.gaussian_scene, pipeline_config, self.background,use_fix_opacity=use_fix_opacity)

                    render_fea_map, render_mask = render_pkg["render"], render_pkg['mask']

                    render_img = self.gaussian_scene.neural_renderer(render_fea_map.unsqueeze(0))[0]


                    if use_render_mask:
                        render_img = render_img*render_mask + (torch.tensor((1, 1, 1)).to(render_img.device).view(3, 1, 1))*(1.-render_mask)
                    else:
                        render_img = render_img
                    img_gt_view = img_gt[i, :, :, :3]
                    gt_mask = img_gt[i, :, :, 3].float()/255.
                    render_mask = render_mask.repeat(3, 1, 1)
                    vis_img = torch.cat((torch.clamp(render_img.permute(1,2,0).detach()*255, 0, 255).byte(), img_gt_view, torch.clamp(render_mask.permute(1,2,0).detach()*255, 0, 255).byte(), torch.clamp(render_fea_map[0:3,:,:].permute(1,2,0).detach()*255, 0, 255).byte()), dim=1)
                    vid_out.write_frame(vis_img.cpu().numpy())
                    if vid_out_gt != None:
                        vid_out_gt.write_frame(img_gt_view.cpu().numpy())
                    if vid_out_render != None:
                        vid_out_render.write_frame(torch.clamp(render_img.permute(1,2,0).detach()*255, 0, 255).byte().cpu().numpy())

        vid_out.close()
        if vid_out_gt != None:
            vid_out_gt.close()
        if vid_out_render != None:
            vid_out_render.close()


    def train_ip2p_fea(self):
        assert self.gaussian_prepared, 'Gaussian is not initialized. Please run train_gaussian at first!'
        self.gaussian_scene.training_setup(optim_config)
        
        # import ipdb
        # ipdb.set_trace()
        ### Load Pix2Pix Model
        ip2p_module = InstructPix2Pix(self.device, ip2p_use_full_precision=False)
        
        text_embedding = ip2p_module.pipe._encode_prompt(
                    self.prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
                ) 
        
        epoch_num = min(int((optim_config.iterations-1)/len(self.folder_dataset)) + 1, 301)
        train_batch_size = 1
        folder_loader = torch.utils.data.DataLoader(
            self.folder_dataset, batch_size=train_batch_size, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
        percep_epoch_start = 5
        save_epoch_freq = 1
        print_freq = 10
        
        for epoch in tqdm(range(epoch_num)):
            if epoch % save_epoch_freq == 0:
                self.eval_fea(epoch)
            current_step = 0
            
            for data_dict in tqdm(folder_loader, total=len(self.folder_dataset)//train_batch_size, desc='training'):
                
                img_gt = data_dict['img_gt']
                frame_ids = data_dict['load_idx']
                mouth_rects = data_dict['bbox_tensor']
                head_rects = data_dict['head_bbox_tensor']
                head_mask = data_dict['head_mask']
                idx = data_dict['idx']
                lmss = data_dict['lmss']
                hair_mask = data_dict['hair_mask']
                head_eye_g_mask = data_dict["head_eye_g_mask"]

                img_gt = img_gt.to(self.device)
                head_mask = head_mask.to(self.device).contiguous().float() / 255
                hair_mask = hair_mask.to(self.device).contiguous().float() / 255
                head_eye_g_mask = head_eye_g_mask.to(self.device).contiguous().float() / 255
                lmss = lmss.to(self.device)
                batch_size = img_gt.shape[0]
                loss_imgs, loss_Gs, loss_ssims, loss_masks, loss_exps = 0., 0., 0., 0., 0.
                use_edit = True if (current_step % diffu_config.ft_step) == 0 else False
                current_step = current_step + 1
                batch_xyz, batch_rot = self.gaussian_scene.get_xyz_rot_batch(frame_ids)
                
                # self.use_fix_opacity=False
                for i in range(batch_size):
                    frame_id = int(frame_ids[i])
                    mouth_rect = mouth_rects[i]
                    head_rect = head_rects[i]
                    self.gaussian_scene.register_xyz_rotation(batch_xyz[i], batch_rot[i])
                    viewpoint_cam = self.gaussian_scene.cameras[frame_id]
                    render_pkg = render_fea(viewpoint_cam, self.gaussian_scene, pipeline_config, self.background,use_fix_opacity=False)
                    render_fea_map, render_mask = render_pkg["render"], render_pkg['mask']
                    render_img = self.gaussian_scene.neural_renderer(render_fea_map.unsqueeze(0))[0]
                    gt_img = img_gt[i, :, :, :3].permute(2,0,1).float()/255.
                    gt_mask = img_gt[i, :, :, 3].float()/255.
                    head_eye_g_mask = head_eye_g_mask[i,:,:,0]

                    if use_edit:
                        edited_image = ip2p_module.edit_image(
                            text_embedding,
                            render_img.unsqueeze(0).detach().type(torch.float16),
                            gt_img.unsqueeze(0).detach().type(torch.float16),
                            guidance_scale=self.text_guide, 
                            image_guidance_scale=self.image_guide, 
                            diffusion_steps=self.diffuse_step,
                            lower_bound=0.50,
                            upper_bound=0.80,   
                            head_rect=head_rect.unsqueeze(0),
                            head_mask=head_eye_g_mask,
                        ).type(torch.float32)
                        # ipdb.set_trace()
                        up_to_dataset = torch.clip(edited_image[0].permute(1, 2, 0)*255, 0., 255.).detach().byte().cpu()
                        self.folder_dataset.image_list[idx] = up_to_dataset
                    else:
                        edited_image = self.folder_dataset.image_list[idx].to(self.device).permute(0, 3, 1, 2).contiguous().float()/255.0

                    loss_imgs += l1_loss(render_img, edited_image[0]) + 0.5*l1_loss(render_fea_map[0:3,:,:], gt_img)
                    loss_ssims += 1.0 - ssim(render_img, edited_image[0])

                    gt_exp = self.emoca_recog.recog(self.resize(gt_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)))
                    render_exp = self.emoca_recog.recog(self.resize(render_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)))
                    loss_exps += l2_loss(render_exp, gt_exp)

                    loss_Gs += torch.mean(self.percep_module(render_img.unsqueeze(0)*2.-1., edited_image*2.-1.))

                    if optim_config.use_mask:
                        l1_mask = l1_loss(render_mask.squeeze(0), gt_mask)
                        loss_masks += l1_mask

                        
                loss_lap = self.gaussian_scene.cal_lap_loss_neural()
                loss_scale = self.gaussian_scene.cal_scale_loss()
                loss_dis = self.gaussian_scene.cal_dis_loss()
                loss_sparse = self.gaussian_scene.opac_sparse_loss()
                
                loss_imgs /= batch_size

                loss_Gs /= batch_size
                loss_ssims /= batch_size

                loss_exps /= batch_size

                loss = 0. 
                loss += loss_lap * 1e2 + loss_dis*1e2 + loss_scale * 1e-0
                loss += (1.0 - optim_config.lambda_dssim) *6* loss_imgs + optim_config.lambda_dssim * loss_ssims
                loss += loss_exps * 5e-1
                loss += loss_sparse*4e-3
                if epoch > percep_epoch_start:
                    loss += 10*loss_Gs

                if optim_config.use_mask:
                    loss += 0.3 * loss_masks

                if current_step % print_freq == 0:
                    print(self.save_dir)
                    print('loss:',loss.item(),'loss_img:', loss_imgs.item(), ', loss_ssims:', loss_ssims.item(), ', loss_G:', loss_Gs.item(), ', loss_masks:', loss_masks.item(), ', loss_exps: ', loss_exps.item())
                
                loss.backward()
                with torch.no_grad():
                    self.gaussian_scene.optimizer.step()
                    self.gaussian_scene.optimizer.zero_grad(set_to_none=True)


            img_gt_view = img_gt[-1, :, :, :3]

            cat_temp = torch.cat((torch.clamp(render_img.permute(1,2,0).detach()*255, 0, 255).byte(), img_gt_view), dim=1)
            img_debug = torch.cat((torch.clamp(edited_image[0].permute(1,2,0).detach()*255, 0, 255).byte(), cat_temp, torch.clamp(render_fea_map[0:3,:,:].permute(1,2,0).detach()*255, 0, 255).byte()), dim=1)[..., [2,1,0]]
            
            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + '.jpg'), img_debug.cpu().numpy())

            mask_temp = torch.cat((torch.clamp(render_mask[0].detach()*255, 0, 255).byte(), torch.clamp(gt_mask.detach()*255, 0, 255).byte()), dim=1)
            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + 'mask.jpg'), mask_temp.cpu().numpy())

          
            if (epoch+1) % save_epoch_freq == 0:
                torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, str(epoch).zfill(4) + '_diff_recon.pt'))
        self.eval_fea(-1, os.path.join(self.save_dir, 'recon.mp4'))
        torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, 'diff_recon.pt'))

    def train_relight_fea(self,bg_setting="left"):
        assert self.gaussian_prepared, 'Gaussian is not initialized. Please run train_gaussian at first!'
        self.gaussian_scene.training_setup(optim_config)
        
        from ..ReLight.gradio_demo import process_relight
        from torchvision import transforms

        resize = transforms.Resize([512,512])   

        
        epoch_num = min(int((optim_config.iterations-1)/len(self.folder_dataset)) + 1, 301)
        train_batch_size = 1
        folder_loader = torch.utils.data.DataLoader(
            self.folder_dataset, batch_size=train_batch_size, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
        percep_epoch_start = 5
        save_epoch_freq = 1
        print_freq = 10
        
        for epoch in tqdm(range(epoch_num)):
            if epoch % save_epoch_freq == 0:
                self.eval_fea(epoch)
            current_step = 0
            
            for data_dict in tqdm(folder_loader, total=len(self.folder_dataset)//train_batch_size, desc='training'):
                
                img_gt = data_dict['img_gt']
                frame_ids = data_dict['load_idx']
                mouth_rects = data_dict['bbox_tensor']
                head_rects = data_dict['head_bbox_tensor']
                head_mask = data_dict['head_mask']
                idx = data_dict['idx']
                lmss = data_dict['lmss']
                hair_mask = data_dict['hair_mask']
                head_eye_g_mask = data_dict["head_eye_g_mask"]

                img_gt = img_gt.to(self.device)
                head_mask = head_mask.to(self.device).contiguous().float() / 255
                hair_mask = hair_mask.to(self.device).contiguous().float() / 255
                head_eye_g_mask = head_eye_g_mask.to(self.device).contiguous().float() / 255
                lmss = lmss.to(self.device)
                batch_size = img_gt.shape[0]
                loss_imgs, loss_Gs, loss_ssims, loss_masks, loss_exps = 0., 0., 0., 0., 0.
                use_edit = True if (current_step % diffu_config.ft_step) == 0 else False
                current_step = current_step + 1
                batch_xyz, batch_rot = self.gaussian_scene.get_xyz_rot_batch(frame_ids)
                
                # self.use_fix_opacity=False
                for i in range(batch_size):
                    frame_id = int(frame_ids[i])
                    mouth_rect = mouth_rects[i]
                    head_rect = head_rects[i]
                    self.gaussian_scene.register_xyz_rotation(batch_xyz[i], batch_rot[i])
                    viewpoint_cam = self.gaussian_scene.cameras[frame_id]
                    render_pkg = render_fea(viewpoint_cam, self.gaussian_scene, pipeline_config, self.background,use_fix_opacity=False)
                    render_fea_map, render_mask = render_pkg["render"], render_pkg['mask']
                    render_img = self.gaussian_scene.neural_renderer(render_fea_map.unsqueeze(0))[0]
                    gt_img = img_gt[i, :, :, :3].permute(2,0,1).float()/255.
                    gt_mask = img_gt[i, :, :, 3].float()/255.
                    head_eye_g_mask = head_eye_g_mask[i,:,:,0]

                    if use_edit:
                        ic_img = torch.clip(gt_img.permute(1, 2, 0)*255, 0., 255.).detach().byte().cpu().numpy()
                        input_fg, results = process_relight(ic_img, self.prompt, 512, 512, highres_scale=1.5, bg_source=bg_setting)
                        results = resize(results)
                        edited_image = results*0.5+0.5

                        edited_image = edited_image*gt_mask + gt_img.unsqueeze(0)*(1.-gt_mask)
                        
                        up_to_dataset = torch.clip(edited_image[0].permute(1, 2, 0)*255, 0., 255.).detach().byte().cpu()
                        self.folder_dataset.image_list[idx] = up_to_dataset
                        debug_img = up_to_dataset[...,[2,1,0]].numpy()
       
                    else:
                        edited_image = self.folder_dataset.image_list[idx].to(self.device).permute(0, 3, 1, 2).contiguous().float()/255.0

                    loss_imgs += l1_loss(render_img, edited_image[0]) + 0.5*l1_loss(render_fea_map[0:3,:,:], gt_img)
                    loss_ssims += 1.0 - ssim(render_img, edited_image[0])

                    gt_exp = self.emoca_recog.recog(self.resize(gt_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)))
                    render_exp = self.emoca_recog.recog(self.resize(render_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)))
                    loss_exps += l2_loss(render_exp, gt_exp)

                    loss_Gs += torch.mean(self.percep_module(render_img.unsqueeze(0)*2.-1., edited_image*2.-1.))

                    if optim_config.use_mask:
                        l1_mask = l1_loss(render_mask.squeeze(0), gt_mask)
                        loss_masks += l1_mask

                        
                loss_lap = self.gaussian_scene.cal_lap_loss_neural()
                loss_scale = self.gaussian_scene.cal_scale_loss()
                loss_dis = self.gaussian_scene.cal_dis_loss()
                loss_sparse = self.gaussian_scene.opac_sparse_loss()
                
                loss_imgs /= batch_size
                loss_Gs /= batch_size
                loss_ssims /= batch_size
                loss_exps /= batch_size

                loss = 0. 
                loss += loss_lap * 1e2 + loss_dis*1e2 + loss_scale * 1e-0
                loss += (1.0 - optim_config.lambda_dssim) *6* loss_imgs + optim_config.lambda_dssim * loss_ssims
                loss += loss_exps * 5e-1
                loss += loss_sparse*4e-3
                if epoch > percep_epoch_start:
                    loss += 10*loss_Gs

                if optim_config.use_mask:
                    loss += 0.3 * loss_masks

                if current_step % print_freq == 0:
                    print('loss:',loss.item(),'loss_img:', loss_imgs.item(), ', loss_ssims:', loss_ssims.item(), ', loss_G:', loss_Gs.item(), ', loss_masks:', loss_masks.item(), ', loss_exps: ', loss_exps.item())
                
                loss.backward()
                with torch.no_grad():
                    self.gaussian_scene.optimizer.step()
                    self.gaussian_scene.optimizer.zero_grad(set_to_none=True)


            img_gt_view = img_gt[-1, :, :, :3]

            cat_temp = torch.cat((torch.clamp(render_img.permute(1,2,0).detach()*255, 0, 255).byte(), img_gt_view), dim=1)
            img_debug = torch.cat((torch.clamp(edited_image[0].permute(1,2,0).detach()*255, 0, 255).byte(), cat_temp, torch.clamp(render_fea_map[0:3,:,:].permute(1,2,0).detach()*255, 0, 255).byte()), dim=1)[..., [2,1,0]]
            
            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + '.jpg'), img_debug.cpu().numpy())

            mask_temp = torch.cat((torch.clamp(render_mask[0].detach()*255, 0, 255).byte(), torch.clamp(gt_mask.detach()*255, 0, 255).byte()), dim=1)
            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + 'mask.jpg'), mask_temp.cpu().numpy())

          
            if (epoch+1) % save_epoch_freq == 0:
                torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, str(epoch).zfill(4) + '_diff_recon.pt'))
            # self.gaussian_scene.save_updated_canonical('debug_canoncial.obj')
        self.eval_fea(-1, os.path.join(self.save_dir, 'recon.mp4'))
        torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, 'diff_recon.pt'))

    def train_style_transfer_fea(self):
        assert self.gaussian_prepared, 'Gaussian is not initialized. Please run train_gaussian at first!'
        self.gaussian_scene.training_setup(optim_config)
        
        if self.ref_image_path is not None:
            self.styletransfer = Style_transfer(torch.device('cuda'), self.img_size[0], self.ref_image_path, 300)
        else:
            self.styletransfer = Style_transfer(self.device, self.img_size[0], 'ref_image/style/cyberpunk.jpg', 250)
        
        epoch_num = min(int((optim_config.iterations-1)/len(self.folder_dataset)) + 1, 301)
        train_batch_size = 1
        folder_loader = torch.utils.data.DataLoader(
            self.folder_dataset, batch_size=train_batch_size, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
        percep_epoch_start = 5
        save_epoch_freq = 1
        print_freq = 10
        
        for epoch in tqdm(range(epoch_num)):
            if epoch % save_epoch_freq == 0:
                self.eval_fea(epoch)
            current_step = 0
            
            for data_dict in tqdm(folder_loader, total=len(self.folder_dataset)//train_batch_size, desc='training'):
                
                img_gt = data_dict['img_gt']
                frame_ids = data_dict['load_idx']
                mouth_rects = data_dict['bbox_tensor']
                head_rects = data_dict['head_bbox_tensor']
                head_mask = data_dict['head_mask']
                idx = data_dict['idx']
                lmss = data_dict['lmss']
                hair_mask = data_dict['hair_mask']
                head_eye_g_mask = data_dict["head_eye_g_mask"]
                ori_img_gt_path = data_dict['ori_img_gt_path']


                img_gt = img_gt.to(self.device)
                head_mask = head_mask.to(self.device).contiguous().float() / 255
                hair_mask = hair_mask.to(self.device).contiguous().float() / 255
                head_eye_g_mask = head_eye_g_mask.to(self.device).contiguous().float() / 255
                lmss = lmss.to(self.device)
                batch_size = img_gt.shape[0]
                loss_imgs, loss_Gs, loss_ssims, loss_masks, loss_exps = 0., 0., 0., 0., 0.
                use_edit = True if (current_step % diffu_config.ft_step) == 0 else False
                current_step = current_step + 1
                batch_xyz, batch_rot = self.gaussian_scene.get_xyz_rot_batch(frame_ids)
                
                for i in range(batch_size):
                    frame_id = int(frame_ids[i])
                    mouth_rect = mouth_rects[i]
                    head_rect = head_rects[i]
                    self.gaussian_scene.register_xyz_rotation(batch_xyz[i], batch_rot[i])
                    viewpoint_cam = self.gaussian_scene.cameras[frame_id]
                    render_pkg = render_fea(viewpoint_cam, self.gaussian_scene, pipeline_config, self.background,use_fix_opacity=False)
                    render_fea_map, render_mask = render_pkg["render"], render_pkg['mask']
                    render_img = self.gaussian_scene.neural_renderer(render_fea_map.unsqueeze(0))[0]
                    gt_img = img_gt[i, :, :, :3].permute(2,0,1).float()/255.
                    gt_mask = img_gt[i, :, :, 3].float()/255.
                    head_eye_g_mask = head_eye_g_mask[i,:,:,0]

                    if use_edit:
                        edited_image = self.styletransfer.transfer(ori_img_gt_path[i])
                        edited_image = edited_image * render_mask + render_img*(1-render_mask)
                        up_to_dataset = torch.clip(edited_image[0].permute(1, 2, 0)*255, 0., 255.).detach().byte().cpu()
                        self.folder_dataset.image_list[idx] = up_to_dataset
                    else:
                        edited_image = self.folder_dataset.image_list[idx].to(self.device).permute(0, 3, 1, 2).contiguous().float()/255.0

                    loss_imgs += l1_loss(render_img, edited_image[0]) + 0.5*l1_loss(render_fea_map[0:3,:,:], gt_img)
                    loss_ssims += 1.0 - ssim(render_img, edited_image[0])

                    gt_exp = self.emoca_recog.recog(self.resize(gt_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)))
                    render_exp = self.emoca_recog.recog(self.resize(render_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)))
                    loss_exps += l2_loss(render_exp, gt_exp)

                    loss_Gs += torch.mean(self.percep_module(render_img.unsqueeze(0)*2.-1., edited_image*2.-1.))

                    if optim_config.use_mask:

                        l1_mask = l1_loss(render_mask.squeeze(0), gt_mask)
                        loss_masks += l1_mask


                        
                loss_lap = self.gaussian_scene.cal_lap_loss_neural()
                loss_scale = self.gaussian_scene.cal_scale_loss()
                loss_dis = self.gaussian_scene.cal_dis_loss()
                loss_sparse = self.gaussian_scene.opac_sparse_loss()
                
                loss_imgs /= batch_size
                loss_Gs /= batch_size
                loss_ssims /= batch_size
                loss_exps /= batch_size

                loss = 0. 
                loss += loss_lap * 1e2 + loss_dis*1e2 + loss_scale * 1e-0
                loss += (1.0 - optim_config.lambda_dssim) *6* loss_imgs + optim_config.lambda_dssim * loss_ssims
                loss += loss_exps * 5e-1
                loss += loss_sparse*4e-3
                if epoch > percep_epoch_start:
                    loss += 10*loss_Gs

                if optim_config.use_mask:
                    loss += 0.3 * loss_masks

                if current_step % print_freq == 0:
                    print(self.save_dir)
                    print('loss:',loss.item(),'loss_img:', loss_imgs.item(), ', loss_ssims:', loss_ssims.item(), ', loss_G:', loss_Gs.item(), ', loss_masks:', loss_masks.item(), ', loss_exps: ', loss_exps.item())
                
                loss.backward()
                with torch.no_grad():
                    self.gaussian_scene.optimizer.step()
                    self.gaussian_scene.optimizer.zero_grad(set_to_none=True)


            img_gt_view = img_gt[-1, :, :, :3]

            cat_temp = torch.cat((torch.clamp(render_img.permute(1,2,0).detach()*255, 0, 255).byte(), img_gt_view), dim=1)
            img_debug = torch.cat((torch.clamp(edited_image[0].permute(1,2,0).detach()*255, 0, 255).byte(), cat_temp, torch.clamp(render_fea_map[0:3,:,:].permute(1,2,0).detach()*255, 0, 255).byte()), dim=1)[..., [2,1,0]]
            
            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + '.jpg'), img_debug.cpu().numpy())

            mask_temp = torch.cat((torch.clamp(render_mask[0].detach()*255, 0, 255).byte(), torch.clamp(gt_mask.detach()*255, 0, 255).byte()), dim=1)
            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + 'mask.jpg'), mask_temp.cpu().numpy())

          
            if (epoch+1) % save_epoch_freq == 0:
                torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, str(epoch).zfill(4) + '_diff_recon.pt'))
            # self.gaussian_scene.save_updated_canonical('debug_canoncial.obj')
        self.eval_fea(-1, os.path.join(self.save_dir, 'recon.mp4'))
        torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, 'diff_recon.pt'))

    def train_edit_anydoor_fea(self):
        assert self.gaussian_prepared, 'Gaussian is not initialized. Please run train_gaussian at first!'
        self.gaussian_scene.training_setup(optim_config)
        
        from ..anydoor.anydoor_apply import inference_single_image
        assert self.ref_image_path is not None
        # prompt = "A beautiful girl,8k, 4k, ultrarealistic, realistic, natural skin, textured skin,realism, realistic texture of clothing and the detailed material texture"
        # negative_prompt = "(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)"
        # from diffusers import StableDiffusionXLImg2ImgPipeline
        # pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        # "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        # )
        # pipe = pipe.to("cuda")
        from torchvision.transforms import ToTensor
        transform = ToTensor()

        image = cv2.imread(self.ref_image_path).astype(np.uint8)
        ref_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        smoothed_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
        
        ref_mask = smoothed_mask[:,:] > 128
        ref_mask = ref_mask.astype(np.uint8)


        epoch_num_stage = 15
        train_batch_size = 1
        folder_loader = torch.utils.data.DataLoader(
            self.folder_dataset, batch_size=train_batch_size, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
        percep_epoch_start = 5
        save_epoch_freq = 1
        print_freq = 10
        
        for epoch in tqdm(range(epoch_num_stage+1)):
            current_step = 0
            
            if epoch < epoch_num_stage:
                for data_dict in tqdm(folder_loader, total=len(self.folder_dataset)//train_batch_size, desc='training'):
                    
                    img_gt = data_dict['img_gt']
                    frame_ids = data_dict['load_idx']
                    mouth_rects = data_dict['bbox_tensor']
                    head_rects = data_dict['head_bbox_tensor']
                    head_mask = data_dict['head_mask']
                    idx = data_dict['idx']
                    lmss = data_dict['lmss']
                    hair_mask = data_dict['hair_mask']
                    head_eye_g_mask = data_dict["head_eye_g_mask"]
                    ori_img_gt_path = data_dict['ori_img_gt_path']
                    seg_masks = data_dict["seg_mask"]


                    img_gt = img_gt.to(self.device)
                    head_mask = head_mask.to(self.device).contiguous().float() / 255
                    hair_mask = hair_mask.to(self.device).contiguous().float() / 255
                    head_eye_g_mask = head_eye_g_mask.to(self.device).contiguous().float() / 255
                    seg_masks = seg_masks.to(self.device).contiguous().float() / 255

                    lmss = lmss.to(self.device)
                    batch_size = img_gt.shape[0]
                    loss_imgs, loss_Gs, loss_ssims, loss_masks, loss_exps = 0., 0., 0., 0., 0.
                    use_edit = True if (current_step % diffu_config.ft_step) == 0 else False
                    current_step = current_step + 1
                    batch_xyz, batch_rot = self.gaussian_scene.get_xyz_rot_batch(frame_ids)
                    
                    for i in range(batch_size):
                        frame_id = int(frame_ids[i])
                        mouth_rect = mouth_rects[i]
                        head_rect = head_rects[i]
                        self.gaussian_scene.register_xyz_rotation(batch_xyz[i], batch_rot[i])
                        viewpoint_cam = self.gaussian_scene.cameras[frame_id]
                        render_pkg = render_fea(viewpoint_cam, self.gaussian_scene, pipeline_config, self.background,use_fix_opacity=False)
                        render_fea_map, render_mask = render_pkg["render"], render_pkg['mask']
                        render_img = self.gaussian_scene.neural_renderer(render_fea_map.unsqueeze(0))[0]
                        gt_img = img_gt[i, :, :, :3].permute(2,0,1).float()/255.
                        gt_mask = img_gt[i, :, :, 3].float()/255.
                        head_eye_g_mask = head_eye_g_mask[i,:,:,0]
                        seg_mask = seg_masks[i,:,:,0]
                        head_mask = head_mask[i,:,:,0]

                        if epoch == 0:


                            nonzero_head = torch.nonzero(head_mask)
                            nonzero_seg = torch.nonzero(seg_mask)
                            left = torch.min(nonzero_seg[:, 1])
                            right = torch.max(nonzero_seg[:, 1])
                            top=torch.max(nonzero_head[:,0])
                            bottom=torch.max(nonzero_seg[:,0])

                            cloth_mask=torch.zeros_like(seg_mask)
                            cloth_mask[top:bottom+1,left:right]=1
                            cloth_mask=cloth_mask.cpu().numpy().astype(np.uint8)

                            gen_image = inference_single_image(ref_image, ref_mask, img_gt[0,:,:,:3].cpu().numpy().copy(), cloth_mask)

                            edited_image=torch.from_numpy(gen_image).permute(2,0,1).float()/255
                            edited_image=edited_image.to(self.device)

                            edited_image = edited_image *(1-head_mask) + gt_img*(head_mask)
                            edited_image=edited_image.unsqueeze(0)

                            up_to_dataset = torch.clip(edited_image[0].permute(1, 2, 0)*255, 0., 255.).detach().byte().cpu()
                            self.folder_dataset.image_list[idx] = up_to_dataset

                            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + '.jpg'), up_to_dataset[:,:,[2,1,0]].cpu().numpy())
                        elif epoch == epoch_num_stage-1:
                            if idx == 0:

                                nonzero_head = torch.nonzero(head_mask)
                                nonzero_seg = torch.nonzero(seg_mask)
                                left = torch.min(nonzero_seg[:, 1])
                                right = torch.max(nonzero_seg[:, 1])
                                top=torch.max(nonzero_head[:,0])
                                bottom=torch.max(nonzero_seg[:,0])

                                cloth_mask=torch.zeros_like(seg_mask)
                                cloth_mask[top:bottom+1,left:right]=1
                                cloth_mask=cloth_mask.cpu().numpy().astype(np.uint8)
                                loss_ori=1000
                                for k in range(0,20):
                                    gen_image = inference_single_image(ref_image, ref_mask, img_gt[0,:,:,:3].cpu().numpy().copy(), cloth_mask)

                                    
                                    edited_image=torch.from_numpy(gen_image).permute(2,0,1).float()/255
                                    edited_image=edited_image.to(self.device)

                                    edited_image = edited_image *(1-head_mask) + gt_img*(head_mask)
                                    edited_image=edited_image.unsqueeze(0)
                                    loss_img= l1_loss(render_img, edited_image[0])
                                    loss=loss_img
                                    print("loss:",loss)
                                    print("loss_ori",loss_ori)
                                    if loss<loss_ori:
                                        edited_image=edited_image
                                        up_to_dataset = torch.clip(edited_image[0].permute(1, 2, 0)*255, 0., 255.).detach().byte().cpu()
                                        loss_ori=loss
                                self.folder_dataset.image_list[idx] = up_to_dataset
                                cv2.imwrite(os.path.join(self.log_dir, '0.jpg'), up_to_dataset[:,:,[2,1,0]].cpu().numpy())

                                    
                        else:
                            edited_image = self.folder_dataset.image_list[idx].to(self.device).permute(0, 3, 1, 2).contiguous().float()/255.0

                        loss_imgs += l1_loss(render_img, edited_image[0]) + 0.5*l1_loss(render_fea_map[0:3,:,:], gt_img)
                        loss_ssims += 1.0 - ssim(render_img, edited_image[0])

                        gt_exp = self.emoca_recog.recog(self.resize(gt_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)))
                        render_exp = self.emoca_recog.recog(self.resize(render_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)))
                        loss_exps += l2_loss(render_exp, gt_exp)

                        loss_Gs += torch.mean(self.percep_module(render_img.unsqueeze(0)*2.-1., edited_image*2.-1.))

                        if optim_config.use_mask:

                            l1_mask = l1_loss(render_mask.squeeze(0), gt_mask)
                            loss_masks += l1_mask


                            
                    loss_lap = self.gaussian_scene.cal_lap_loss_neural()
                    loss_scale = self.gaussian_scene.cal_scale_loss()
                    loss_dis = self.gaussian_scene.cal_dis_loss()
                    loss_sparse = self.gaussian_scene.opac_sparse_loss()
                    
                    loss_imgs /= batch_size
                    loss_Gs /= batch_size
                    loss_ssims /= batch_size
                    loss_exps /= batch_size

                    loss = 0. 
                    loss += loss_lap * 1e2 + loss_dis*1e2 + loss_scale * 1e-0
                    loss += (1.0 - optim_config.lambda_dssim) *6* loss_imgs + optim_config.lambda_dssim * loss_ssims
                    loss += loss_exps * 5e-1
                    loss += loss_sparse*4e-3
                    if epoch > percep_epoch_start:
                        loss += 10*loss_Gs
                    if optim_config.use_mask:
                        loss += 5 * loss_masks

                    if current_step % print_freq == 0:
                        print(self.save_dir)
                        print('loss:',loss.item(),'loss_img:', loss_imgs.item(), ', loss_ssims:', loss_ssims.item(), ', loss_G:', loss_Gs.item(), ', loss_masks:', loss_masks.item(), ', loss_exps: ', loss_exps.item())
                    
                    loss.backward()
                    with torch.no_grad():
                        self.gaussian_scene.optimizer.step()
                        self.gaussian_scene.optimizer.zero_grad(set_to_none=True)

            else:
                for data_dict in tqdm(folder_loader, total=len(self.folder_dataset)//train_batch_size, desc='training'):
                    
                    img_gt = data_dict['img_gt']
                    frame_ids = data_dict['load_idx']
                    mouth_rects = data_dict['bbox_tensor']
                    head_rects = data_dict['head_bbox_tensor']
                    head_masks = data_dict['head_mask']
                    idx = data_dict['idx']
                    lmss = data_dict['lmss']
                    hair_mask = data_dict['hair_mask']
                    head_eye_g_mask = data_dict["head_eye_g_mask"]
                    ori_img_gt_path = data_dict['ori_img_gt_path']
                    seg_masks = data_dict["seg_mask"]

                    if idx != 0:
                        continue
                    else:
                        for k in range(50):
                            img_gt = img_gt.to(self.device)
                            head_masks = head_masks.to(self.device).contiguous().float() / 255
                            hair_mask = hair_mask.to(self.device).contiguous().float() / 255
                            head_eye_g_mask = head_eye_g_mask.to(self.device).contiguous().float() / 255
                            seg_masks = seg_masks.to(self.device).contiguous().float() / 255

                            lmss = lmss.to(self.device)
                            batch_size = img_gt.shape[0]
                            loss_imgs, loss_Gs, loss_ssims, loss_masks, loss_exps = 0., 0., 0., 0., 0.
                            use_edit = True if (current_step % diffu_config.ft_step) == 0 else False
                            current_step = current_step + 1
                            batch_xyz, batch_rot = self.gaussian_scene.get_xyz_rot_batch(frame_ids)
                            
                            for i in range(batch_size):
                                frame_id = int(frame_ids[i])
                                mouth_rect = mouth_rects[i]
                                head_rect = head_rects[i]
                                self.gaussian_scene.register_xyz_rotation(batch_xyz[i], batch_rot[i])
                                viewpoint_cam = self.gaussian_scene.cameras[frame_id]
                                render_pkg = render_fea(viewpoint_cam, self.gaussian_scene, pipeline_config, self.background,use_fix_opacity=False)
                                render_fea_map, render_mask = render_pkg["render"], render_pkg['mask']
                                render_img = self.gaussian_scene.neural_renderer(render_fea_map.unsqueeze(0))[0]
                                gt_img = img_gt[i, :, :, :3].permute(2,0,1).float()/255.
                                gt_mask = img_gt[i, :, :, 3].float()/255.
                                seg_mask = seg_masks[i,:,:,0]
                                head_mask = head_masks[i,:,:,0]

                                edited_image = self.folder_dataset.image_list[idx].to(self.device).permute(0, 3, 1, 2).contiguous().float()/255.0

                                loss_imgs += l1_loss(render_img, edited_image[0]) + 0.5*l1_loss(render_fea_map[0:3,:,:], gt_img)
                                loss_ssims += 1.0 - ssim(render_img, edited_image[0])

                                gt_exp = self.emoca_recog.recog(self.resize(gt_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)))
                                render_exp = self.emoca_recog.recog(self.resize(render_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)))
                                loss_exps += l2_loss(render_exp, gt_exp)

                                loss_Gs += torch.mean(self.percep_module(render_img.unsqueeze(0)*2.-1., edited_image*2.-1.))

                                if optim_config.use_mask:
                                    # ipdb.set_trace()
                                    l1_mask = l1_loss(render_mask.squeeze(0), gt_mask)
                                    loss_masks += l1_mask
                                    
                            loss_lap = self.gaussian_scene.cal_lap_loss_neural()
                            loss_scale = self.gaussian_scene.cal_scale_loss()
                            loss_dis = self.gaussian_scene.cal_dis_loss()
                            loss_sparse = self.gaussian_scene.opac_sparse_loss()
                            
                            loss_imgs /= batch_size
                            loss_Gs /= batch_size
                            loss_ssims /= batch_size
                            loss_exps /= batch_size

                            loss = 0. 
                            loss += loss_lap * 1e2 + loss_dis*1e2 + loss_scale * 1e-0
                            loss += (1.0 - optim_config.lambda_dssim) *6* loss_imgs + optim_config.lambda_dssim * loss_ssims
                            loss += loss_exps * 5e-1
                            loss += loss_sparse*4e-3
                            if epoch > percep_epoch_start:
                                loss += 10*loss_Gs

                            if optim_config.use_mask:
                                loss += 5 * loss_masks

                            if current_step % print_freq == 0:
                                print(self.save_dir)
                                print('loss:',loss.item(),'loss_img:', loss_imgs.item(), ', loss_ssims:', loss_ssims.item(), ', loss_G:', loss_Gs.item(), ', loss_masks:', loss_masks.item(), ', loss_exps: ', loss_exps.item())
                            
                            loss.backward()
                            with torch.no_grad():
                                self.gaussian_scene.optimizer.step()
                                self.gaussian_scene.optimizer.zero_grad(set_to_none=True)
                        break


            if epoch % save_epoch_freq == 0:
                self.eval_fea(epoch, use_render_mask=False)
            current_step = 0

            img_gt_view = img_gt[-1, :, :, :3]

            # ipdb.set_trace()

            cat_temp = torch.cat((torch.clamp(render_img.permute(1,2,0).detach()*255, 0, 255).byte(), img_gt_view.to(self.device)), dim=1)
            img_debug = torch.cat((torch.clamp(edited_image[0].permute(1,2,0).detach()*255, 0, 255).byte(), cat_temp, torch.clamp(render_fea_map[0:3,:,:].permute(1,2,0).detach()*255, 0, 255).byte()), dim=1)[..., [2,1,0]]
            
            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + '.jpg'), img_debug.cpu().numpy())

            mask_temp = torch.cat((torch.clamp(render_mask[0].detach()*255, 0, 255).byte(), torch.clamp(gt_mask.detach()*255, 0, 255).byte()), dim=1)
            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + 'mask.jpg'), mask_temp.cpu().numpy())

          
            if (epoch+1) % save_epoch_freq == 0:
                torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, str(epoch).zfill(4) + '_diff_recon.pt'))
            # self.gaussian_scene.save_updated_canonical('debug_canoncial.obj')
        self.eval_fea(-1, os.path.join(self.save_dir, 'recon.mp4'))
        torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, 'diff_recon.pt'))
