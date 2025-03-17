import argparse
import os


import pickle

import os.path as osp
import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLX
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d

from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from text2animation.load_dictionary import load_dictionary


encoded_word = None

def encode_text(text: str):
    global encoded_word
    if encoded_word is None:
        encoded_word = load_dictionary()

    keys = []
    words = text.split(' ')
    for word in words:
        word = word.replace('_', ' ').lower()
        if word not in encoded_word:
            print(f'\"{word}\" not found')
            continue
        else:
            keys.append(encoded_word[word])
    return keys

def text2animation(sentence, out_video):
    cfg_file = 'C:\\workspace\\HybrIK\\configs\\smplx\\256x192_hrnet_rle_smplx_kid.yaml'
    CKPT = 'C:\\workspace\\HybrIK\\pretrained_models\\hybrikx_rle_hrnet.pth'
    cfg = update_config(cfg_file)
    cfg['MODEL']['EXTRA']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)
    cfg['LOSS']['ELEMENTS']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)


    hybrik_model = builder.build_sppe(cfg.MODEL)

    print(f'Loading model from {CKPT}...')
    save_dict = torch.load(CKPT, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict)


    hybrik_model.cuda(opt.gpu)

    hybrik_model.eval()


    write_stream = cv2.VideoWriter(
        out_video,
        cv2.VideoWriter_fourcc(*'mp4v'),  
        30,  
        (1280, 720)  
    )
    assert write_stream.isOpened(), 'Cannot open video for writing'

    smplx_faces = torch.from_numpy(hybrik_model.smplx_layer.faces.astype(np.int32))
    print('### Run Model...')
    idx = 0
    base_folder = "D:\\vsl\\data\\output-data\\3d-sign-lang-hybrik-full"

    keys = encode_text(text= sentence)
    for key in keys:
        folder_path = os.path.join(base_folder, key)

        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                if file_name.endswith('.pkl'):  
                    with torch.no_grad():
                        with open(file_path, 'rb') as file:
                            color_batch = pickle.load(file)
                            
                        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
                        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
                        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

                        
                        color = image_vis_batch[0]
                        valid_mask = valid_mask_batch[0].cpu().numpy()
                        alpha = 1.0  
                        image_vis = alpha * color[:, :, :3] * valid_mask + (1 - valid_mask) * np.zeros_like(color[:, :, :3])

                        
                        image_vis = image_vis.astype(np.uint8)
                        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
                        write_stream.write(image_vis)


    write_stream.release()



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='text2animation_hybrIK-X')


    parser.add_argument('--gpu',
                        help='gpu',
                        default=0,
                        type=int)
    # parser.add_argument('--video-name',
    #                     help='video name',
    #                     default='',
    #                     type=str)
    parser.add_argument('--sentence',
                        '-s',
                        type=str,
                        required=True,
                        help='text sentence')

    parser.add_argument('--out-video',
                        '-o',
                        default='',
                        type=str)
    opt = parser.parse_args()

    text2animation(opt.sentence, opt.out_video)


