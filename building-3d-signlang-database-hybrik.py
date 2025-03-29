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


import mediapipe as mp


det_transform = T.Compose([T.ToTensor()])

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def changfps30(videos_folder):

    videos_folder30 = "D:\\vsl\\video30fps_HybrIK"
    target_fps = 30
    os.makedirs(videos_folder30, exist_ok=True)
    for file_name in os.listdir(videos_folder):
        input_video_path = os.path.join(videos_folder, file_name)
        if not file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        output_video_path = os.path.join(videos_folder30, file_name)
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Không thể mở video: {input_video_path}")
            continue
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec để ghi video
        out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (frame_width, frame_height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
    return videos_folder30

def build_signlang_3d(videos_folder, gpu, out_dir):
    

    videos_list = sorted(os.listdir(videos_folder))
    
    for video in videos_list:
        try:
            video_path = osp.join(videos_folder, video)
            print("video_path: ", video_path)
            video_basename = os.path.basename(video_path).split('.')[0]
            dir_path = f'D:\\vsl\\data\\output-data\\3d-sign-lang-hybrik-full\\{video_basename}'
            if os.path.exists(dir_path):
                print(f"Processing for video '{video_basename}' is already completed. Skipping...")
                continue  
            cfg_file = 'C:\\workspace\\HybrIK\\configs\\smplx\\256x192_hrnet_rle_smplx_kid.yaml'
            CKPT = 'C:\\workspace\\HybrIK\\pretrained_models\\hybrikx_rle_hrnet.pth'

            cfg = update_config(cfg_file)
            cfg['MODEL']['EXTRA']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)
            cfg['LOSS']['ELEMENTS']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)

            bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
            bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
            dummpy_set = edict({
                'joint_pairs_17': None,
                'joint_pairs_24': None,
                'joint_pairs_29': None,
                'bbox_3d_shape': bbox_3d_shape
            })

            transformation = SimpleTransform3DSMPLX(
                dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=bbox_3d_shape,
                rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False, add_dpg=False,
                loss_type=cfg.LOSS['TYPE'])

            det_model = fasterrcnn_resnet50_fpn(pretrained=True)
            hybrik_model = builder.build_sppe(cfg.MODEL)

            print(f'Loading model from {CKPT}...')
            save_dict = torch.load(CKPT, map_location='cpu')
            if type(save_dict) == dict:
                model_dict = save_dict['model']
                hybrik_model.load_state_dict(model_dict)
            else:
                hybrik_model.load_state_dict(save_dict)

            det_model.cuda(gpu)
            hybrik_model.cuda(gpu)
            det_model.eval()
            hybrik_model.eval()

            print('### Extract Image...')
            

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if not os.path.exists(os.path.join(out_dir, f'{video_basename}')):
                os.makedirs(os.path.join(out_dir, f'{video_basename}'))
            """
            xoa frame khong chua ki hieu

            """
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
            mp_drawing = mp.solutions.drawing_utils

            cap = cv2.VideoCapture(video_path)

            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)
                if result.multi_hand_landmarks:
                    frame_filename = os.path.join(f"{out_dir}/{video_basename}", f"frame_{frame_count:04d}.png")
                    cv2.imwrite(frame_filename, frame)
                    frame_count += 1
            cap.release()
            hands.close()
            """
            ---------------------------------------------------------------
            """
            # files = os.listdir(f'{out_dir}/{video_basename}')
            files = os.listdir("D:\\vsl\\data\\output-data\\W00246")
            files.sort()
            # img_path_list = [os.path.join(out_dir, f'{video_basename}', file) for file in files if file.endswith(('.jpg', '.png'))]
            img_path_list = [os.path.join("D:\\vsl\\data\\output-data\\W00246", file) for file in files if file.endswith(('.jpg', '.png'))]
            
            prev_box = None
            renderer = None
            smplx_faces = torch.from_numpy(hybrik_model.smplx_layer.faces.astype(np.int32))
            print('### Run Model...')
            idx = 0
            for img_path in tqdm(img_path_list):
            # for img_path in tqdm("D:\\vsl\\data\\output-data\\W00246"):

                dirname = os.path.dirname(img_path)
                basename = os.path.basename(img_path)

                with torch.no_grad():
                    # Run Detection
                    input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    det_input = det_transform(input_image).to(gpu)
                    det_output = det_model([det_input])[0]

                    if prev_box is None:
                        tight_bbox = get_one_box(det_output)
                        if tight_bbox is None:
                            continue
                    else:
                        tight_bbox = get_one_box(det_output)

                    if tight_bbox is None:
                        tight_bbox = prev_box

                    prev_box = tight_bbox

                    # Run HybrIK
                    pose_input, bbox, img_center = transformation.test_transform(
                        input_image.copy(), tight_bbox)
                    pose_input = pose_input.to(gpu)[None, :, :, :]

                    pose_output = hybrik_model(
                        pose_input, flip_test=True,
                        bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
                        img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float(),
                    )

                    uv_jts = pose_output.pred_uvd_jts.reshape(-1, 3)[:, :2]
                    # uv_jts = pose_output.pred_uvd_jts
                    
                    transl = pose_output.transl.detach()

                    # Visualization
                    image = input_image.copy()
                    focal = 1000.0
                    bbox_xywh = xyxy2xywh(bbox)
                    focal = focal / 256 * bbox_xywh[2]

                    vertices = pose_output.pred_vertices.detach()
                    verts_batch = vertices
                    transl_batch = transl

                    color_batch = render_mesh(
                        vertices=verts_batch, faces=smplx_faces,
                        translation=transl_batch,
                        focal_length=focal, height=image.shape[0], width=image.shape[1])

                    # dir_path = f'D:\\vsl\\data\\output-data\\3d-sign-lang-hybrik-full\\{video_basename}'
                    os.makedirs(dir_path, exist_ok=True)
                    character_pose_fn = osp.join(dir_path, str(idx).zfill(12)+'.pkl')
                    with open(character_pose_fn, 'wb') as file:
                        pickle.dump(color_batch, file)
                    print('turn {idx}')
                    idx += 1
        except KeyboardInterrupt:
            return
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HybrIK')

    parser.add_argument('--gpu',
                        help='gpu',
                        default=0,
                        type=int)
    parser.add_argument('--videos-folder',
                        help='video folder',
                        default='',
                        type=str)
                        
    parser.add_argument('--out-dir',
                        help='output folder',
                        default='D:\\vsl\\data\\output-data\\raw_images',
                        type=str)

    opt = parser.parse_args()
    
    opt.videos_folder = changfps30(opt.videos_folder)
    print("30fpsmain_successed")
    build_signlang_3d(opt.videos_folder, opt.gpu, opt.out_dir)

    
