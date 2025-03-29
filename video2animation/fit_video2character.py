import os
import os.path as osp
import cv2
import time
import numpy as np

import env

from smplifyx.data_parser import read_keypoints
from video2animation.video_processing import to30fps
from video2animation.pose_estimation import estimating_pose
from video2animation.pose_estimation_alphapose import pose_estimate
from video2animation.sl_detector import get_sl_range
from video2animation.joints_autofill import fill_joints
from video2animation.fit_pose2character import fit_pose2character, ImgSize
from video2animation.pose_estimation_mediapipe import estimate_pose_mediapipe


def get_3d_output_folder(video_path: str, gender: str, sl_only: bool):
    if sl_only:
        output_folder = osp.join(env.OUTPUT_3D_SIGN_LANG, osp.splitext(osp.split(video_path)[1])[0], gender)
    else:
        output_folder = osp.join(env.OUTPUT_3D_POSE, osp.splitext(osp.split(video_path)[1])[0], gender)
    return output_folder


def fit_video2character(video_path: str,
                        use_hands=True,
                        use_face=True,
                        gender='neutral',
                        frame_step=1,
                        no_pose_estimation=False,
                        detect_sign_language=True,
                        debug=False,
                        overwrite=False):
    """
    Return:
        :character_pose_data_folder: the folder contain character pose data
        :30fps_video: the video used in pose estimation (with is necessary for future comparison)
        :sl_frame_start: index of the frame which starting hand signing from the 30fps video
        :sl_frame_length: the number of frames which contain hand sign
    """

    print(f'\nGenerating 3D pose for {gender} character from {video_path}')

    video_path = to30fps(video_path)
    cap = cv2.VideoCapture(video_path)

    if no_pose_estimation:
        vid_name = osp.splitext(osp.split(video_path)[1])[0]
        pose_folder = osp.join(env.OPENPOSE_OUTPUT_KEYPOINT, vid_name)
        print(f'\nWaiting for estimating pose at {video_path}', flush=True)
        while len(os.listdir(pose_folder)) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            time.sleep(60)
        print('Pose estimation finished\n')
    else:
        print("process pose estimation")
        # openpose
        # pose_folder = estimating_pose(video_path, use_hands=use_hands, use_face=use_face, debug=debug, overwrite=overwrite)

        # alphapose
        
        pose_folder = pose_estimate(video_path)

        # mediapipe
        # pose_folder = estimate_pose_mediapipe(video_path)

    

    print("\ndone_estimate\n")
    if detect_sign_language:
        sl_frame_range = get_sl_range(openpose_save=pose_folder,
                                    fps=cap.get(cv2.CAP_PROP_FPS),
                                    frame_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    frame_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        sl_frame_range = (0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    
    # Load motion data from disk
    
    pose_data = []
    keypoint_files = sorted(os.listdir(pose_folder))
    for keypoints_fn in keypoint_files:
        keypoints = read_keypoints(osp.join(pose_folder, keypoints_fn), use_hands=use_hands, use_face=use_face)
        keypoints = np.stack(keypoints.keypoints)[[0]]
        pose_data.append(keypoints)

    # Remove non-sign-lang frame data
    # pose_data = fill_joints(pose_data=pose_data)
    
    # print("Kích thước của mảng pose_data là:", len(pose_data))

    

    character_pose_data_folder = get_3d_output_folder(video_path=video_path,
                                                      gender=gender,
                                                      sl_only=detect_sign_language)

    os.makedirs(character_pose_data_folder, exist_ok=True)
    for idx, keypoints in enumerate(pose_data[sl_frame_range[0]:sum(sl_frame_range)]):
        if idx % frame_step != 0 and idx != sl_frame_range[1] - 1:
            continue

        print(f'\nWorking with {video_path} on frame {idx+sl_frame_range[0]}')
        character_pose_fn = osp.join(character_pose_data_folder, str(idx).zfill(12)+'.pkl')
        fit_pose2character(keypoints=keypoints,
                           img_size=ImgSize(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           result_fn=character_pose_fn,
                           gender=gender,
                           use_hands=use_hands,
                           use_face=use_face,
                           overwrite=overwrite)
        

    # print(sl_frame_range[0])
    # print("\n")
    # print(sum(sl_frame_range))
    # print("\n")
    
        
    

    return character_pose_data_folder, video_path, sl_frame_range[0], sl_frame_range[1]