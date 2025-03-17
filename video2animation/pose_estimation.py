import os
import os.path as osp

import env


def estimating_pose(video_path: str,
                    use_hands=True,
                    use_face=False,
                    debug=False,
                    overwrite=False):
    
    video_path = osp.abspath(video_path)
    vid_name = osp.splitext(osp.split(video_path)[1])[0]
    model_folder = env.OPENPOSE_MODEL
    output_folder = osp.join(env.OPENPOSE_OUTPUT_KEYPOINT, vid_name)
    os.makedirs(output_folder, exist_ok=True)

    if osp.exists('./openpose.bin'):
        openpose = './openpose.bin'
    else:
        openpose = 'D:\\videos2signlangani\\bin\\OpenPoseDemo.exe'
    
    # Chỉ bật chế độ phát hiện tay, tắt cơ thể và khuôn mặt
    # cmd = f'{openpose} --video \"{video_path}\" --write_json \"{output_folder}\" --model_folder \"{model_folder}\" --display 0 --number_people_max 1 --body 0 --hand'
    cmd = f'{openpose} --video \"{video_path}\" --write_json \"{output_folder}\" --model_folder \"{model_folder}\" --display 0 --number_people_max 1 --hand'

    # if use_hands:
    #     cmd += ' --hand'
        # cmd += ' --hand_detector 1'  # Sử dụng hand_detector 1 để tự động phát hiện bàn tay

    if debug:
        openpose_images_folder = osp.join(env.OPENPOSE_OUTPUT_IMAGES, vid_name)
        os.makedirs(openpose_images_folder, exist_ok=True)
        cmd += f' --write_images \"{openpose_images_folder}\"'
    else:
        cmd += f' --render_pose 0'
    
    if not overwrite:
        saved_keypoints = os.listdir(output_folder)
        frame_start = 0
        while (vid_name+'_'+str(frame_start).zfill(12)+'_keypoints.json') in saved_keypoints:
            frame_start += 1
        cmd += f' --frame_first {frame_start}'

    print(cmd, flush=True)
    os.system(cmd)
    print('\n')
    print("da done pose estimate\n")
    
    return output_folder
