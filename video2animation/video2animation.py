import os
import os.path as osp

import env

from video2animation.fit_video2character import fit_video2character
from video2animation import video_processing
from animating.character import load_smplx_3D_model, load_pose
from animating.scene import init_scene
from animating.render import render, init_camera
from video_player.play_videos import play_videos


def video2animation(video_path,
                    use_hands=True,
                    use_face=True,
                    gender='neutral',
                    frame_step=1,
                    render_video=True,
                    debug=False,
                    overwrite=False):

    character_pose_data, video_path, sl_frame_start, sl_frame_length = fit_video2character(video_path,
                                                                                           use_hands=use_hands,
                                                                                           use_face=use_face,
                                                                                           gender=gender,
                                                                                           frame_step=frame_step,
                                                                                           detect_sign_language=False,
                                                                                           debug=debug,
                                                                                           overwrite=overwrite)

    # init blender scene
    init_scene()
    armature = load_smplx_3D_model(gender=gender)
    init_camera(armature)

    # load pose
    pose_files = []
    character_pose_fn = sorted(os.listdir(character_pose_data))
    for pose_fn in character_pose_fn:
        if int(osp.splitext(pose_fn)[0]) % frame_step == 0:
            pose_files.append(osp.join(character_pose_data, pose_fn))
    for idx, pose_fn in enumerate(pose_files):
        load_pose(armature=armature,
                  pose_fn=pose_fn,
                  frame=idx*frame_step)

    if render_video:
        video_name = osp.splitext(osp.split(video_path)[1])[0]
        output_file = osp.join(env.OUTPUT_VIDEOS, video_name, video_name+f'_{gender}_{frame_step}.mp4')
        render(0, sl_frame_length-1, output_file=output_file)
        play_videos(output_file,
                    video_processing.trim(video_path, sl_frame_start, sl_frame_length),
                    output_file=osp.splitext(output_file)[0]+'.parallel'+osp.splitext(output_file)[1])
