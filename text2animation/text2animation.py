import os
import os.path as osp

import env

from text2animation.load_dictionary import load_dictionary
from video2animation.fit_video2character import get_3d_output_folder
from animating.character import load_smplx_3D_model, load_pose
from animating.scene import init_scene
from animating.render import render, init_camera


encoded_word = None


def _set_rest_pose(armature: any, frame: int):
    rest_pose_fn = r'data\rest_pose.pkl'
    load_pose(armature=armature,
              pose_fn=rest_pose_fn,
              frame=frame)


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


def text2animation(text: str, gender='neutral', frame_step=5, frame_between_word=10, render_video=True, output_file=None):

    # init blender scene
    # print("khoi dong\n")
    init_scene()
    armature = load_smplx_3D_model(gender=gender)

    # print("load xong")
    init_camera(armature)

    frame_counter = 0
    _set_rest_pose(armature=armature, frame=frame_counter)
    frame_counter += frame_between_word * 2

    keys = encode_text(text=text)
    print("keys la: ", keys)
    for key in keys:
        character_pose_data = get_3d_output_folder(video_path=key,
                                                   gender=gender,
                                                   sl_only=True)
        character_pose_fn = sorted(os.listdir(character_pose_data))
        sl_frame_length = int(osp.splitext(character_pose_fn[-1])[0]) + 1

        # load pose
        pose_files = []
        for pose_fn in character_pose_fn:
            idx = int(osp.splitext(pose_fn)[0])
            if (idx % frame_step == 0 and idx < sl_frame_length - frame_step) or (idx == sl_frame_length - 1):
                pose_files.append((idx, osp.join(character_pose_data, pose_fn)))
        for idx, pose in pose_files:
            load_pose(armature=armature,
                      pose_fn=pose,
                      frame=frame_counter+idx)
        frame_counter += sl_frame_length + frame_between_word

    frame_counter += frame_between_word
    _set_rest_pose(armature=armature, frame=frame_counter)

    if render_video:
        if not output_file:
            video_name = text + '.mp4'
            output_file = osp.join(env.OUTPUT_VIDEOS, 'sentences', video_name)
        render(0, frame_counter, output_file=output_file)
