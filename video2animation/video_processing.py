import os
import os.path as osp
import ffmpeg

import env


def change_fps(src, target_fps: int):
    output_file = osp.join(env.TEMP_FOLDER, f'{target_fps}fps_videos', osp.split(src)[1])
    os.makedirs(osp.split(output_file)[0], exist_ok=True)

    # os.system(f'ffmpeg -i {src} -filter:v fps={target_fps} {output_file} -quiet')
    ffmpeg.input(src).filter('fps', fps=target_fps).output(output_file).run(quiet=True, overwrite_output=True)

    return output_file


def to30fps(src):
    return change_fps(src, 30)


def to_mp4(src):
    output_file = src+'.mp4'

    ffmpeg.input(src).output(output_file).run(quiet=True, overwrite_output=True)

    return output_file


def trim(src, frame_start, length):
    output_file = osp.splitext(src)[0] + f'({frame_start}-{frame_start+length-1})' + osp.splitext(src)[1]
    os.makedirs(osp.split(output_file)[0], exist_ok=True)

    ffmpeg.input(src).trim(start_frame=frame_start, end_frame=frame_start+length).setpts("PTS-STARTPTS").output(output_file).run(quiet=True, overwrite_output=True)

    return output_file
