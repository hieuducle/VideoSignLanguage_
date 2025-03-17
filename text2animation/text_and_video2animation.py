import os
import os.path as osp

import env

from text2animation.text2animation import text2animation, encode_text
from video2animation.fit_video2character import fit_video2character


def text_and_video2animation(text: str, gender='neutral', frame_step=5, frame_between_word=10, render_video=True):

    keys = encode_text(text=text)
    for key in keys:
        video_path = osp.join(env.INPUT_FOLDER, 'sl_videos', key+'.mp4')
        fit_video2character(video_path=video_path,
                            use_hands=True,
                            use_face=False,
                            gender=gender,
                            frame_step=frame_step,
                            debug=False,
                            overwrite=False)
        
    text2animation(text=text,
                   gender=gender,
                   frame_step=frame_step,
                   frame_between_word=frame_between_word,
                   render_video=render_video)
