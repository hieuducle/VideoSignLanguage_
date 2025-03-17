



import os
import configargparse
from text2animation.text2animation import text2animation
import bpy


def main(sentence: str, output: str):
    output = os.path.abspath(output)
    os.makedirs(os.path.split(output)[0], exist_ok=True)
    text2animation(text=sentence,
                   gender='neutral',
                   frame_step=5,
                   frame_between_word=10,
                   render_video=True,
                   output_file=output + '.mp4')
    # bpy.ops.wm.save_mainfile(filepath=output + '.blend')


args = {}
parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                                    description='3D sign language animation for text sentence',
                                    prog='vsl')
parser.add_argument('--sentence',
                    '-s',
                    type=str,
                    required=True,
                    help='text sentence')
parser.add_argument('--output',
                    '-o',
                    type=str,
                    required=False,
                    help='Output path + name (without extension)')
args = vars(parser.parse_args())
main(**args)