import os
import os.path as osp
import configargparse
import json

from video2animation.fit_video2character import fit_video2character

import env


def main(videos_folder: str, gender: str, frame_step: int, no_pose_estimation, div: int, mod: int, debug: bool, overwrite: bool):
    
    videos_list = sorted(os.listdir(videos_folder))
    errors = {'errors': []}
    for idx, video in enumerate(videos_list):
        if idx % div == mod:
            try:
                video_path = osp.join(videos_folder, video)
                fit_video2character(video_path=video_path,
                                    use_hands=True,
                                    use_face=True,
                                    gender=gender,
                                    frame_step=frame_step,
                                    no_pose_estimation=no_pose_estimation,
                                    debug=debug,
                                    overwrite=overwrite)
            except KeyboardInterrupt:
                return
            except:
                print(f'\nSomething wrong with {video}\n')
                errors['errors'].append(video)
    
    

    with open(osp.join(env.OUTPUT_FOLDER, 'error-videos.json'), 'w') as file:
        json.dump(errors, file, indent=4)
    


if __name__ == '__main__':
    args = {}
    parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                                      description='Build 3D sign language database from videos',
                                      prog='vsl')
    parser.add_argument('--videos-folder',
                        type=str,
                        required=True,
                        help='Folder contains sign language videos')
    parser.add_argument('--frame-step',
                        type=int,
                        default=1,
                        required=False,
                        help='Frame step')
    parser.add_argument('--gender',
                        type=str,
                        default='neutral',
                        required=False,
                        help='Gender: male, female or neutral')
    parser.add_argument('--no-pose-estimation',
                        type=bool,
                        default=False,
                        required=False,
                        help='Gender: male, female or neutral')
    parser.add_argument('--div',
                        type=int,
                        default=1,
                        required=False,
                        help='File id\'s division')
    parser.add_argument('--mod',
                        type=int,
                        default=0,
                        required=False,
                        help='File id\'s modulo')
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        required=False)
    parser.add_argument('--overwrite',
                        type=bool,
                        default=False,
                        required=False)
    args = vars(parser.parse_args())

    main(**args)