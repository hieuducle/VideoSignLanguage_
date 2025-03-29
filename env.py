import os.path as osp


INPUT_FOLDER = 'data\\input-data'


OUTPUT_FOLDER = 'data\\output-data'
OUTPUT_3D_POSE = osp.join(OUTPUT_FOLDER, '3d-pose')
OUTPUT_3D_SIGN_LANG = osp.join(OUTPUT_FOLDER, '3d-sign-lang')
OUTPUT_VIDEOS = osp.join(OUTPUT_FOLDER, 'videos')


TEMP_FOLDER = 'data\\temp'
OPENPOSE_OUTPUT_KEYPOINT = osp.join(TEMP_FOLDER, 'openpose-keypoints')
ALPHAPOSE_OUTPUT_KEYPOINT = osp.join(TEMP_FOLDER, 'alphapose-keypoints')
OPENPOSE_OUTPUT_IMAGES = osp.join(TEMP_FOLDER, 'openpose-images')


MODELS_FOLDER = 'models'
OPENPOSE_MODEL = osp.join(MODELS_FOLDER, 'openpose')
# SIGN_LANG_DETECTION = osp.join(MODELS_FOLDER, 'sl_detection\\model.h5')

SIGN_LANG_DETECTION = 'C:\\videos2signlangani\\models\\sl_detection\\model.h5'

# print(SIGN_LANG_DETECTION)

SMPLX_MODEL = MODELS_FOLDER
SMPLX_PRE_CONFIG = 'data\\fit_smplx.yaml'
VPOSE_MODEL = osp.join(MODELS_FOLDER, 'V02_05')

QIPEDC_DATASET = 'data\\qipedc_dataset.json'
