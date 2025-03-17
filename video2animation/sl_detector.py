import os
import tensorflow
import numpy as np
from pose_format.utils.openpose import load_openpose_directory

import env


def load_model():
    global sl_detect_model
    try:
        sl_detect_model
    except:
        # load sign language detection model
        print('\nLoading sign language classification model --------------', flush=True)
        print(env.SIGN_LANG_DETECTION)

        print('load_model__________________truoc\n')
        sl_detect_model = tensorflow.keras.models.load_model("D:\\videos2signlangani\\model.h5", compile=False)
        sl_detect_model.summary()
        
        print('\n')
        print('load_model__________________\n')

    return sl_detect_model


def get_sl_range(openpose_save: str, fps: int, frame_width: int, frame_height: int):
    """
    Return 2 values frame_start and frame_length
        :frame_start: the first frame contains hand sign language gesture
        :frame_length: the number of frames contains sign
    """
    sl_detect_model = load_model()

    print("trc khi pose data")
    pose_data = load_openpose_directory(openpose_save, fps=fps, width=frame_width, height=frame_height)

    print('sau khi pose data')
    # normalize the pose
    pose_data.normalize(pose_data.header.normalization_info(
        p1=("pose_keypoints_2d", "RShoulder"),
        p2=("pose_keypoints_2d", "LShoulder")
    ))

    # transform pose data to fit sl_detect_model prediction input shape
    data = pose_data.body.data.copy()
    data = np.transpose(data, (1, 0, 2, 3))
    data = np.delete(data, np.s_[25:], 2)

    temp = data
    data = []
    for idx in range(len(temp[0])):
        if idx == 0:
            continue
        a = temp[0][idx]
        b = temp[0][idx-1]
        data.append([])
        for idx in range(25):
            if not np.any(a[idx]) or not np.any(b[idx]):
                data[-1].append(0)
            else:
                data[-1].append(np.linalg.norm(a[idx] -
                                b[idx]) * fps)
    data = [data]
    # predicting in forward and backward with LSTM model
    forward_predict = sl_detect_model.predict(data)
    backward_predict = sl_detect_model.predict(np.flip(data, 1))
    backward_predict = np.flip(backward_predict, 1)

    # get label {0, 1} from confident
    forward_predict = np.array([np.argmax(frame)
                                for frame in forward_predict[0]])
    backward_predict = np.array([np.argmax(frame)
                                for frame in backward_predict[0]])

    # merge forward and backward label using logical And(&) operation
    data = np.logical_and(forward_predict, backward_predict)

    # return the first and the last index which is True
    true_idx = np.where(data == True)[0]
    # in case all frame are False, aka no sign language in video
    if np.size(true_idx) == 0:
        true_idx = [0]

    # return where sign language frame start and the total number of sign language frame
    frame_start = true_idx[0]
    frame_end = true_idx[-1] + 1

    frame_start += 5
    frame_end -= 3

    frame_length = frame_end - frame_start + 1
    frame_start = 38
    frame_length = 45

    return frame_start, frame_length
