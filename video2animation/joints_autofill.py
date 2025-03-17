import numpy as np


mid_shoulder_id = 1
right_shoulder_id = 2
right_elbow_id = 3
right_hand_id = 4
left_shoulder_id = 5
left_elbow_id = 6
left_hand_id = 7
mid_hip_id = 8
right_hip_id = 9
left_hip_id = 12

base_mid_shoulder = np.array([639.127, 341.447, 0], dtype=np.float32)
base_right_shoulder = np.array([521.651, 345.319, 0], dtype=np.float32)
base_right_elbow = np.array([498.12698, 543.28, 0], dtype=np.float32)
base_right_hand = np.array([484.40497, 684.344, 0], dtype=np.float32)
base_left_shoulder = np.array([758.716, 341.394, 0], dtype=np.float32)
base_left_elbow = np.array([778.276, 541.187, 0], dtype=np.float32)
base_left_hand = np.array([792.045, 680.39703, 0], dtype=np.float32)
base_mid_hip = np.array([641.173, 697.91, 0], dtype=np.float32)
base_right_hip = np.array([568.682, 697.933, 0], dtype=np.float32)
base_left_hip = np.array([711.662, 697.968, 0], dtype=np.float32)

base_shoulder_length = np.linalg.norm(base_mid_shoulder - base_right_shoulder) + np.linalg.norm(base_mid_shoulder - base_left_shoulder)
scale_mid_hip_from_mid_shoulder = (base_mid_hip - base_mid_shoulder) / base_shoulder_length
scale_right_hip_from_mid_hip = (base_right_hip - base_mid_hip) / base_shoulder_length
scale_left_hip_from_mid_hip = (base_left_hip - base_mid_hip) / base_shoulder_length
scale_right_hand_from_right_elbow = (base_right_hand - base_right_elbow) / base_shoulder_length
scale_left_hand_from_left_elbow = (base_left_hand - base_left_elbow) / base_shoulder_length


def fill_joints(pose_data):
    """
    Auto fill missing hip joint base on shoulder
    Joint index in keypoints object
        1: mid shoulder
        2: right shoulder
        3: right elbow
        4: right hand
        5: left shoulder
        6: left elbow
        7: left hand
        8: mid hip
        9: right hip
        12: left hip
    """

    new_pose_data = []
    for keypoints in pose_data:

        right_shoulder_length = 0
        left_shoulder_length = 0
        if not np.array_equal(keypoints[0][right_shoulder_id][:2], [0,0]):
            right_shoulder_length = np.linalg.norm(keypoints[0][mid_shoulder_id][:2]-keypoints[0][right_shoulder_id][:2])
        if not np.array_equal(keypoints[0][left_shoulder_id][:2], [0,0]):
            left_shoulder_length = np.linalg.norm(keypoints[0][mid_shoulder_id][:2]-keypoints[0][left_shoulder_id][:2])
        if right_shoulder_length == 0:
            right_shoulder_length = left_shoulder_length
        if left_shoulder_length == 0:
            left_shoulder_length = right_shoulder_length
        shoulder_length = right_shoulder_length + left_shoulder_length

        keypoints[0][mid_hip_id] = keypoints[0][mid_shoulder_id] + shoulder_length * scale_mid_hip_from_mid_shoulder
        keypoints[0][right_hip_id] = keypoints[0][mid_hip_id] + shoulder_length * scale_right_hip_from_mid_hip
        keypoints[0][left_hip_id] = keypoints[0][mid_hip_id] + shoulder_length * scale_left_hip_from_mid_hip

        if np.array_equal(keypoints[0][right_hand_id][:2], [0,0]):
            keypoints[0][right_hand_id] = keypoints[0][right_elbow_id] + shoulder_length * scale_right_hand_from_right_elbow
        if np.array_equal(keypoints[0][left_hand_id][:2], [0,0]):
            keypoints[0][left_hand_id] = keypoints[0][left_elbow_id] + shoulder_length * scale_left_hand_from_left_elbow

        new_pose_data.append(keypoints)

    return new_pose_data
