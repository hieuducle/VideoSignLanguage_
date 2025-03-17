# import os
# import os.path as osp
# import mediapipe as mp
# import cv2
# import json
# import env



# def estimate_pose(video_path: str):

#     # Khởi tạo MediaPipe Hands và Pose.
#     mp_hands = mp.solutions.hands
#     mp_pose = mp.solutions.pose
#     mp_face_mesh = mp.solutions.face_mesh

#     hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)

#     # Mở video.
#     # video_path = video_path
#     video_path = osp.abspath(video_path)
#     vid_name = osp.splitext(osp.split(video_path)[1])[0]
#     output_folder = osp.join(env.OPENPOSE_OUTPUT_KEYPOINT, vid_name)
#     os.makedirs(output_folder, exist_ok=True)

#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"Error: Cannot open video file {video_path}")
#     else:
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

#         # Tạo thư mục để lưu các tệp JSON.
#         # output_folder = 'D:\\videos2signlangani\\out_mediapipe_full'
#         # os.makedirs(output_folder, exist_ok=True)

#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Chuyển đổi hình ảnh sang RGB.
#             image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Xử lý khung hình để xác định hai tay và cơ thể.
#             hands_results = hands.process(image_rgb)
#             pose_results = pose.process(image_rgb)
            
#             # Xử lý khung hình để xác định khuôn mặt.
#             face_results = face_mesh.process(image_rgb)

#             # Duyệt qua các landmarks của khuôn mặt.
#             face_keypoints = []
#             if face_results.multi_face_landmarks:
#                 for face_landmarks in face_results.multi_face_landmarks:
#                     keypoints = []
#                     for landmark in face_landmarks.landmark:
#                         x = landmark.x * frame_width
#                         y = landmark.y * frame_height
#                         z = landmark.z
#                         visibility = landmark.visibility
#                         if x == 0 and y == 0 and z == 0:  # Bỏ qua các điểm không xác định
#                             keypoints.extend([0, 0, 0])
#                         else:
#                             keypoints.extend([x, y, visibility])

#                     # Chỉ lấy 68 keypoints đầu tiên nếu có nhiều hơn.
#                     face_keypoints = keypoints[:68 * 3]

#             # Nếu không phát hiện được khuôn mặt, đặt giá trị mặc định
#             if not face_keypoints:
#                 face_keypoints = [0] * 68 * 3  # 68 điểm keypoint, mỗi điểm có (x, y, visibility)

#             # Duyệt qua các landmarks của hai tay.
#             left_keypoints = []
#             right_keypoints = []
#             if hands_results.multi_hand_landmarks:
#                 for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
#                     hand_label = handedness.classification[0].label
#                     keypoints = []
#                     for landmark in hand_landmarks.landmark:
#                         x = landmark.x * frame_width
#                         y = landmark.y * frame_height
#                         z = landmark.z
#                         visibility = getattr(landmark, 'visibility', 1)
#                         if x == 0 and y == 0 and z == 0:  # Bỏ qua các điểm không xác định
#                             continue
#                         keypoints.extend([x, y, visibility])

#                     if hand_label == 'Left':
#                         left_keypoints = keypoints
#                     elif hand_label == 'Right':
#                         right_keypoints = keypoints

#             # Nếu không phát hiện được tay trái hoặc tay phải, đặt giá trị mặc định
#             if not left_keypoints:
#                 left_keypoints = [0] * 21 * 3  # 21 điểm keypoint, mỗi điểm có (x, y, visibility)
#             if not right_keypoints:
#                 right_keypoints = [0] * 21 * 3  # 21 điểm keypoint, mỗi điểm có (x, y, visibility)

#             # Duyệt qua các landmarks của cơ thể và tính toán keypoint giữa left_shoulder và right_shoulder.
#             pose_keypoints = []
#             left_shoulder_x, left_shoulder_y = None, None
#             right_shoulder_x, right_shoulder_y = None, None

#             left_hip_x, left_hip_y = None, None
#             right_hip_x, right_hip_y = None, None

#             if pose_results.pose_landmarks:
#                 for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
#                     # Truy cập keypoint của vai trái (left shoulder)
#                     if i == mp_pose.PoseLandmark.LEFT_SHOULDER.value:
#                         left_shoulder_x = landmark.x * frame_width
#                         left_shoulder_y = landmark.y * frame_height

#                     # Truy cập keypoint của vai phải (right shoulder)
#                     if i == mp_pose.PoseLandmark.RIGHT_SHOULDER.value:
#                         right_shoulder_x = landmark.x * frame_width
#                         right_shoulder_y = landmark.y * frame_height

#                     # Truy cập keypoint của háng trái (left hip)
#                     if i == mp_pose.PoseLandmark.LEFT_HIP.value:
#                         left_hip_x = landmark.x * frame_width
#                         left_hip_y = landmark.y * frame_height

#                     # Truy cập keypoint của háng phải (right hip)
#                     if i == mp_pose.PoseLandmark.RIGHT_HIP.value:
#                         right_hip_x = landmark.x * frame_width
#                         right_hip_y = landmark.y * frame_height

#                 # Tính toán keypoint giữa left_shoulder và right_shoulder
#                 if left_shoulder_x is not None and right_shoulder_x is not None:
#                     mid_point_shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
#                     mid_point_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

#                 # Tính toán keypoint giữa left_hip và right_hip
#                 if left_hip_x is not None and right_hip_x is not None:
#                     mid_point_hip_x = (left_hip_x + right_hip_x) / 2
#                     mid_point_hip_y = (left_hip_y + right_hip_y) / 2

#                 # Khởi tạo các keypoint mặc định cho các điểm còn lại
#                 default_keypoints = [0] * 3  # [x, y, visibility] mặc định là [0, 0, 0]

#                 # Thêm các keypoint còn lại vào danh sách pose_keypoints theo thứ tự yêu cầu
#                 keypoint_mapping = {
#                     0: mp_pose.PoseLandmark.NOSE,
#                     1: None,
#                     2: mp_pose.PoseLandmark.RIGHT_SHOULDER,
#                     3: mp_pose.PoseLandmark.RIGHT_ELBOW,
#                     4: mp_pose.PoseLandmark.RIGHT_WRIST,
#                     5: mp_pose.PoseLandmark.LEFT_SHOULDER,
#                     6: mp_pose.PoseLandmark.LEFT_ELBOW,
#                     7: mp_pose.PoseLandmark.LEFT_WRIST,
#                     8: None,  # MidHip is already added separately
#                     9: mp_pose.PoseLandmark.RIGHT_HIP,
#                     10: mp_pose.PoseLandmark.RIGHT_KNEE,
#                     11: mp_pose.PoseLandmark.RIGHT_ANKLE,
#                     12: mp_pose.PoseLandmark.LEFT_HIP,
#                     13: mp_pose.PoseLandmark.LEFT_KNEE,
#                     14: mp_pose.PoseLandmark.LEFT_ANKLE,
#                     15: mp_pose.PoseLandmark.RIGHT_EYE,
#                     16: mp_pose.PoseLandmark.LEFT_EYE,
#                     17: mp_pose.PoseLandmark.RIGHT_EAR,
#                     18: mp_pose.PoseLandmark.LEFT_EAR,
#                     19: None,  # LBigToe (default)
#                     20: None,  # LSmallToe (default)
#                     21: None,  # LHeel (default)
#                     22: None,  # RBigToe (default)
#                     23: None,  # RSmallToe (default)
#                     24: None,  # RHeel (default)
#                 }

#                 for idx, landmark in keypoint_mapping.items():
#                     if landmark is None and idx != 1 and idx != 8:
#                         pose_keypoints.extend(default_keypoints)
#                     else:
#                         if idx == 1:  # Neck (mid_point_shoulder)
#                             pose_keypoints.extend([mid_point_shoulder_x, mid_point_shoulder_y, 1])
#                         elif idx == 8:  # MidHip (mid_point_hip)
#                             pose_keypoints.extend([mid_point_hip_x, mid_point_hip_y, 1])
#                         elif idx == 19 or idx == 20 or idx == 21 or idx == 22 or idx == 23 or idx == 24:
#                             pose_keypoints.extend(default_keypoints)
#                         else:
#                             pose_keypoints.extend([
#                                 pose_results.pose_landmarks.landmark[landmark.value].x * frame_width,
#                                 pose_results.pose_landmarks.landmark[landmark.value].y * frame_height,
#                                 pose_results.pose_landmarks.landmark[landmark.value].visibility
#                             ])

#             else:
#                 pose_keypoints = [0] * 33 * 3  # 33 điểm keypoint của cơ thể, mỗi điểm có (x, y, visibility)

            
#             json_data = {
#                 "version": 1.1,
#                 "people": [
#                     {
#                         "hand_left_keypoints_2d": left_keypoints,
#                         "hand_right_keypoints_2d": right_keypoints,
#                         "pose_keypoints_2d": pose_keypoints,
#                         "face_keypoints_2d": face_keypoints  
#                     }
#                 ]
#             }

#             # save json
#             new_name = f"{vid_name}_{str(frame_count).zfill(12)}_keypoints.json"
#             output_path = os.path.join(output_folder, new_name)
#             with open(output_path, 'w') as f:
#                 json.dump(json_data, f)

#             frame_count += 1

#         cap.release()
#         hands.close()
#         pose.close()
#         face_mesh.close()
#     return output_folder



import os
import os.path as osp
import mediapipe as mp
import cv2
import json
import env

def estimate_pose_mediapipe(video_path: str):
    # Initialize MediaPipe Hands, Pose, and FaceMesh.
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.85)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.85)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)

    # Open the video.
    video_path = osp.abspath(video_path)
    vid_name = osp.splitext(osp.split(video_path)[1])[0]
    output_folder = osp.join(env.OPENPOSE_OUTPUT_KEYPOINT, vid_name)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if all expected JSON files already exist.
    all_files_exist = True
    for i in range(frame_count):
        new_name = f"{vid_name}_{str(i).zfill(12)}_keypoints.json"
        if not osp.exists(osp.join(output_folder, new_name)):
            all_files_exist = False
            break

    if all_files_exist:
        print("All keypoints files already exist, skipping processing.")
        cap.release()
        return output_folder

    print("Processing video and saving keypoints files.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands, pose, and face.
        hands_results = hands.process(image_rgb)
        pose_results = pose.process(image_rgb)
        face_results = face_mesh.process(image_rgb)

        # Process face landmarks.
        face_keypoints = []
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                keypoints = []
                for landmark in face_landmarks.landmark:
                    x = landmark.x * frame_width
                    y = landmark.y * frame_height
                    z = landmark.z
                    # goc
                    # visibility = landmark.visibility
                    # sua
                    visibility = 1
                    if x == 0 and y == 0 and z == 0:
                        keypoints.extend([0, 0, 0])
                    else:
                        keypoints.extend([x, y, visibility])

                face_keypoints = keypoints[:68 * 3]

        if not face_keypoints:
            face_keypoints = [0] * 68 * 3

        # Process hand landmarks.
        left_keypoints = []
        right_keypoints = []
        if hands_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                hand_label = handedness.classification[0].label
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x * frame_width
                    y = landmark.y * frame_height
                    z = landmark.z
                    # goc
                    # visibility = getattr(landmark, 'visibility', 1)
                    # sua
                    visibility = 1
                    if x == 0 and y == 0 and z == 0:
                        continue
                    keypoints.extend([x, y, visibility])

                if hand_label == 'Left':
                    left_keypoints = keypoints
                elif hand_label == 'Right':
                    right_keypoints = keypoints

        if not left_keypoints:
            left_keypoints = [0] * 21 * 3
        if not right_keypoints:
            right_keypoints = [0] * 21 * 3

        # Process pose landmarks.
        pose_keypoints = []
        left_shoulder_x, left_shoulder_y = None, None
        right_shoulder_x, right_shoulder_y = None, None
        left_hip_x, left_hip_y = None, None
        right_hip_x, right_hip_y = None, None

        if pose_results.pose_landmarks:
            for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                if i == mp_pose.PoseLandmark.LEFT_SHOULDER.value:
                    left_shoulder_x = landmark.x * frame_width
                    left_shoulder_y = landmark.y * frame_height
                if i == mp_pose.PoseLandmark.RIGHT_SHOULDER.value:
                    right_shoulder_x = landmark.x * frame_width
                    right_shoulder_y = landmark.y * frame_height
                if i == mp_pose.PoseLandmark.LEFT_HIP.value:
                    left_hip_x = landmark.x * frame_width
                    left_hip_y = landmark.y * frame_height
                if i == mp_pose.PoseLandmark.RIGHT_HIP.value:
                    right_hip_x = landmark.x * frame_width
                    right_hip_y = landmark.y * frame_height

            if left_shoulder_x is not None and right_shoulder_x is not None:
                mid_point_shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
                mid_point_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

            if left_hip_x is not None and right_hip_x is not None:
                mid_point_hip_x = (left_hip_x + right_hip_x) / 2
                mid_point_hip_y = (left_hip_y + right_hip_y) / 2

            default_keypoints = [0] * 3

            keypoint_mapping = {
                0: mp_pose.PoseLandmark.NOSE,
                1: None,
                2: mp_pose.PoseLandmark.RIGHT_SHOULDER,
                3: mp_pose.PoseLandmark.RIGHT_ELBOW,
                4: mp_pose.PoseLandmark.RIGHT_WRIST,
                5: mp_pose.PoseLandmark.LEFT_SHOULDER,
                6: mp_pose.PoseLandmark.LEFT_ELBOW,
                7: mp_pose.PoseLandmark.LEFT_WRIST,
                8: None,
                9: mp_pose.PoseLandmark.RIGHT_HIP,
                10: mp_pose.PoseLandmark.RIGHT_KNEE,
                11: mp_pose.PoseLandmark.RIGHT_ANKLE,
                12: mp_pose.PoseLandmark.LEFT_HIP,
                13: mp_pose.PoseLandmark.LEFT_KNEE,
                14: mp_pose.PoseLandmark.LEFT_ANKLE,
                15: mp_pose.PoseLandmark.RIGHT_EYE,
                16: mp_pose.PoseLandmark.LEFT_EYE,
                17: mp_pose.PoseLandmark.RIGHT_EAR,
                18: mp_pose.PoseLandmark.LEFT_EAR,
                19: None,
                20: None,
                21: None,
                22: None,
                23: None,
                24: None,
            }

            for idx, landmark in keypoint_mapping.items():
                if landmark is None and idx != 1 and idx != 8:
                    pose_keypoints.extend(default_keypoints)
                else:
                    if idx == 1:
                        pose_keypoints.extend([mid_point_shoulder_x, mid_point_shoulder_y, 1])
                    elif idx == 8:
                        pose_keypoints.extend([mid_point_hip_x, mid_point_hip_y, 1])
                    elif idx in [19, 20, 21, 22, 23, 24]:
                        pose_keypoints.extend(default_keypoints)
                    else:
                        pose_keypoints.extend([
                            pose_results.pose_landmarks.landmark[landmark.value].x * frame_width,
                            pose_results.pose_landmarks.landmark[landmark.value].y * frame_height,
                            # pose_results.pose_landmarks.landmark[landmark.value].visibility
                            1
                        ])
        else:
            pose_keypoints = [0] * 33 * 3

        json_data = {
            "version": 1.1,
            "people": [
                {
                    "hand_left_keypoints_2d": left_keypoints,
                    "hand_right_keypoints_2d": right_keypoints,
                    "pose_keypoints_2d": pose_keypoints,
                    "face_keypoints_2d": face_keypoints
                }
            ] 
        }

        # Save JSON with the desired naming convention
        new_name = f"{vid_name}_{str(frame_count).zfill(12)}_keypoints.json"
        output_path = os.path.join(output_folder, new_name)
        with open(output_path, 'w') as f:
            json.dump(json_data, f)

        frame_count += 1

    cap.release()
    hands.close()
    pose.close()
    face_mesh.close()
    return output_folder

