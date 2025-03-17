import bpy
from math import pi


def load_smplx_3D_model(gender='neutral', texture='smplx_texture_f_alb.png'):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.window_manager.smplx_tool.smplx_gender = gender
    bpy.context.window_manager.smplx_tool.smplx_handpose = "flat"
    bpy.ops.scene.smplx_add_gender()
    bpy.context.window_manager.smplx_tool.smplx_texture = texture
    bpy.ops.object.smplx_set_texture()
    armature = bpy.context.object
    if armature.type == 'MESH':
        armature = armature.parent
    return armature


def load_pose(armature, pose_fn, frame):
    bpy.ops.object.mode_set()
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = armature.children[0]

    bpy.context.scene.frame_set(frame)
    bpy.ops.object.smplx_update_joint_locations()
    bpy.ops.object.smplx_set_poseshapes()
    bpy.ops.object.smplx_load_pose(filepath=pose_fn)

    # Adjusting pelvis' rotation
    pelvis = armature.pose.bones['pelvis']
    pelvis.rotation_mode = 'XYZ'
    if pelvis.rotation_euler.x <= 0:
        pelvis.rotation_euler.x += pi
    else:
        pelvis.rotation_euler.x -= pi
    pelvis.rotation_euler.y *= -1
    pelvis.rotation_euler.z *= -1
    pelvis.rotation_mode = 'QUATERNION'

    ign_bone_names = []
    # # Middle body
    # ign_bone_names.extend(['spine1', 'spine2', 'spine3'])
    # Lower body
    ign_bone_names.extend(['right_hip', 'right_knee', 'right_ankle', 'right_foot', 'left_hip', 'left_knee', 'left_ankle', 'left_foot'])
    # Upper body
    ign_bone_names.extend(['neck', 'jaw', 'head', 'right_eye_smplhf', 'left_eye_smplhf'])
    ## Pelvis
    # ign_bone_names.extend(['pelvis'])

    for bone_name in ign_bone_names:
        armature.pose.bones[bone_name].rotation_quaternion = (1, 0, 0, 0)

    # Update meshes
    bpy.ops.object.smplx_set_poseshapes()
    bpy.ops.object.smplx_measurements_to_shape()

    add_keyframe(armature=armature)


def add_keyframe(armature):
    bpy.ops.object.mode_set()
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = armature

    bpy.ops.object.posemode_toggle()
    bpy.ops.pose.select_all(action='SELECT')
    for bone in armature.pose.bones:
        bone.rotation_mode = 'XYZ'
        bone.keyframe_insert('rotation_euler')
        # bone.keyframe_insert('rotation_quaternion')
    bpy.ops.object.posemode_toggle()