import bpy
import os
import gc

def init_camera(object):
    camera_data = bpy.data.cameras.new(name='Camera')
    camera = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.rotation_euler[0] = 1.57
    camera.location = (0, -2.3, 0.1)
    bpy.context.scene.camera = camera
    return camera


def set_timewindow(frame_start, frame_end):
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end


def render(frame_start, frame_end, output_file: str='output.mp4', width: int = 1280, height: int = 720, fps: int = 30):
    output_file = os.path.abspath(output_file)
    print(f"\n\nRendering to {output_file}")

    scene = bpy.context.scene

    # Set frame range & fps
    set_timewindow(frame_start, frame_end)
    scene.render.fps = int(fps)

    # Resolution
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    # Output format (.mp4)
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.filepath = output_file
    scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'

    # Render with EEVEEE engine (for speed)
    scene.render.engine = 'BLENDER_EEVEE'
    bpy.ops.render.render(animation=True)
    print(f"Video saved to {output_file}")
    

     # Call garbage collection to free memory
    gc.collect()

    # Quit Blender after rendering (optional)
    bpy.ops.wm.quit_blender()
