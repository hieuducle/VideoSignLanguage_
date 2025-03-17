import os
import bpy


def init_scene():
    clear_scene()
    init_background()
    install_addon(overwrite=False)


def clear_scene():
    # Clear default objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def init_background():
    C = bpy.context
    scn = C.scene

    # Get the environment node tree of the current scene
    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes

    # Clear all nodes
    tree_nodes.clear()

    # Add Background node
    node_background = tree_nodes.new(type='ShaderNodeBackground')

    # Add Environment Texture node
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
    # Load and assign the image to the node property
    node_environment.image = bpy.data.images.load(os.path.abspath(r'res\blender\background\light.exr'))
    node_environment.location = -300,0

    # Add Output node
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
    node_output.location = 200,0

    # Link all nodes
    links = node_tree.links
    links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    # Using standard color
    scn.view_settings.view_transform = 'Standard'

    # Transparent background
    scn.render.film_transparent = True


def install_addon(overwrite=False):
    bpy.ops.preferences.addon_install(overwrite=overwrite, filepath=r'res\blender\addon\smplx_blender_addon_300_20220623.zip')
    bpy.ops.preferences.addon_enable(module='smplx_blender_addon')
