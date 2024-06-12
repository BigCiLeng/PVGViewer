import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
from tkinter import filedialog
import os
import sys
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
from util_envlight import EnvLight
import torch
import json

# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


g_camera = util.Camera(1080, 1920)
BACKEND_OGL=0
BACKEND_CUDA=1
g_renderer_list = [
    None, # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer: GaussianRenderBase = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.
g_auto_sort = False
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 7

def impl_glfw_init():
    window_name = "NeUVF editor"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)

def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False

def update_activated_renderer_state(gaus: util_gau.GaussianData, cam_time_stamp = 0, time_stamp = 0):
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_gaussian_data_from_pvg(gaus, cam_time_stamp, time_stamp)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)
    
def update_cam(cam_config, cam_idx, g_camera):
    rotation_matrix = np.array(cam_config[cam_idx]["rotation"])
    # rotation_matrix[:,1] *= -1
    g_camera.w = cam_config[cam_idx]["width"]
    g_camera.h = cam_config[cam_idx]["height"]
    roll, pitch, yaw = g_camera.RM2EA(rotation_matrix)
    g_camera.yaw = yaw
    g_camera.pitch = pitch
    g_camera.position = np.array(cam_config[cam_idx]["position"])
    g_camera.cx = cam_config[cam_idx]["cx"]
    g_camera.cy = cam_config[cam_idx]["cy"]
    g_camera.fx = cam_config[cam_idx]["fx"]
    g_camera.fy = cam_config[cam_idx]["fy"]
    g_camera.up = rotation_matrix[:,1]
    g_camera.target = g_camera.position + rotation_matrix[:,2] * 0.0000000000000001
    g_camera.fovy = g_camera.get_fov()
    g_camera.cam_time_stamp = cam_config[cam_idx]["timestamp"]
    g_camera.last_x = g_camera.w / 2
    g_camera.last_y = g_camera.h / 2
    return g_camera

def main(args):
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, \
        g_render_mode, g_render_mode_tables
        
    imgui.create_context()
    # if args.hidpi is not None:
    #     imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    from renderer_cuda_pvg import CUDARenderer
    g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    g_renderer = g_renderer_list[BACKEND_CUDA]

    # gaussian data
    # gaussians = util_gau.naive_gaussian()
    try:
        from util_pvg import GaussianModel
        gaussians = GaussianModel(args)
        env_map = EnvLight(resolution=1024).cuda()
        
        gs_checkpoint = os.path.join(args.file_path, "chkpnt.pth")
        env_checkpoint = os.path.join(args.file_path, "env_light_chkpnt.pth")
        with open(os.path.join(args.file_path, "cameras.json")) as f:
            cam_config = json.load(f)
        (model_params, first_iter) = torch.load(gs_checkpoint)
        gaussians.restore(model_params, args)
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
        update_camera_pose_lazy()
        g_renderer.update_env_map(env_map)
        g_renderer.update_gaussian_data_from_pvg(gaussians)
        g_renderer.sort_and_update(g_camera)
    except RuntimeError as e:
        pass
    
    cam_idx = 0
    g_camera = update_cam(cam_config, cam_idx, g_camera)
    
    update_activated_renderer_state(gaussians, cam_time_stamp=g_camera.cam_time_stamp)
    # settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()
        
        g_renderer.draw()

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_control_win:
            if imgui.begin("Control", True):
                # rendering backend
                # changed, g_renderer_idx = imgui.combo("backend", g_renderer_idx, ["ogl", "cuda"][:len(g_renderer_list)])
                # if changed:
                #     g_renderer = g_renderer_list[g_renderer_idx]
                #     update_activated_renderer_state(gaussians)

                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")

                changed, g_renderer.reduce_updates = imgui.checkbox(
                        "reduce updates", g_renderer.reduce_updates,
                    )

                # imgui.text(f"# of Gaus = {len(gaussians)}")
                # if imgui.button(label='open ply'):
                #     file_path = filedialog.askopenfilename(title="open ply",
                #         initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                #         filetypes=[('ply file', '.ply')]
                #         )
                #     if file_path:
                #         try:
                #             gaussians = util_gau.load_ply(file_path)
                #             env_map = EnvLight(resolution=1024).cuda()
                #             base_dir = os.path.dirname(file_path)
                #             env_checkpoint = os.path.join(base_dir, "env_light_chkpnt.pth")
                #             try:
                #                 (light_params, _) = torch.load(env_checkpoint)
                #                 env_map.restore(light_params)
                #             except RuntimeError as e:
                #                 pass
                #             g_renderer.update_env_map(env_map)
                #             g_renderer.update_gaussian_data(gaussians)
                #             g_renderer.sort_and_update(g_camera)
                #         except RuntimeError as e:
                #             pass
                
                # camera fov
                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                update_camera_intrin_lazy()
                changed, g_camera.cam_time_stamp = imgui.slider_float(
                    "cam_timestamp", g_camera.cam_time_stamp, -0.5, 0.5, "t = %.3f"
                )
                # changed1, time_stamp = imgui.slider_float(
                #     "timestamp", time_stamp, -0.5, 0.5, "t = %.3f"
                # )
                if changed:
                    update_activated_renderer_state(gaus=gaussians, cam_time_stamp=g_camera.cam_time_stamp)
                
                changed, cam_idx = imgui.slider_int(
                    "cam_idx", cam_idx, 0, len(cam_config), format="%d"
                )                
                if changed:
                    g_camera = update_cam(cam_config, cam_idx, g_camera)
                    update_activated_renderer_state(gaussians, cam_time_stamp=g_camera.cam_time_stamp)

                # scale modifier
                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.
                    changed = True
                    
                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)
                                           
                # sort button
                if imgui.button(label='sort Gaussians'):
                    g_renderer.sort_and_update(g_camera)
                imgui.same_line()
                changed, g_auto_sort = imgui.checkbox(
                        "auto sort", g_auto_sort,
                    )
                if g_auto_sort:
                    g_renderer.sort_and_update(g_camera)
                
                if imgui.button(label='save image'):
                    width, height = glfw.get_framebuffer_size(window)
                    nrChannels = 3;
                    stride = nrChannels * width;
                    stride += (4 - stride % 4) if stride % 4 else 0
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                    imageio.imwrite("save.png", img[::-1])
                    # save intermediate information
                    # np.savez(
                    #     "save.npz",
                    #     gau_xyz=gaussians.xyz,
                    #     gau_s=gaussians.scale,
                    #     gau_rot=gaussians.rot,
                    #     gau_c=gaussians.sh,
                    #     gau_a=gaussians.opacity,
                    #     viewmat=g_camera.get_view_matrix(),
                    #     projmat=g_camera.get_project_matrix(),
                    #     hfovxyfocal=g_camera.get_htanfovxy_focal()
                    # )
                imgui.end()

        if g_show_camera_win:
            if imgui.button(label='rot 180'):
                g_camera.flip_ground()

            changed, g_camera.target_dist = imgui.slider_float(
                    "t", g_camera.target_dist, 1., 8., "target dist = %.3f"
                )
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.rot_sensitivity = imgui.slider_float(
                    "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset r"):
                g_camera.rot_sensitivity = 0.02

            changed, g_camera.trans_sensitivity = imgui.slider_float(
                    "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                    "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset z"):
                g_camera.zoom_sensitivity = 0.01

            changed, g_camera.roll_sensitivity = imgui.slider_float(
                    "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset ro"):
                g_camera.roll_sensitivity = 0.03

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file \n  by click 'open ply' button")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
            imgui.end()
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

from omegaconf import OmegaConf
if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="NeUVF editor with optional HiDPI support.")
    # parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    parser.add_argument("--config", type=str, default="./configs/waymo_reconstruction.yaml")
    parser.add_argument("--base_config", type=str, default="./configs/base_config.yaml")
    args, _ = parser.parse_known_args()
    # print(args)
    
    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    print(args)
    main(args)
