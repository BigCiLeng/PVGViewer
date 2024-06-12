'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''
from OpenGL import GL as gl
import OpenGL.GL.shaders as shaders
import util
import util_gau
import numpy as np
import torch
from renderer_ogl import GaussianRenderBase
from dataclasses import dataclass
from cuda import cudart as cu
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import kornia
import torch.nn.functional as F
try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except:
    wglSwapIntervalEXT = None
from util_pvg import GaussianModel

VERTEX_SHADER_SOURCE = """
#version 450

smooth out vec4 fragColor;
smooth out vec2 texcoords;

vec4 positions[3] = vec4[3](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(3.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -3.0, 0.0, 1.0)
);

vec2 texpos[3] = vec2[3](
    vec2(0, 0),
    vec2(2, 0),
    vec2(0, 2)
);

void main() {
    gl_Position = positions[gl_VertexID];
    texcoords = texpos[gl_VertexID];
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330

smooth in vec2 texcoords;

out vec4 outputColour;

uniform sampler2D texSampler;

void main()
{
    outputColour = texture(texSampler, texcoords);
}
"""

@dataclass
class GaussianDataCUDA:
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor
    features: torch.Tensor | None
    mask: torch.Tensor | None
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-2]
    

@dataclass
class GaussianRasterizationSettingsStorage:
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool


def gaus_cuda_from_cpu(gau: util_gau) -> GaussianDataCUDA:
    gaus =  GaussianDataCUDA(
        xyz = torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot = torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale = torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity = torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh = torch.tensor(gau.sh).float().cuda().requires_grad_(False)
    )
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus
    

class CUDARenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        self.raster_settings = {
            "image_height": int(h),
            "image_width": int(w),
            "tanfovx": np.tan(-0.5),
            "tanfovy": np.tan(-0.5),
            "bg": torch.Tensor([0., 0., 0]).float().cuda(),
            "scale_modifier": 1.,
            "viewmatrix": None,
            "projmatrix": None,
            "sh_degree": 3,  # ?
            "campos": None,
            "prefiltered": False,
            "debug": False
        }
        gl.glViewport(0, 0, w, h)
        self.program = util.compile_shaders(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
        # setup cuda
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise RuntimeError(
                "OpenGL context may be running on integrated graphics"
            )
        
        self.vao = gl.glGenVertexArrays(1)
        self.tex = None
        self.set_gl_texture(h, w)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.need_rerender = True
        self.update_vsync()

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        else:
            print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.need_rerender = True
        self.gaussians = gaus_cuda_from_cpu(gaus)
        self.raster_settings["sh_degree"] = int(np.round(np.sqrt(self.gaussians.sh_dim))) - 1

    def update_gaussian_data_from_pvg(self, pc: GaussianModel, cam_time_shift=0.1, time_shift=0.1):
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        means3D = pc.get_xyz_SHM(cam_time_shift)
        marginal_t = pc.get_marginal_t(cam_time_shift)
        opacity = opacity * marginal_t
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features

        features = torch.zeros_like(means3D[:, :0])
        s_other = 0

        mask = marginal_t[:, 0] > 0.05
        masked_means3D = means3D[mask]
        masked_xyz_homo = torch.cat([masked_means3D, torch.ones_like(masked_means3D[:, :1])], dim=1)
        masked_depth = (masked_xyz_homo @ self.raster_settings["viewmatrix"][:, 2:3])
        depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32).to(masked_depth.device)
        depth_alpha[mask] = torch.cat([masked_depth, torch.ones_like(masked_depth).to(masked_depth.device)], dim=1)
        features = torch.cat([features, depth_alpha], dim=1)

        self.need_rerender = True
        self.gaussians = GaussianDataCUDA(
            xyz = means3D.float().cuda(),
            rot = rotations.float().cuda(),
            scale = scales.float().cuda(),
            opacity = opacity.float().cuda(),
            sh = shs.float().cuda(),
            features = features.cuda(),
            mask = mask.cuda(),
        )
    def update_env_map(self, env_map):
        self.env_map = env_map
    def sort_and_update(self, camera: util.Camera):
        self.need_rerender = True

    def set_scale_modifier(self, modifier):
        self.need_rerender = True
        self.raster_settings["scale_modifier"] = float(modifier)

    def set_render_mod(self, mod: int):
        self.need_rerender = True

    def set_gl_texture(self, h, w):
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            w,
            h,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to register opengl texture")
    
    def set_render_reso(self, w, h):
        self.need_rerender = True
        self.raster_settings["image_height"] = int(h)
        self.raster_settings["image_width"] = int(w)
        gl.glViewport(0, 0, w, h)
        self.set_gl_texture(h, w)
        self.grid = kornia.utils.create_meshgrid(int(self.raster_settings["image_height"]), int(self.raster_settings["image_width"]), normalized_coordinates=False, device='cuda')[0]

    def update_camera_pose(self, camera: util.Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        proj[[2, 3], :] = -proj[[2, 3], :]
        self.raster_settings["viewmatrix"] = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings["campos"] = torch.tensor(camera.position).float().cuda()
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()

    def update_camera_intrin(self, camera: util.Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        proj[[2, 3], :] = -proj[[2, 3], :]
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()
        self.focal = camera.fy
        self.cx = camera.cx
        self.cy = camera.cy
    
    def get_world_direction(self):
        u, v = self.grid.unbind(-1)
        directions = torch.stack(
            [
                (u - self.cx + 0.5) / self.focal,
                (v - self.cy + 0.5) / self.focal,
                torch.ones_like(u),
            ], dim=0
        )
        directions = F.normalize(directions, dim=0)
        c2w = self.raster_settings["viewmatrix"].transpose(0, 1).inverse()[:3, :3]
        directions = (c2w @ directions.reshape(3, -1)).reshape(3, int(self.raster_settings["image_height"]), int(self.raster_settings["image_width"]))
        return directions
        
    def draw(self):
        if self.reduce_updates and not self.need_rerender:
            gl.glUseProgram(self.program)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            return

        self.need_rerender = False

        raster_settings = GaussianRasterizationSettings(**self.raster_settings)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        with torch.no_grad():
            if self.gaussians.mask is not None:
                contrib, img, rendered_feature, radii = rasterizer(
                    means3D = self.gaussians.xyz,
                    means2D = None,
                    shs = self.gaussians.sh,
                    colors_precomp = None,
                    features = self.gaussians.features,
                    opacities = self.gaussians.opacity,
                    scales = self.gaussians.scale,
                    rotations = self.gaussians.rot,
                    cov3D_precomp = None,
                    mask = self.gaussians.mask,
                )
                rendered_other, rendered_depth, rendered_opacity = rendered_feature.split([0, 1, 1], dim=0)
                world_direction = self.get_world_direction()
                bg_color_from_env_map = self.env_map(world_direction.permute(1, 2, 0)).permute(2, 0, 1)
                img = img + (1 - rendered_opacity) * bg_color_from_env_map
            else:
                img, radii = rasterizer(
                    means3D = self.gaussians.xyz,
                    means2D = None,
                    shs = self.gaussians.sh,
                    colors_precomp = None,
                    opacities = self.gaussians.opacity,
                    scales = self.gaussians.scale,
                    rotations = self.gaussians.rot,
                    cov3D_precomp = None
                )

        img = img.permute(1, 2, 0)
        img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
        img = img.contiguous()
        height, width = img.shape[:2]
        # transfer
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to map graphics resource")
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to get mapped array")
        
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            img.data_ptr(),
            4 * 4 * width,
            4 * 4 * width,
            height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to copy from tensor to texture")
        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to unmap graphics resource")

        gl.glUseProgram(self.program)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
