
"""
you need to install trimesh and pyrender if you want to render mesh
pip install trimesh
pip install pyrender
"""

import os
import pdb
import time

import trimesh
import pyrender
import numpy as np
import torch

def create_raymond_lights():
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


class Renderer:

    def __init__(self, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """

        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1, 0, 0], rot_angle=0, is_right=1):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(),
                                   vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t=None,
            rot=None,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
    ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])
        if is_right:
            mesh_base_color = mesh_base_color[::-1]
        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle,
                                        is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)


