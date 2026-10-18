"""Projection of a camera ray to an explicitly assumed horizontal plane."""
import numpy as np


def project_to_plane(center_xy, image_size, geometry: dict, plane_z_m: float):
    x, y = center_xy
    width, height = image_size
    half_y = np.tan(np.deg2rad(geometry["vertical_fov_deg"]) / 2)
    ray_camera = np.array([(2*x - 1) * half_y * width/height, (1 - 2*y) * half_y, -1.0])
    ray = np.asarray(geometry["rotation_camera_to_world"]) @ ray_camera
    origin = np.asarray(geometry["position_m"])
    if abs(ray[2]) < 1e-8:
        return None
    distance = (plane_z_m - origin[2]) / ray[2]
    if distance <= 0:
        return None
    return origin + distance * ray
