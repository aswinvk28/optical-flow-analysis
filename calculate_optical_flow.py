# work from SceneNet
# https://robotvault.bitbucket.io/scenenet-rgbd.html

import math
import numpy as np
import os
import pathlib
import random
import scenenet_pb2 as sn
import sys
import scipy.misc
import cv2
import imageio
import json
import matplotlib

def normalize(v):
    return v/np.expand_dims(np.linalg.norm(v, axis=2),2)

def normalize_norm(v):
    return v/np.linalg.norm(v)

def load_depth_map_in_m(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (320,240))
    return (image * 0.001)

def pixel_to_ray(pixel,vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    x, y = pixel
    data = np.zeros((pixel_height, pixel_width, 3))
    x_vect = np.tan(np.radians(hfov/2.0)) * ((2.0 * ((x+0.5)/pixel_width)) - 1.0)
    y_vect = np.tan(np.radians(vfov/2.0)) * ((2.0 * ((y+0.5)/pixel_height)) - 1.0)
    data[:,:,0] = x_vect
    data[:,:,1] = y_vect
    data[:,:,2] = 1.0
    return data

def normalised_pixel_to_ray_array(width=320,height=240):
    pixel_to_ray_array = np.zeros((height,width,3))
    pixel = np.meshgrid(np.arange(0,width), np.arange(0,height))
    pixel_to_ray_array[:,:,:] = normalize(
        pixel_to_ray(pixel,pixel_height=height,pixel_width=width)
    )
    return pixel_to_ray_array

def points_in_camera_coords(depth_map,pixel_to_ray_array):
    assert depth_map.shape[0] == pixel_to_ray_array.shape[0]
    assert depth_map.shape[1] == pixel_to_ray_array.shape[1]
    assert len(depth_map.shape) == 2
    assert pixel_to_ray_array.shape[2] == 3
    camera_relative_xyz = np.ones((depth_map.shape[0],depth_map.shape[1],4))
    for i in range(3):
        camera_relative_xyz[:,:,i] = depth_map * pixel_to_ray_array[:,:,i]
    return camera_relative_xyz

def flatten_points(points):
    return points.reshape(-1, 4)

def reshape_points(height,width,points):
    other_dim = points.shape[1]
    return points.reshape(height,width,other_dim)

def transform_points(transform,points):
    assert points.shape[2] == 4
    height = points.shape[0]
    width = points.shape[1]
    points = flatten_points(points)
    return reshape_points(height,width,(transform.dot(points.T)).T)

def world_to_camera_with_pose(view_pose):
    lookat_pose = position_to_np_array(view_pose.lookat)
    camera_pose = position_to_np_array(view_pose.camera)
    up = np.array([0,1,0])
    R = np.diag(np.ones(4))
    R[2,:3] = normalize_norm(lookat_pose - camera_pose)
    R[0,:3] = normalize_norm(np.cross(R[2,:3],up))
    R[1,:3] = -normalize_norm(np.cross(R[0,:3],R[2,:3]))
    T = np.diag(np.ones(4))
    T[:3,3] = -camera_pose
    return R.dot(T)

def camera_to_world_with_pose(view_pose):
    return np.linalg.inv(world_to_camera_with_pose(view_pose))

def camera_point_to_uv_pixel_location(point,vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    point = point / point[2]
    u = ((pixel_width/2.0) * ((point[0]/math.tan(math.radians(hfov/2.0))) + 1))
    v = ((pixel_height/2.0) * ((point[1]/math.tan(math.radians(vfov/2.0))) + 1))
    return (u,v)

def position_to_np_array(position):
    return np.array([position['x'],position['y'],position['z']]) if type(position) == dict else \
        np.array([position.x,position.y,position.z])

def interpolate_poses(start_pose,end_pose,alpha):
    assert alpha >= 0.0
    assert alpha <= 1.0
    camera_pose = alpha * position_to_np_array(end_pose['camera'])
    camera_pose += (1.0 - alpha) * position_to_np_array(start_pose['camera'])
    lookat_pose = alpha * position_to_np_array(end_pose['lookat'])
    lookat_pose += (1.0 - alpha) * position_to_np_array(start_pose['lookat'])
    timestamp = alpha * end_pose['timestamp'] + (1.0 - alpha) * start_pose['timestamp']
    pose = sn.Pose()
    pose.camera.x = camera_pose[0]
    pose.camera.y = camera_pose[1]
    pose.camera.z = camera_pose[2]
    pose.lookat.x = lookat_pose[0]
    pose.lookat.y = lookat_pose[1]
    pose.lookat.z = lookat_pose[2]
    pose.timestamp = timestamp
    return pose

def world_point_to_uv_pixel_location_with_interpolated_camera(point,shutter_open,shutter_close,alpha):
    view_pose = interpolate_poses(shutter_open,shutter_close,alpha)
    wTc = world_to_camera_with_pose(view_pose)
    point_in_camera_coords = wTc.dot(np.array(point))
    uv = camera_point_to_uv_pixel_location(point_in_camera_coords)
    return uv

# Expects:
# an nx4 array of points of the form [[x,y,z,1],[x,y,z,1]...] in world coordinates
# a length three array [x,y,z] for camera start and end (of a shutter) and lookat start/end in world coordinates

# Returns:
# a nx2 array of the horizontal and vertical pixel location time derivatives (i.e. pixels per second in the horizontal and vertical)
# NOTE: the pixel coordinates are defined as (0,0) in the top left corner, to (320,240) in the bottom left
def optical_flow(points,shutter_open,shutter_close,alpha=0.5,shutter_time=(1.0/60),
                 hfov=60,pixel_width=320,vfov=45,pixel_height=240):
    # Alpha is the linear interpolation coefficient, 0.5 takes the derivative in the midpoint
    # which is where the ground truth renders are taken.  The photo render integrates via sampling
    # over the whole shutter open-close trajectory
    view_pose = interpolate_poses(shutter_open,shutter_close,alpha)
    wTc = world_to_camera_with_pose(view_pose)
    camera_pose = position_to_np_array(view_pose['camera'] if type(view_pose) == dict else view_pose.camera)
    lookat_pose = position_to_np_array(view_pose['lookat'] if type(view_pose) == dict else view_pose.lookat)

    # Get camera pixel scale constants
    uk = (pixel_width/2.0) * ((1.0/math.tan(math.radians(hfov/2.0))))
    vk = (pixel_height/2.0) * ((1.0/math.tan(math.radians(vfov/2.0))))

    # Get basis vectors
    ub1 = lookat_pose - camera_pose
    b1 = normalize_norm(ub1)
    ub2 = np.cross(b1,np.array([0,1,0]))
    b2 = normalize_norm(ub2)
    ub3 = np.cross(b2,b1)
    b3 = -normalize_norm(ub3)

    # Get camera pose alpha derivative
    camera_end = position_to_np_array(shutter_close['camera'])
    camera_start = position_to_np_array(shutter_open['camera'])
    lookat_end = position_to_np_array(shutter_close['lookat'])
    lookat_start= position_to_np_array(shutter_open['lookat'])
    dc_dalpha = camera_end - camera_start

    # Get basis vector derivatives
    # dub1 means d unnormalised b1
    db1_dub1 = (np.eye(3) - np.outer(b1,b1))/np.linalg.norm(ub1)
    dub1_dalpha = lookat_end - lookat_start - camera_end + camera_start
    db1_dalpha = db1_dub1.dot(dub1_dalpha)
    db2_dub2 = (np.eye(3) - np.outer(b2,b2))/np.linalg.norm(ub2)
    dub2_dalpha = np.array([-db1_dalpha[2],0,db1_dalpha[0]])
    db2_dalpha = db2_dub2.dot(dub2_dalpha)
    db3_dub3 = (np.eye(3) - np.outer(b3,b3))/np.linalg.norm(ub3)
    dub3_dalpha = np.array([
            -(db2_dalpha[2]*b1[1]+db1_dalpha[1]*b2[2]),
            -(db2_dalpha[0]*b1[2] + db1_dalpha[2]*b2[0])+(db2_dalpha[2]*b1[0]+db1_dalpha[0]*b2[2]),
            (db1_dalpha[1]*b2[0]+db2_dalpha[0]*b1[1])
        ])
    db3_dalpha = -db3_dub3.dot(dub3_dalpha)

    # derivative of the rotated translation offset
    dt3_dalpha = np.array([
            -db2_dalpha.dot(camera_pose)-dc_dalpha.dot(b2),
            -db3_dalpha.dot(camera_pose)-dc_dalpha.dot(b3),
            -db1_dalpha.dot(camera_pose)-dc_dalpha.dot(b1),
        ])

    # camera transform derivative
    dT_dalpha = np.empty((4,4))
    dT_dalpha[0,:3] = db2_dalpha
    dT_dalpha[1,:3] = db3_dalpha
    dT_dalpha[2,:3] = db1_dalpha
    dT_dalpha[:3,3] = dt3_dalpha

    # Calculate 3D point derivative alpha derivative
    dpoint_dalpha = dT_dalpha.dot(points.T)
    point_in_camera_coords = wTc.dot(np.array(points.T))

    # Calculate pixel location alpha derivative
    du_dalpha = uk * (dpoint_dalpha[0] * point_in_camera_coords[2] - dpoint_dalpha[2] * point_in_camera_coords[0])
    dv_dalpha = vk * (dpoint_dalpha[1] * point_in_camera_coords[2] - dpoint_dalpha[2] * point_in_camera_coords[1])
    du_dalpha = du_dalpha/(point_in_camera_coords[2]*point_in_camera_coords[2])
    dv_dalpha = dv_dalpha/(point_in_camera_coords[2]*point_in_camera_coords[2])

    # Calculate pixel location time derivative
    du_dt = du_dalpha / shutter_time
    dv_dt = dv_dalpha / shutter_time
    return np.vstack((du_dt,dv_dt)).T

def flow_to_hsv_image(flow, magnitude_scale=1.0/100.0):

    height = 240
    width = 320
    pixel = np.meshgrid(np.arange(0,height), np.arange(0,width))
    hsv = np.empty((height,width,3))
    v = np.linalg.norm(flow, axis=2)
    idxs = np.where(v < 1e-8)
    hsv[idxs[0],idxs[1],0:3] = 0.0
    idxs = np.where(v >= 1e-8)
    direction = flow[idxs[0],idxs[1],:] / np.expand_dims(v[idxs],1)
    theta = np.arctan2(direction[:,1],direction[:,0])
    theta[theta<=0] = theta[theta<=0] + 2*np.pi
    if np.sum((theta < 0) | (theta > 2*np.pi)) > 0:
        raise Exception("Invalid value for theta")

    hsv[idxs[0],idxs[1],0] = theta / (2*np.pi)
    hsv[idxs[0],idxs[1],1] = 1.0
    hsv[idxs[0],idxs[1],2] = np.min(np.dstack((v[idxs].flatten() * magnitude_scale, np.ones(v[idxs].flatten().shape[0]))), axis=2).flatten()
    return hsv

def compute_flow(image, view=None):
    if view is None:
        view = json.load(open("view.json", "r"))

    depth_map = load_depth_map_in_m(image)
    cached_pixel_to_ray_array = normalised_pixel_to_ray_array(width=depth_map.shape[1],height=depth_map.shape[0])
    depth_map[depth_map == 0.0] = 1000.0

    # This is a 320x240x3 array, with each 'pixel' containing the 3D point in camera coords
    points_in_camera = points_in_camera_coords(depth_map,cached_pixel_to_ray_array)

    # Transform point from camera coordinates into world coordinates
    ground_truth_pose = interpolate_poses(view['shutter_open'],view['shutter_close'],0.5)
    camera_to_world_matrix = camera_to_world_with_pose(ground_truth_pose)
    points_in_world = transform_points(camera_to_world_matrix,points_in_camera)

    # Calculate optical flow
    points_in_world = flatten_points(points_in_world)

    uv = world_point_to_uv_pixel_location_with_interpolated_camera(
        points_in_world.T,
        view['shutter_open'],
        view['shutter_close'],
        0.5
    )
    uv = np.dstack((uv[0].reshape(240,320), uv[1].reshape(240,320)))

    optical_flow_derivatives = optical_flow(points_in_world,view['shutter_open'],view['shutter_close'])
    optical_flow_derivatives = reshape_points(240,320,optical_flow_derivatives)

    # Write out hsv optical flow image.  We use the matplotlib hsv colour wheel
    hsv = flow_to_hsv_image(optical_flow_derivatives)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    return rgb, optical_flow_derivatives, uv
