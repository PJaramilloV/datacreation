import numpy as np
import torch
import open3d as o3d
import sys

# Load the .npy file containing the point cloud data as a PyTorch tensor
data_dir = 'data/pjaramil/__precol/repair/'
file = 'NA-bottle_599.npy'
if len(sys.argv) >1:
    if len(sys.argv) != 3:
        print("If arguments are given data_dir AND file are required")
        sys.exit(1)
    data_dir = sys.argv[1]
    file = sys.argv[2]
npy_file_path = data_dir + file
point_cloud_tensor = np.load(npy_file_path)

# Ensure that the tensor is in the right shape (N, 3) where N is the number of points
# If your tensor has different dimensions, you may need to reshape it accordingly.
# For example, if your tensor has shape (3, N), you can transpose it.
if point_cloud_tensor.shape[0] != 3:
    point_cloud_tensor = point_cloud_tensor.T

# Extract the XYZ coordinates from the tensor
points = point_cloud_tensor.T

# Define the output .ply file path
ply_file_path = npy_file_path.split('.npy')[0] + '_view.ply'

def export_pointcloud(name, points, normals=None):
    if len(points.shape) > 2:
        points = points[0]
        if normals is not None:
            normals = normals[0]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if normals is not None:
            normals = normals.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(name, pcd)

export_pointcloud(ply_file_path, points)

print(f"Point cloud data saved to {ply_file_path}")
