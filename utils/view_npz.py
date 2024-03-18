from skimage import measure
import open3d as o3d
import numpy as np
import trimesh
import torch
import sys
import meshlib.mrmeshpy  as mr
import meshlib.mrmeshnumpy as mrn



# Load the .npy file containing the point cloud data as a PyTorch tensor
data_dir = 'data/pjaramil/__iberian/adapted/'
file = 'exc_vox.npz'

def print_extremes(PointCloud, prefix):
    x_coords = PointCloud[:, 0]
    y_coords = PointCloud[:, 1]
    z_coords = PointCloud[:, 2]

    # Find extents along each axis
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    print(f"{prefix}coordinate limits: \n  ({x_min:4.3f},{x_max:4.3f})\n  ({y_min:4.3f},{y_max:4.3f})\n  ({z_min:4.3f},{z_max:4.3f})")

def process_pc(file_path):
    data = np.load(file_path)
    points = data['points']  # Assuming 'points' is the key for the point cloud data
    print_extremes(points, "Old ")
    points *= 512
    
    # Create an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    print(f"New coordinate bounds: {point_cloud.get_min_bound()},{point_cloud.get_max_bound()}")

    # Save the point cloud in .ply format
    new_name = export_name(file_path)
    o3d.io.write_point_cloud(new_name, point_cloud)
    return new_name 

def process_vox(file_path):
    voxel_data = np.load(file_path)
    voxels = voxel_data['psr']
    print(f"Object original extents: {voxels.shape}")

    # # Convert voxel data to a mesh representation (e.g., using marching cubes algorithm)
    # vertices, faces, _, _ = measure.marching_cubes(voxels)

    # # Create a trimesh object from the mesh representation
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # print(f"Object new extents:     {mesh.bounds}")

    

    # Save the mesh in .ply format
    new_name = export_name(file_path)
    #mesh.export(new_name)}

    voxels[voxels!= 0] = 1
    simpleVolume = mrn.simpleVolumeFrom3Darray(voxels)
    floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume )
    mesh = mr.gridToMesh(floatGrid, mr.Vector3f(1, 1, 1), 0.1)
    mr.saveMesh(mesh, new_name)
    return new_name

def export_name(file_path):
    return file_path.split('.npz')[0] + '_view.ply'

cases = {
    'pc': process_pc,
    'vox': process_vox,
}

def export_ply(name):
    data_type = name.split('_')[-1].split('.')[0]
    out_file = cases.get(data_type, None)(name)
     
    if out_file is None:
        raise KeyError(f"Unspecified data type of file: {data_type} from file {name}")
    return out_file

if __name__ == '__main__':
    if len(sys.argv) >1:
        if len(sys.argv) != 3:
            print("If arguments are given data_dir AND file are required")
            sys.exit(1)
        data_dir = sys.argv[1]
        file = sys.argv[2]

    npz_file_path = data_dir + file
    np.set_printoptions(precision=5, suppress=True)
    ply_file_path = export_ply(npz_file_path)

    print(f"Point cloud data saved to {ply_file_path}")
