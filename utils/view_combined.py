import open3d as o3d
import numpy as np
import os
import subprocess

def colorize_point_cloud(point_cloud, color):
    # Create a color array with the same size as the point cloud
    colors = np.full((len(point_cloud.points), 3), color, dtype=np.float32)
    
    # Assign the colors to the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def npy_version(file:str):
    return file.split('_view')[0] + '.npy'

def open_file_in_vscode(file_path):
    # Get the path to the VSCode executable
    vscode_path = subprocess.check_output("which code", shell=True, text=True).strip()

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Use the `code` command to open the file in the active VSCode window
    subprocess.run([vscode_path, "--add", file_path])

def get_gt(npy_input_file):
    results_dir = npy_input_file.split('results')[0]
    file = f"{npy_input_file.split('syn/broken')[1].split('/')[0]}.npy"
    repair_file = os.path.join(results_dir, 'repair', file)
    pcd = np.load(repair_file)
    idxs = np.random.choice(np.arange(0, pcd.shape[0],1), 800, replace=False)
    pcd = pcd[idxs]
    return pcd

def merge_point_clouds_and_filter(ply_file1, ply_file2, output_file, filter_duplicates=True, script_dir ='',data_dir=''):

    # Check if the input PLY files exist
    if not os.path.exists(ply_file1):
        subprocess.run(['python', f'{script_dir}view_npy.py', data_dir, npy_version(ply_file1)])
    if not os.path.exists(ply_file2):
        subprocess.run(['python', f'{script_dir}view_npy.py', data_dir, npy_version(ply_file2)])

    if data_dir:
        ply_file1 = data_dir + ply_file1
        ply_file2 = data_dir + ply_file2
        output_file = data_dir + output_file

    # Load the point clouds from the PLY files
    pcd1 = o3d.io.read_point_cloud(ply_file1)
    pcd2 = o3d.io.read_point_cloud(ply_file2)

    # Colorize the point clouds
    colorize_point_cloud(pcd1, [1.0, 0.0, 0.0])  # Red
    colorize_point_cloud(pcd2, [0.0, 1.0, 0.0])  # Green

    # Merge the point clouds
    merged_pcd = pcd1 + pcd2

    if filter_duplicates:
        # Remove points in pcd2 that are present in pcd1
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.001)  # Optional downsampling for faster processing

    if optimize_space:
        print(f"Deleted file {ply_file2} to save disk space")
        os.remove(ply_file2)

    # Save the merged point cloud to a new PLY file
    o3d.io.write_point_cloud(output_file, merged_pcd)
    print(f"saved: {output_file}")

    if view_files_now:
        open_file_in_vscode(output_file)
        open_file_in_vscode(ply_file1)

view_files_now = True
optimize_space = True
if __name__ == "__main__":
    data_dir = 'data/pjaramil/__precol/results/syn/brokenNA-bottle_599'
    ply_file1 = "input_view.ply"
    ply_file2 = "sample_view.ply"
    output_file = "merged.ply"
    script_dir = "code/"
    filter_duplicates = True  # Set this to True to filter duplicates, or False to keep all points

    data_dir += '/' if data_dir[-1] != '/' else ''
    
    merge_point_clouds_and_filter(ply_file1, ply_file2, output_file, filter_duplicates, script_dir=script_dir ,data_dir=data_dir)