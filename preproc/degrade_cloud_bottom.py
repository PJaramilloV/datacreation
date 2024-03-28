import multiprocessing
from tqdm import tqdm
import numpy as np
import subprocess 
import argparse
import mcubes
import os

parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--threads', type=int, default=14, help="define number of threads")
parser.add_argument('--dataset', type=str, default='', help='directory housing a dataset complete objects .npy files') 
parser.add_argument('--breakage', type=float, default=0.20, help='percentage of the bottom body to be broken off completely')
parser.add_argument('--variance', type=float, default=0.1, help='random factor by which to modify breakage height, such that breakage_height = breakage * (1 +- variance*random(0,1) )')
parser.add_argument('--voxsize', type=int, default=256, help='length of the cubic space for meshes')
parser.add_argument('--jagged', type=float, default=0.25, help='percentage of the bottom body above the complete breakage to be part of a partial breakage (additive with --breakage)')
parser.add_argument('--angle', type=float, default=0.75, help='depression angle (degrees [0-90[) of the jagged breakage')
parser.add_argument('--jags', type=float, default=7, help='number of "teeths" protuding from the upper body through the breakage')
parser.add_argument('--maxx', type=float, default=0, help='maximum rotation appliable to the objects in the X axis in both directions (in degrees)')
parser.add_argument('--maxy', type=float, default=0, help='maximum rotation appliable to the objects in the Y axis in both directions (in degrees)')
parser.add_argument('--maxz', type=float, default=0, help='maximum rotation appliable to the objects in the Z axis in both directions (in degrees)')
parser.add_argument('--seed', type=int, default=1234, help='sets numpy random seed')
opt = parser.parse_args()
breakage = opt.breakage
variance = opt.variance
dataset = opt.dataset
voxsize = opt.voxsize
grid_range = [(0,voxsize) for _ in range(3)]
downward = np.array([0,-1,0])
np.random.seed(opt.seed)

def remove_bottom_points_v2(point_cloud, jagged, jagged_angle, teeth):
    #   EXPERIMENTAL - NOT WORKING
    # Calculate the thresholds for the breakages
    y_min = np.min(point_cloud[:, 1]) 
    y_max = np.max(point_cloud[:, 1])
    
    full_breakoff = y_min + breakage * (y_max - y_min)
    partial_breakoff = y_min + jagged * (y_max - y_min)
    teeth_to_make = teeth

    depression_angle = np.radians(90-jagged_angle)
    print(y_min, partial_breakoff)
    y_within_range = np.logical_and(point_cloud[:, 1] <= partial_breakoff, True) # point_cloud[:, 1] >= full_breakoff, 
    points_within_range = point_cloud[y_within_range]

    # Create a mask to identify points with no points underneath
    mask = np.ones(len(points_within_range), dtype=bool)

    broken_off = point_cloud[point_cloud[:, 1] < full_breakoff]
    
    while teeth_to_make:
        point_idx = np.random.randint(0, points_within_range.shape[0])
        print(point_idx)
        if not mask[point_idx]: # broken off already
            continue

        mask[point_idx] = 0
        break_seed = points_within_range[point_idx]
        print(break_seed)

        vectors = points_within_range - break_seed
        dot_prds = np.dot(vectors, downward)
        magnitudes = np.linalg.norm(vectors, axis=1)
        cos_th = dot_prds / magnitudes
        angles = np.arccos(cos_th)
        print(angles[angles > depression_angle].shape[0], angles.shape[0])
        mask[angles > depression_angle] = 0

        teeth_to_make -= 1
        if np.sum(mask) == 0:
            print('all gone')
            break
        print('.')
        break


    filtered_indices = mask
    for p in points_within_range:
        pass #print(p)

    # Apply the mask to the points within the range
    jagged_cloud = points_within_range[filtered_indices]

    # Remove filtered points from the original point cloud
    filtered_points = np.where(y_within_range)[0][filtered_indices]
    point_cloud = np.delete(point_cloud, filtered_points, axis=0)
    point_cloud = point_cloud[point_cloud[:, 1] > full_breakoff]

    # Append points underneath the bottom 10% to the filtered point cloud
    #jagged_cloud = np.vstack((jagged_cloud, np.array(broken_off)))

    return point_cloud, jagged_cloud

def rotX(theta):
    "Theta in radians"
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def rotY(theta):
    "Theta in radians"
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotZ(theta):
    "Theta in radians"
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotate_points(points):
    theta_x, theta_y, theta_z = opt.maxx, opt.maxy, opt.maxz
    thetas = [th * np.pi /180 for th in [theta_x, theta_y, theta_z]]
    # np.random.random doesn't work randomly through threads
    #   solution: make a random state taking a coordinate sum from the 
    #             coordinates of a point of the object shape
    rs = np.random.RandomState(int(sum( points[points.shape[0]//2]  )%2**31))
    for th, rotation in zip(thetas, [rotX, rotY, rotZ]):
        if not th:
            continue
        applied_th = (1 - 2* rs.random()) * th
        points = np.dot(points, rotation(applied_th).T)
    return points, rs


def volume_from_points(points):
    max_val = np.max(points)
    points = (points / max_val) * voxsize * 0.975
    volume, _ = np.histogramdd(points, bins=voxsize, range=grid_range)
    return volume

def remove_bottom_points(point_cloud, rand_state=np.random.RandomState(0)):
    y_min = np.min(point_cloud[:, 1]) 
    y_max = np.max(point_cloud[:, 1])
    
    full_breakoff = y_min + (y_max - y_min) * breakage * (1 + variance * (1 - 2 * rand_state.random()) ) 
    broken, piece = point_cloud[point_cloud[:, 1] > full_breakoff], point_cloud[point_cloud[:, 1] <= full_breakoff]
    return broken, piece

def break_piece(npy_path):
    pc = np.load(npy_path)
    pc, rs = rotate_points(pc)
    broken_cloud, piece_cloud = remove_bottom_points(pc, rs)
    return broken_cloud, piece_cloud

def break_and_save(npy_path):
    broken, piece = break_piece(npy_path)

    broken_npy_path = npy_path.replace('complete', 'broken')
    repair_npy_path = npy_path.replace('complete', 'repair')

    np.save(broken_npy_path, broken)
    np.save(repair_npy_path, piece)

    if len(broken.shape) != 3:
        if broken.shape[0] == 3:
            broken = broken.T
        if broken.shape[1] == 3:
            broken = volume_from_points(broken)
    if len(piece.shape) != 3:
        if piece.shape[0] == 3:
            piece = piece.T
        if piece.shape[1] == 3:
            piece = volume_from_points(piece)
    
    if len(broken.shape) != 3 or len(piece.shape) != 3:
        raise ValueError(f"Check array dimensions, should be (x,x,x). Shapes (3,x) and (x,3) are supported but this is {broken.shape} and {piece.shape}")
    
    broken_v, broken_f = mcubes.marching_cubes(broken, 0)
    repair_v, repair_f = mcubes.marching_cubes(piece, 0)

    broken_obj_path = broken_npy_path.replace('.npy','.obj')
    repair_obj_path = repair_npy_path.replace('.npy','.obj')

    mcubes.export_obj(broken_v, broken_f, broken_obj_path)
    mcubes.export_obj(repair_v, repair_f, repair_obj_path)
    


def dev():
    jagged = opt.jagged + breakage
    jagged_angle = opt.angle
    teeth = opt.jags

    assert 0 < breakage < 1
    assert breakage <= jagged < 1
    assert 0 <= jagged_angle < 90 
    assert 0 < teeth

    

    # Load the point cloud data from .npy file
    data_dir = 'path/to/data' # 'data/pjaramil/__precol/CH_mesh/bowl/test/' (PJV)
    file_path = "file.npy"  # Update this with your file path
    point_cloud_data = np.load(data_dir+file_path)
    points = np.random.randint(0, point_cloud_data.shape[0], 32768//2)
    point_cloud_data = point_cloud_data[points, :]

    broken, fallen_out = remove_bottom_points_v2(point_cloud_data, jagged, jagged_angle, teeth)
    np.save(data_dir+'_candidate.npy', broken)
    np.save(data_dir+'_tmp.npy', fallen_out)

    # Save or use the filtered point cloud data as required
    subprocess.run(['python','utils/view_npy.py',data_dir,'_tmp.npy'])
    subprocess.run(['python','utils/view_npy.py',data_dir,'_candidate.npy'])

if __name__ == '__main__':
    if not dataset:
        dev()
        exit()

    # ---- Mass processing ----
    data_dir = 'data/pjaramil/' # (PJV)
    threads = opt.threads
    parallel = opt.multiprocessing
    directory = os.path.join(data_dir, dataset, 'complete')

    counter = 0
    database = []
    for root, dirs, files in os.walk(directory, topdown=False):
        repair_dir = root.replace('complete', 'repair')
        for filename in files:
            if filename.endswith('.npy'):
                counter += 1
                if not os.path.exists(os.path.join(repair_dir, filename)):
                    datapoint = ''
                    datapoint = os.path.join(root, filename)
                    database.append(datapoint)
    print(f"Found {len(database) }/{counter} files unprocessed")
    if parallel:
        pool = multiprocessing.Pool(threads)
        try:
            for _ in tqdm(pool.imap_unordered(break_and_save, database), total=len(database)):
                pass
        except KeyboardInterrupt:
            exit()
        pool.close()