from center.center_tracker import CenterTracker
from radius.radius_tracker import RadiiTracker


config={'r_in_center': 3, 'r_out_center': 7, 'r_in_radius': 3, 'r_out_radius': 8, 'ring_index': 1, 'brightness': 49, 'clipLimit': 5, 'titleGridSize1': 4, 'titleGridSize2': 4, 'blur_level1': 5, 'blur_level2': 5, 'dp': 0.9, 'param1': 200, 'param2': 0.9, 'resolution': 0.01, 'size': 50, 'rolling_window': 'median', 'window': 50, 'movement_thresh': 1.4, 'output_path': '005_obj1',
            'excel_file': '/scratch/lustre/home/tana9239/HyperparamOpt/hpc_excels_250/005_obj1.xlsx'}

center_track=CenterTracker(config,[116,278],250)
center_track.load_metadata("/scratch/lustre/home/tana9239/HyperparamOpt")
print(center_track.config)
# centers=center_track.run_tracker()
print(centers)