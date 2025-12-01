from center.center_tracker import CenterTracker
from radius.radius_tracker import RadiiTracker
import pandas as pd
from utils.evaluation import Evaluator


config={'r_in_center': 3, 'r_out_center': 7, 'r_in_radius': 3, 'r_out_radius': 8, 'ring_index': 1, 'brightness': 49, 'clipLimit': 5, 'titleGridSize1': 4, 'titleGridSize2': 4, 'blur_level1': 5, 'blur_level2': 5, 'dp': 0.9, 'param1': 200, 'param2': 0.9, 'resolution': 0.01, 'size': 50, 'rolling_window': 'median', 'window': 50, 'movement_thresh': 1.4, 'output_name': f'009_obj1','output_path':'/scratch/lustre/home/tana9239/HyperparamOpt/Videos/Video009',
            'excel_file': '/scratch/lustre/home/tana9239/HyperparamOpt/hpc_excels_250/009_obj1.xlsx'}

# config={'r_in_center': 3, 'r_out_center': 7, 'r_in_radius': 3, 'r_out_radius': 8, 'ring_index': 1, 'brightness': 49, 'clipLimit': 5, 'titleGridSize1': 4, 'titleGridSize2': 4, 'blur_level1': 5, 'blur_level2': 5, 'dp': 0.9, 'param1': 200, 'param2': 0.9, 'resolution': 0.01, 'size': 50, 'rolling_window': 'median', 'window': 50, 'movement_thresh': 1.4, 'output_name': f'036_obj2','output_path':'/scratch/lustre/home/tana9239/HyperparamOpt/Videos/Video036',
#             'excel_file': '/scratch/lustre/home/tana9239/HyperparamOpt/hpc_excels_250/036_obj2.xlsx'}

center_track=CenterTracker(config,[90,99],250)
# center_track=CenterTracker(config,[207,605],250)
center_track.load_metadata("/scratch/lustre/home/tana9239/HyperparamOpt")
centers=center_track.run_tracker(frame_range=[0,400])
print(centers)
centers.to_excel('/scratch/lustre/home/tana9239/ref_test/test_centers.xlsx')
# centers=pd.read_excel('/scratch/lustre/home/tana9239/ref_test/test_centers.xlsx')



radius_track=RadiiTracker(config,[90,99],250)
radius_track._load_metadata("/scratch/lustre/home/tana9239/HyperparamOpt")
final=radius_track.run_tracker(centers,frame_range=[0,400])

# final=pd.read_excel('/scratch/lustre/home/tana9239/ref_test/test_radii_preeval.xlsx')

# evaluation=Evaluator(final,'/scratch/lustre/home/tana9239/HyperparamOpt/hpc_excels_250/036_obj2.xlsx',config,start_frames=0,end_frames=400)
evaluation=Evaluator(final,'/scratch/lustre/home/tana9239/HyperparamOpt/hpc_excels_250/009_obj1.xlsx',config,start_frames=0,end_frames=400)
evaluation.evaluate_center()
evaluation.print_center_eval()
evaluation.evaluate_radius()
evaluation.print_radius_eval()

print(final)
final.to_excel('/scratch/lustre/home/tana9239/ref_test/test_radii.xlsx')