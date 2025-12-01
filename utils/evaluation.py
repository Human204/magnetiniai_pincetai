import os

import cv2

import numpy as np

import pandas as pd

def mad(data):
    return np.median(np.abs(data - np.median(data)))
def calculate_distance(predicted, actual):
    return np.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)
def should_ignore_frame(frame_idx, ignored_frames_list):
    """Check if a frame should be ignored"""
    return frame_idx in ignored_frames_list

class Evaluator:
    def __init__(self,results_df,true_excel,config,start_frames=0,end_frames=5000,ignored_frames=None):
        self.results_df=results_df
        self.true_df=pd.read_excel(true_excel,header=1)

        if ignored_frames is None:
            self.ignored_frames = []
        else:
            self.ignored_frames=ignored_frames
            
        self.tracked_radii=pd.read_excel('/scratch/lustre/home/tana9239/Radius/tracked_radii.xlsx', header=0)
        self.start_frames=start_frames
        self.end_frames=end_frames
        self.config=config

        self.center_evaluation=None
        self.radius_evaluation=None
    
    def evaluate_center(self):
        actual_coords=list(zip(self.true_df['x'],self.true_df['y']))
    
        actual_coords = actual_coords[self.start_frames:self.end_frames]
        
        predicted_coords = list(zip(self.results_df['x'],self.results_df['y']))

        filtered_actual,filtered_predicted=self.filter_out_frames(actual_coords,predicted_coords)
        
        distances = [
            calculate_distance(predicted, actual)
            for predicted, actual in zip(filtered_predicted, filtered_actual)
        ]


        actual_coords=np.array(actual_coords)
        actual_deltas=actual_coords[1:]-actual_coords[:-1]
        actual_movement = np.linalg.norm(actual_deltas, axis=1)
        # print(actual_movement)

        predicted_coords=np.array(predicted_coords)
        predicted_deltas=predicted_coords[1:]-predicted_coords[:-1]
        predicted_movement = np.linalg.norm(predicted_deltas, axis=1)

        diff_changes = [abs(a - b) for a, b in zip(actual_movement, predicted_movement)]
        
        df = pd.DataFrame(distances)
        self.center_evaluation = {
            'min_distance': float(df.min().iloc[0]) if len(df) > 0 else 0,
            'max_distance': float(df.max().iloc[0]) if len(df) > 0 else 0,
            '0.25': float(df.quantile(0.25).iloc[0]) if len(df) > 0 else 0,
            'median': float(df.quantile(0.5).iloc[0]) if len(df) > 0 else 0,
            '0.75': float(df.quantile(0.75).iloc[0]) if len(df) > 0 else 0,
            'std_dev': float(df.std().iloc[0]) if len(df) > 0 else 0,
            'mad': mad(np.array(df)) if len(df) else 0,

            'center_change_mean_diff': np.mean(diff_changes) if diff_changes else 0,
            'center_change_std_diff': np.std(diff_changes) if diff_changes else 0,
            'center_change_min_diff': np.min(diff_changes) if diff_changes else 0,
            'center_change_max_diff': np.max(diff_changes) if diff_changes else 0,
            'center_change_mad_diff': mad(np.array(diff_changes)) if diff_changes else 0
        }
        
        return self.center_evaluation

    def print_center_eval(self):
        print('-----------------------------------------------')
        print("center eval:")
        print('-----------------------------------------------') 
        print(f'video: {self.config["output_name"]}')
        print(f"mean = {self.center_evaluation['median']}")
        print(f"std_dev = {self.center_evaluation['std_dev']}")
        print(f"max_distance = {self.center_evaluation['max_distance']}")
        print(f"mad = {self.center_evaluation['mad']}")
    
        print(f"center_change_mean_diff = {self.center_evaluation['center_change_mean_diff']}")
        print(f"center_change_std_diff = {self.center_evaluation['center_change_std_diff']}")
        print(f"center_change_max_diff = {self.center_evaluation['center_change_max_diff']}")
        print(f"center_change_mad_diff = {self.center_evaluation['center_change_mad_diff']}")
        print('-----------------------------------------------')

    def evaluate_radius(self):
        actual_radii = self.tracked_radii[self.config['output_name']].tolist()
        
        actual_radii = actual_radii[self.start_frames:self.end_frames]
        predicted_radii = self.results_df['radius'].tolist()

        filtered_actual,filtered_predicted=self.filter_out_frames(actual_radii,predicted_radii)

        radius_differences = [abs(a-b) for a, b in zip(filtered_actual, filtered_predicted) if a != -1]
    
        changes_actual = [abs(j - i) for i, j in zip(filtered_actual[:-1], filtered_actual[1:]) if i != -1 and j != -1]
        changes_predicted = [abs(j - i) for i, j in zip(filtered_predicted[:-1], filtered_predicted[1:])]
        radius_changes_diff = [abs(a - b) for a, b in zip(changes_actual, changes_predicted)]

        self.radius_evaluation = {
            'radius_mean_diff': np.mean(radius_differences) if radius_differences else 0,
            'radius_std_diff': np.std(radius_differences) if radius_differences else 0,
            'radius_min_diff': np.min(radius_differences) if radius_differences else 0,
            'radius_max_diff': np.max(radius_differences) if radius_differences else 0,
            'radius_mad_diff': mad(np.array(radius_differences)) if radius_differences else 0,
    
            'radius_change_mean_diff': np.mean(radius_changes_diff) if radius_changes_diff else 0,
            'radius_change_std_diff': np.std(radius_changes_diff) if radius_changes_diff else 0,
            'radius_change_min_diff': np.min(radius_changes_diff) if radius_changes_diff else 0,
            'radius_change_max_diff': np.max(radius_changes_diff) if radius_changes_diff else 0,
            'radius_change_mad_diff': mad(np.array(radius_changes_diff)) if radius_changes_diff else 0
        }
        return self.radius_evaluation

    def print_radius_eval(self):
        print('-----------------------------------------------')
        print("radius eval:")
        print('-----------------------------------------------') 
        print(f'video: {self.config["output_name"]}')
        print(f"mean = {self.radius_evaluation['radius_mean_diff']}")
        print(f"std_dev = {self.radius_evaluation['radius_std_diff']}")
        print(f"max_distance = {self.radius_evaluation['radius_max_diff']}")
        print(f"mad = {self.radius_evaluation['radius_mad_diff']}")
    
        print(f"radius_change_mean_diff = {self.radius_evaluation['radius_change_mean_diff']}")
        print(f"radius_change_std_diff = {self.radius_evaluation['radius_change_std_diff']}")
        print(f"radius_change_max_diff = {self.radius_evaluation['radius_change_max_diff']}")
        print(f"radius_change_mad_diff = {self.radius_evaluation['radius_change_mad_diff']}")
        print('-----------------------------------------------')
        
    def filter_out_frames(self,list1,list2):
        filtered_actual=[]
        filtered_predicted=[]
        for i, (actual, predicted) in enumerate(zip(list1, list2)):
            frame_idx = self.start_frames + i
            if not should_ignore_frame(frame_idx, self.ignored_frames):
                filtered_actual.append(actual)
                filtered_predicted.append(predicted)
        return filtered_actual, filtered_predicted
        

    
        
        