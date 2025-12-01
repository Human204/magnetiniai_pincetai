import os

import cv2

import numpy as np

import pandas as pd



from utils.prepare_image import prepare_image
from center.center_fun import ring_image
from center.post_processing import post_processing
from utils.visualizations import visualize_center_track

class CenterTracker:
    def __init__(self,config,crop_top_left=None,crop_size=0):
        self.config=config
        
        if crop_top_left==None:
            self.crop_top_left=[0,0]
        else:
            self.crop_top_left=crop_top_left

        self.crop_size=crop_size
        self.video_path=""

        self.start_frames=0
        self.end_frames=5000
        self.prev_radius=None
        self.predicted_df=None
        self.final_df=None

    def load_metadata(self,video_path):
        excel_file = self.config['excel_file']
        self.metadata = pd.read_excel(excel_file, header=None).iloc[0]
        self.video_path = os.path.join(video_path, self.metadata[0])

    def run_tracker(self,ignored_frames=None,frame_range=None):
        if frame_range!=None:
            self.start_frames=frame_range[0]
            self.end_frames=frame_range[1]
        if ignored_frames!=None:
            self.ignored_frames=ignored_frames

        rolling_window = 'mean' if self.config["rolling_window"] == 'mean' else 'median'
        blur_level=(self.config["blur_level1"],self.config["blur_level2"])
        excel_file=self.config['excel_file']

        ring_index_hough=None

        path='/scratch/lustre/home/tana9239/HyperparamOpt/'+self.metadata[0]
        print(path)
        
        df_list=[]
        for filename in sorted(os.listdir(path))[self.start_frames:self.end_frames]:
            df_list.append(self._process_frame(filename))

        self.predicted_df=pd.concat(df_list, ignore_index=True)
        self.final_df=post_processing(self.config['output_path'],self.predicted_df,2,self.config['movement_thresh'],
                                      self.start_frames,self.end_frames) 
        
        self.final_df['frame_number']=range(self.start_frames,min(self.end_frames,len(self.final_df['x'])))
        return self.final_df


    def _process_frame(self,filename):
        try:
            image=prepare_image(self.video_path,filename,self.crop_top_left,self.crop_size)

            self.prev_radius,center_coords=ring_image(image,int(self.config["r_in_center"]),int(self.config["r_out_center"]),
                                                      3,self.metadata[0],
                                                      filename,self.prev_radius,self.config['brightness'],self.config['clipLimit'],
                                                      (int(self.config['titleGridSize1']),int(self.config['titleGridSize2']))
                                                      ,self.config['dp'],self.config['param1'],self.config['param2'],
                                                      self.config['resolution'])

            visualize_center_track(image,center_coords)
            del image
            return center_coords

        except Exception as e:
            print(f"failed at {filename}: {e}")
            return pd.DataFrame({'x':[-1],'y':[-1]})