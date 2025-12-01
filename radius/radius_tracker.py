import os

import pandas as pd
import traceback


from radius.trends import compute_video_radial_profiles, detect_radius_trends, should_ignore_frame, get_current_trend

from utils.prepare_image import prepare_image

from radius.radius_fun import radius_interpolated

class RadiiTracker:
    def __init__(self,config,crop_top_left=None,crop_size=0):
        self.config=config
        
        if crop_top_left==None:
            self.crop_top_left=[0,0]
        else:
            self.crop_top_left=crop_top_left

        self.crop_size=crop_size
        self.video_path=""

        self.ignored_frames=[]
        self.start_frames=0
        self.end_frames=5000
        self.trend_segments=[]
        self.prev_prof=None
        self.prev_radius=None
        self.prev_peak=None
        self.results_list=[]

        self.radii_df=None

    def _load_metadata(self,video_path):
        excel_file = self.config['excel_file']
        self.metadata = pd.read_excel(excel_file, header=None).iloc[0]
        self.video_path = os.path.join(video_path, self.metadata[0])
    
    def run_tracker(self,centers,ignored_frames=None,frame_range=None):
        if frame_range!=None:
            self.start_frames=frame_range[0]
            self.end_frames=frame_range[1]
        if ignored_frames!=None:
            self.ignored_frames=ignored_frames

        center_data_slice = centers.iloc[self.start_frames:self.end_frames].reset_index(drop=True)
        radial_profiles, valid_frames = compute_video_radial_profiles(self.video_path, center_data_slice, self.start_frames, self.end_frames, self.ignored_frames)

        print("Detecting radius trends")
        self.trend_segments, peak_positions = detect_radius_trends(radial_profiles)

        print("Trends: ")
        for start, end, trend in self.trend_segments:
            print(f"  Frames {start}-{end}: {trend}")

        frame_files = sorted([f for f in os.listdir(self.video_path) if f.endswith(".png")])[self.start_frames:self.end_frames]

        for i, filename in enumerate(frame_files):
            self._process_frame(i,filename,center_data_slice)

        self.radii_df=pd.DataFrame({'frame_number':range(self.start_frames,min(self.end_frames,len(self.results_list))),
                                    'radius': self.results_list})
        
        self._post_process_results()

        self.radii_df=pd.merge(self.radii_df,centers,on="frame_number")
        self.radii_df.set_index('frame_number')

        return self.radii_df

    def _process_frame(self,i,filename,center_data_slice):
        # try:
        frame_global_idx=self.start_frames+i
        frame_local_idx=i

        if should_ignore_frame(frame_global_idx, self.ignored_frames):
            print(f"Skipping ignored frame {frame_global_idx}")
            return -1
        
        ccol = center_data_slice['y'].iloc[i]
        crow = center_data_slice['x'].iloc[i]

        if self.config['output_name']=='036_obj1':
            crow = center_data_slice['y'].iloc[i]
            ccol = center_data_slice['x'].iloc[i]

        current_trend = get_current_trend(frame_local_idx, self.trend_segments)

        print(f"{self.config['output_name']}, Frame {frame_global_idx}, Local Index {frame_local_idx}, Trend: {current_trend}")

        image=prepare_image(self.video_path,filename,self.crop_top_left,self.crop_size)

        radius,self.prev_prof,self.prev_peak=radius_interpolated(image,int(ccol),int(crow),self.config['output_name'],frame_global_idx,
                                                            self.prev_radius,self.trend_segments,self.prev_peak)
        
        self.results_list.append(radius)
        # except Exception as e:
        #     print(f'Failed for {filename}: {e}')
        #     self.results_list.append(-1)

        
    def _post_process_results(self):
        for i in range(1, len(self.radii_df) - 1):
            if self.radii_df.loc[i, 'radius'] == -1:
                prev_val = self.radii_df.loc[i - 1, 'radius']
                next_val = self.radii_df.loc[i + 1, 'radius']
                self.radii_df.loc[i, 'radius'] = (prev_val + next_val) / 2

            if abs(self.radii_df.loc[i-1,'radius'] - self.radii_df.loc[i+1,'radius']) < abs(self.radii_df.loc[i-1,'radius'] - self.radii_df.loc[i,'radius']):
                self.radii_df.loc[i,'radius']=(self.radii_df.loc[i-1,'radius']+self.radii_df.loc[i+1,'radius'])/2