import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev,BSpline
import numpy as np
import os
import matplotlib.patches as patches

def visualize_radius_track(image,crow,ccol,radius,prof,roi,prof_roi,tck,x_local,I_fit_local,local_start,local_end,output_name,frame):
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(prof)
    plt.axvline(radius)
    
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    patch = plt.Circle((crow, ccol), radius, fill=False, color='red')
    plt.gca().add_patch(patch)
    plt.title(f'Radius: {radius:.1f}')

    plt.subplot(1,3,3)
    plt.plot(range(roi[0], roi[1]), prof[roi[0]:roi[1]])
    
    
    x_full = np.arange(0, len(prof_roi), 0.01) 
    I_fit_full = splev(x_full, tck)
    plt.plot(x_full + roi[0], I_fit_full, color='red', alpha=0.3, label='Full fit')
    
    plt.plot(x_local + roi[0], I_fit_local, color='blue', linewidth=2, label='Local high-res')
    
    plt.axvspan(local_start + roi[0], local_end + roi[0], alpha=0.2, color='green', label='Search window')

    plt.axvline(radius)
    
    # plt.legend()
    
    save_dir = f'/scratch/lustre/home/tana9239/Radius/results_no_fast/{output_name}'
    save_path = os.path.join(save_dir, f"{frame}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def visualize_center_track(image,center_df):
    print(image.shape)
    print(center_df.iloc[0]['x'])
    plt.imshow(image)
    plt.axvline(center_df.iloc[0]['x'])
    plt.axhline(center_df.iloc[0]['y'])

    save_dir = f'/scratch/lustre/home/tana9239/ref_test'
    save_path = os.path.join(save_dir, f"test.png")
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path)