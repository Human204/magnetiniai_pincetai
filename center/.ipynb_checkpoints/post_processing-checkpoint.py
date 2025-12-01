import os

import cv2

import numpy as np


import pandas as pd

def interpolation(data,size=50):
    return(np.interp([i for i in range(1,len(data)*size)],xp=[i*size for i in range(0,len(data))],fp=data))

def post_processing(output_path,predicted_df,window,movement_thresh,start_frames,end_frames):
    size=2
    if size>window:
        size=window
    movement_threshold=movement_thresh*len(os.listdir(output_path))
    # Path(result_folder).mkdir(parents=True, exist_ok=True)

    preProcData=predicted_df
    center_x_interpolated=interpolation(preProcData['x'],size)
    center_y_interpolated=interpolation(preProcData['y'],size)
    window=window

    dfList=list(zip(center_x_interpolated,center_y_interpolated))
    preProcData=pd.DataFrame(dfList,columns=['x','y'])

    preProcData['diff_center_y'] = preProcData['y'].diff().abs().fillna(0)
    preProcData['diff_center_x'] = preProcData['x'].diff().abs().fillna(0)
    totalMovement=0

    for index, row in preProcData.iterrows():
        totalMovement=totalMovement+row['diff_center_x']+row['diff_center_y']
    
    preProcData['smooth_y'] = preProcData['y'].rolling(window=window).median()
    preProcData['smooth_x'] = preProcData['x'].rolling(window=window).median()

    preProcData.fillna(method='bfill', inplace=True) 
    df = pd.DataFrame(columns=['y', 'x'])

    totalMovement = preProcData['diff_center_x'].sum() + preProcData['diff_center_y'].sum()

    df_list=[]
    count=0
    
    if totalMovement < movement_threshold:
        files = sorted(os.listdir(output_path))[start_frames:end_frames]
        
        for i, filename in enumerate(files):
            process_image_low_movement(filename, output_path, preProcData, i,df_list,size)
    else:
        files = sorted(os.listdir(output_path))[start_frames:end_frames]
        for i, filename in enumerate(files):
            process_image_high_movement(filename, output_path, preProcData, i,df_list,size)

    df = pd.concat(df_list, ignore_index=True)

    return df

def process_image_high_movement(filename, output_path, preProcData, i,df_list,size=50):
    i=i*size
    file_path = os.path.join(output_path, filename)
    
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    ccol=preProcData['x'][i]
    crow=preProcData['y'][i]

    new_data = pd.DataFrame({
        'x': [ccol],
        'y': [crow]
    })
    df_list.append(new_data)

def process_image_low_movement(filename, output_path, preProcData, i,df_list,size=50):
    i=i*size
    file_path = os.path.join(output_path, filename)

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


    movementNearFramesX=preProcData['diff_center_x'][i:i+100].sum()
    movementNearFramesY=preProcData['diff_center_y'][i:i+100].sum()

    if((movementNearFramesX+movementNearFramesY)>10):
        ccol=preProcData['x'][i]
        crow=preProcData['y'][i]
    else:
        ccol=preProcData['smooth_x'][i]
        crow=preProcData['smooth_y'][i]


    new_data = pd.DataFrame({
        'x': [ccol],
        'y': [crow]
    })
    df_list.append(new_data)