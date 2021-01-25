"""
Testing pipeline for the segmentation of DRM data by the LRC-MRM method.

Corresponding author:
Name:   Mallory Wittwer
Email:    mallory.wittwer@ntu.edu.sg
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lib.preprocessing import NMFDataCompressor
from lib.segmentation import (
    fit_lrc_model,
    lrc_mrm_segmentation,
    )
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max

def set_selector(data, eulers, cps, sample_size, max_spacing):
    
    global rx, ry, s0, s1
    global segmentation, gbs
    
    rx, ry, s0, s1 = data.shape
    
    def data_window_generator(data, eulers, rx, ry, window_size=(200,200)):
        window_size_x, window_size_y = window_size
        x_indeces = np.floor(rx//window_size_x)
        residual_x = rx - window_size_x*x_indeces
        y_indeces = np.floor(ry//window_size_y)
        residual_y = ry - window_size_y*y_indeces
        for kx in range(int(x_indeces)):
            xstart = kx*window_size_x
            xend = xstart + window_size_x if xstart + window_size_x <= rx else xstart + residual_x
            for ky in range(int(y_indeces)):
                ystart = ky*window_size_y
                yend = ystart + window_size_y if ystart + window_size_y <= ry else ystart + residual_y
                data_slice = data[xstart:xend, ystart:yend]
                eulers_slice = eulers[xstart:xend, ystart:yend]
                yield data_slice, eulers_slice
    
    xtr = np.empty((0,6,72))
    xte = np.empty((0,6,72))
    ytr = np.empty((0,3))
    yte = np.empty((0,3))
    
    for data_slice, eulers_slice in data_window_generator(data, eulers, rx, ry):
        
        rxx, ryy, *_ = data_slice.shape
        
        dataset_slice = {
            'data':data_slice.reshape((rxx*ryy, s0*s1)), 
            'eulers':eulers_slice.reshape((rxx*ryy, 3)),
            'spatial_resol':(rxx,ryy), 
            'angular_resol':(s0,s1),
            }
        
        compressor = NMFDataCompressor(cps)
        compressor.fit(dataset_slice, sample_size)
        compressed_dataset = compressor.transform(data_slice.reshape((rxx*ryy, s0*s1)))
        
        
        ### TEMP - SAVE DATA SLICE
        # np.save('C:/Users/mallo/Desktop/data_slice_inconel.npy', data_slice.reshape((rxx, ryy, s0, s1)))
        
        
        dataset_slice['data'] = compressed_dataset
        
        lrc_model, metrics = fit_lrc_model(
            dataset_slice,
            model=LogisticRegression(penalty='none'), 
            training_set_size=sample_size,
            test_set_size=sample_size,
        )
        dataset_slice = lrc_mrm_segmentation(dataset_slice, lrc_model)
        
        segmentation = dataset_slice.get('segmentation').reshape((rxx, ryy))
        
        gbs = dataset_slice.get('boundaries').reshape((rxx, ryy))
        distance = ndimage.distance_transform_edt(~gbs)
        local_maxi = peak_local_max(
            distance, indices=False, 
            labels=segmentation, 
            num_peaks_per_label=1,
            threshold_abs=max_spacing, # previously 2 for the training set
            )
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(eulers_slice)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(segmentation)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(gbs)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(distance)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(local_maxi)
        plt.show()
        
        
        ### TEMP - SAVE EULERS SLICE AND DATA SLICE
        # np.save('C:/Users/mallo/Desktop/eulers_slice_inconel.npz', eulers_slice)
        
        
        print(f'> Found {local_maxi.sum()} datapoints')
        
        if local_maxi.sum()>1:
            
            data_slice = data_slice[local_maxi]
            eulers_slice = eulers_slice[local_maxi]
            
            print('SELECTED: ', data_slice.shape, eulers_slice.shape)
            
            xtr_slice, xte_slice, ytr_slice, yte_slice = train_test_split(
                data_slice, 
                eulers_slice, 
                test_size=0.1,
                shuffle=True,
            )
            
            xtr = np.vstack((xtr, xtr_slice))
            xte = np.vstack((xte, xte_slice))
            ytr = np.vstack((ytr, ytr_slice))
            yte = np.vstack((yte, yte_slice))
            
            print('Training set: ', xtr.shape, xte.shape, ytr.shape, yte.shape)
            
            # break
            
    return xtr, ytr, xte, yte

def run_test(dataset, cps, sample_size, max_spacing):
    '''Runs a pipleine test segmentation based on specified folder'''
    # dataset = DataImporter(folder).load_npy_data()    
    data = dataset.get('data')
    eulers = dataset.get('eulers')
    rx, ry = dataset.get('spatial_resol')
    s0, s1 = dataset.get('angular_resol')
    print('Data: ', data.shape, eulers.shape)
    
    xtrs, ytrs, xtes, ytes = set_selector(
        data.reshape((rx,ry,s0,s1)),
        eulers.reshape((rx,ry,3)), 
        cps=cps, sample_size=sample_size, max_spacing=max_spacing)
    
    print('XTRS: ', xtrs.shape, ytrs.shape, xtes.shape, ytes.shape)

    return xtrs, ytrs, xtes, ytes

if __name__=='__main__': 
    
    root = 'C:/Users/mallo/Documents/Github/pydrm/data/inconel/'
    suffix_data = '_cropped_aligned-filled-v2.npy'
    suffix_eulers = '_eulers_cropped_aligned-filled.npy'
    
    # 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'X1', 'X1c', 'X1d', 'X2'
    names = ['R6', 'X1', 'X1d', 'X2']
    
    for name in names:
        
        data = np.load(root+name+suffix_data)
        rx, ry, s0, s1 = data.shape
        eulers = np.load(root+name+suffix_eulers)
        
        dataset = {
            'data':data,
            'eulers':eulers,
            'spatial_resol':(rx, ry),
            'angular_resol':(s0, s1),
        }
        
        xtrs, ytrs, xtes, ytes = run_test(
            dataset,
            cps=20,
            sample_size=2000,
            max_spacing = 8,
        )
        
        np.save(f'C:/Users/mallo/Desktop/r2test/test-filled-OFlow/{name}_xtrs.npy', xtrs)
        np.save(f'C:/Users/mallo/Desktop/r2test/test-filled-OFlow/{name}_xtes.npy', xtes)
        np.save(f'C:/Users/mallo/Desktop/r2test/test-filled-OFlow/{name}_ytrs.npy', ytrs)
        np.save(f'C:/Users/mallo/Desktop/r2test/test-filled-OFlow/{name}_ytes.npy', ytes)
    
