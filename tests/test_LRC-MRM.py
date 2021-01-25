"""
Testing pipeline for the segmentation of DRM data by the LRC-MRM method.

Corresponding author:
Name:   Mallory Wittwer
Email:    mallory.wittwer@ntu.edu.sg
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.linear_model import LogisticRegression
from lib.baseline import BaselineModel
from lib.preprocessing import DataImporter, NMFDataCompressor
from lib.segmentation import (
    fit_lrc_model,
    lrc_mrm_segmentation,
    compare_to_reference, 
    size_dist_plot,
    reconstruction_analysis,
    )

def timeit(method):
    '''Used to measure function running time.'''
    def timed(*args, **kwargs):
        print('\n> Starting: {} \n'.format(method.__name__))
        ts = perf_counter()
        result = method(*args, **kwargs)
        te = perf_counter()
        print('\n> Timer ({}): {:.2f} sec.'.format(method.__name__, te-ts))
        return result
    return timed

@timeit
def run_test(folder, cps, sample_size, spatial_resolution, with_baseline=True,
             with_regularization=False,
             ):
    '''
    Runs a pipleine test segmentation based on specified folder
    '''
    
    if type(cps)==list:
        
        accs = []
        miss = []
        
        for c in cps:
        
            ### IMPORT THE DATA TO A DATASET INSTANCE
            dataset = DataImporter(folder).load_npy_data()
            
            ### FIT NMF DIMENSIONALITY REDUCTION MODEL
            compressor = NMFDataCompressor(c) ### cps
            compressor.fit(dataset, sample_size)
            dataset['compressor'] = compressor
            
            ### REPLACE ORIGINAL DRM DATASET BY NMF FEATURE VECTOR DATASET
            compressed_dataset = compressor.transform(dataset.get('data'))
            dataset['data'] = compressed_dataset
            
            ### LRC MODEL TRAINING
            lrc_model, metrics = fit_lrc_model(
                dataset,
                model=LogisticRegression(penalty='none'), 
                training_set_size=sample_size,
                test_set_size=sample_size,
            )
            
            acc = metrics['test_set_accuracy']
            print(f'>>> Test set accuracy for {c} components: {acc}')
            accs.append(acc)
            
            ### OPTIONAL: SEGMENTI THE DOMAIN BY LRC-MRM METHOD
            dataset = lrc_mrm_segmentation(dataset, lrc_model)
            mutual_info_score = compare_to_reference(dataset)
            miss.append(mutual_info_score)
        
        fig, ax = plt.subplots(figsize=(8,8), dpi=200)
        ax.plot(cps, accs)
        ax.set_title('Proxy Accuracy')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(8,8), dpi=200)
        ax.plot(cps, miss)
        ax.set_title(f'MIS (best: {cps[np.argmax(miss)]})')
        plt.show()
        
        cps = cps[np.argmax(accs)]
        print('Finished NMF components testing. USING: ', cps)
        
    elif type(cps)==int:
        pass
    
    else:
        print('Wrong NMF components type --> Use an INT or list of INTS.')
    
    ### IMPORT THE DATA TO A DATASET INSTANCE
    dataset = DataImporter(folder).load_npy_data()
    
    ### FIT NMF DIMENSIONALITY REDUCTION MODEL
    compressor = NMFDataCompressor(cps)
    compressor.fit(dataset, sample_size)
    dataset['compressor'] = compressor
    
    ### REPLACE ORIGINAL DRM DATASET BY NMF FEATURE VECTOR DATASET
    compressed_dataset = compressor.transform(dataset.get('data'))
    dataset['data'] = compressed_dataset
    
    ### LRC MODEL TRAINING
    lrc_model, metrics = fit_lrc_model(
        dataset,
        # model=LogisticRegression(penalty='none'),
        model=LogisticRegression(penalty='l2'),
        training_set_size=sample_size,
        test_set_size=sample_size,
        
        with_regularization=with_regularization,
        
    )

    ### SEGMENTI THE DOMAIN BY LRC-MRM METHOD
    dataset = lrc_mrm_segmentation(dataset, lrc_model)
    
    ### COMPUTE SCORING AGAINST REFERENCE
    mutual_info_score = compare_to_reference(dataset)

    ### REPORT RESULTS
    print('''
---------------------------------------
          \n> Test set accuracy: {:.3f}
          \n> Mutual information score (MIS): {:.3f}
          \n---------------------------------------
          '''.format(
          metrics['test_set_accuracy'],
          mutual_info_score))
        
    ### PLOT GRAIN SIZE DISTRIBUTION
    dataset['um_per_px'] = spatial_resolution
    size_dist_plot(dataset)
    
    ### (OPTIONAL) BASELINE MODEL TRAINING (FOR BENCHMARKING)
    if with_baseline:
        baseline_model, baseline_metrics = fit_lrc_model(
            dataset,
            model=BaselineModel(), 
            training_set_size=sample_size,
            test_set_size=sample_size,
        )
        dataset = lrc_mrm_segmentation(dataset, baseline_model)
        baseline_mutual_info_score = compare_to_reference(dataset)
        print('''
---------------------------------------
              \n> BASELINE Test set accuracy: {:.3f}
              \n> BASELINE Mutual information score (MIS): {:.3f}
              \n---------------------------------------
              '''.format(
              baseline_metrics['test_set_accuracy'],
              baseline_mutual_info_score))
    
    return dataset#, metrics, mutual_info_score, baseline_metrics, baseline_mutual_info_score
    
if __name__=='__main__': 
    
    # Number of components in the NMF decomposition
    nmf_components = 20 # Note: 10 for inconel works best (lower angular resol)
    
    # Random sampling size in both NMF and LRC fitting
    sample_size = 10_000
    
    # ### RUN TEST ON NICKEL SAMPLE
    # dataset = run_test(
    #     # Relative path to data folder
    #     folder='../data/coin/try2/', 
    #     cps=nmf_components,
    #     sample_size=sample_size,
    #     # Resloution of the domain in um/px
    #     spatial_resolution=13.06,
    #     with_baseline=False,
    #     with_regularization=False,
    # )    
    
    # data = DataImporter('../data/coin/try2/').load_npy_data().get('data')
    # sizes, errs = reconstruction_analysis(dataset, data)   
    
    # dataset = run_test(
    #     # Relative path to data folder
    #     folder='../data/inconel_R4_subset/', 
    #     cps=nmf_components,
    #     sample_size=sample_size,
    #     # Resloution of the domain in um/px
    #     spatial_resolution=18.90,
    #     with_baseline=True,
    # )
    
    ### RUN TEST ON ALUMINIUM SAMPLE
    
    # dataset = run_test(
    #     folder='../data/aluminium/', 
    #     cps=nmf_components, 
    #     # cps=[5, 10, 20, 30, 40, 50, 70, 100],
    #     sample_size=sample_size,
    #     spatial_resolution=22.10,
    #     with_baseline=False,
    #     with_regularization=True,
    # )
    
    
    
    # ### Intra-granular feature vector variance (IGFV)
    # import pandas as pd
    # data = dataset.get('data') # (None, 20)
    # seg = dataset.get('segmentation') # (None,)
    # rx, ry = dataset.get('spatial_resol')
    # fv_mean = np.mean(data, 1)
    # data = np.vstack((seg, fv_mean)).T
    # df = pd.DataFrame(data=data, columns=['grain', 'fv_mean'])
    # df['grainID'] = df['grain'].astype('str')
    # groups = df.groupby('grainID').std().sort_values('grain', ascending=False)
    # groups = groups[~pd.isna(groups)] # does not work
    # im = np.empty((rx, ry), dtype=np.float64)
    # s = seg.reshape((rx, ry))
    # for grain_id, value in zip(groups.index, groups['fv_mean']):
    #     im[s==int(float(grain_id))] = value
    # fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    # ax.imshow(im[rx//4+50:rx//2+50,ry//4:ry//2])
    # ax.axis('off')
    # plt.show()
    
    
    
    # data = DataImporter('../data/aluminium/').load_npy_data().get('data')
    # reconstruction_analysis(dataset, data) 
    
    ### RUN TEST ON TITANIUM SAMPLE
    # accs, miss = [], []
    # baccs, bmiss = [], []
    for k in range(1):
        dataset = run_test(#, metrics, mis, bmetrics, bmis = run_test(
            folder='../data/titanium/', #'../data/titanium/', 
            cps=nmf_components, 
            sample_size=sample_size,
            spatial_resolution=18.33,
            with_baseline=False,
        )
        # accs.append(metrics['test_set_accuracy'])
        # miss.append(mis)
        # baccs.append(bmetrics['test_set_accuracy'])
        # bmiss.append(bmis)
        
    # import numpy as np
    
    # print('>>> Acc mean: ', np.mean(np.array(accs)))
    # print('>>> Acc std: ', np.std(np.array(accs)))
    # print('>>> MIS mean: ', np.mean(np.array(miss)))
    # print('>>> MIS std: ', np.std(np.array(miss)))
    
    # print('>>> bAcc mean: ', np.mean(np.array(baccs)))
    # print('>>> bAcc std: ', np.std(np.array(baccs)))
    # print('>>> bMIS mean: ', np.mean(np.array(bmiss)))
    # print('>>> bMIS std: ', np.std(np.array(bmiss)))
    
    data = DataImporter('../data/titanium/').load_npy_data().get('data')
    e, s = reconstruction_analysis(dataset, data)
