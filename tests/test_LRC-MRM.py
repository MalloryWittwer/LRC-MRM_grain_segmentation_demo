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
    compute_merging_fraction,
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
def run_test(sample, cps, sample_size, 
             with_baseline=True, 
             with_regularization=False,
             with_reconstruction_analysis=False,
             with_merging_fraction=False,
             ):
    '''
    Runs a pipleine test segmentation based on specified folder
    '''
    
    folder, spatial_resolution = sample
    
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
    dataset, metrics = fit_lrc_model(
        dataset,
        model=LogisticRegression(penalty='l2'),
        training_set_size=sample_size,
        test_set_size=sample_size,
        with_regularization=with_regularization,
    )
    
    ### SEGMENT THE DOMAIN BY LRC-MRM METHOD
    dataset = lrc_mrm_segmentation(dataset)
    
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
        baseline_dataset = dataset.copy()
        baseline_model, baseline_metrics = fit_lrc_model(
            baseline_dataset,
            model=BaselineModel(), 
            training_set_size=sample_size,
            test_set_size=sample_size,
        )
        baseline_dataset = lrc_mrm_segmentation(baseline_dataset)
        baseline_mutual_info_score = compare_to_reference(baseline_dataset)
        print('''
---------------------------------------
              \n> BASELINE Test set accuracy: {:.3f}
              \n> BASELINE Mutual information score (MIS): {:.3f}
              \n---------------------------------------
              '''.format(
              baseline_metrics['test_set_accuracy'],
              baseline_mutual_info_score))
    
    if with_reconstruction_analysis:
        # NMF reconstruction error analysis shown in the paper
        data = DataImporter(folder).load_npy_data().get('data')
        reconstruction_analysis(dataset, data) 
        
    if with_merging_fraction:
        # Generates the merging fraction bar plot shown in the paper.
        compute_merging_fraction(dataset, bin_width=10)
    
    return dataset, metrics

if __name__=='__main__': 
    
    # (Sensible default) Number of components in the NMF decomposition
    nmf_components = 20 # Note: 10 for i718 works better (lower angular resol)
    
    # Random sampling size in both NMF and LRC model fittings
    sample_size = 10_000
    
    # Relative path to folder and spatial resolutions in um/px, for each sample
    samples = {
        'Ti':('../data/titanium/', 18.33),
        'Al':('../data/aluminium/', 22.10),
        'I718':('../data/i718/', 18.90),
        'Ni':('../data/nickel/', 13.06),
    }
    
    ### Run test on specific sample (here, Ti)
    dataset, metrics = run_test(
            sample = samples.get('Ti'),
            cps=nmf_components,
            sample_size=sample_size,
            # Optional: benchmark against baseline model
            with_baseline=False, 
            # Optional: use cross-validation of log. regression L2 penalty
            with_regularization=False,
            # Optional: compute NMF reconstruction error as in the paper
            with_reconstruction_analysis=False,
            # Optional: compute merging fractions as in the paper
            with_merging_fraction=False,
        )
    
    ### Alternatively, run test on all samples at once
    # for key in samples.keys():
    #     dataset, metrics = run_test(
    #         sample = samples.get(key),
    #         cps=nmf_components,
    #         sample_size=sample_size,
    #         with_baseline=False,
    #         with_regularization=False,
    #         with_reconstruction_analysis=False,
    #         with_merging_fraction=False,
    #     )
    
    