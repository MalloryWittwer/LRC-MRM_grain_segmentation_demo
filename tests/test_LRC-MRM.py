"""
Testing pipeline for the segmentation of DRM data by the LRC-MRM method.

Corresponding author:
Name:   Mallory Wittwer
Email:    mallory.wittwer@ntu.edu.sg
"""

from time import perf_counter
from sklearn.linear_model import LogisticRegression
from lib.baseline import BaselineModel
from lib.preprocessing import DataImporter, NMFDataCompressor
from lib.segmentation import (
    fit_lrc_model,
    lrc_mrm_segmentation,
    compare_to_reference, 
    size_dist_plot,
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
def run_test(folder, cps, sample_size, spatial_resolution, with_baseline=True):
    '''Runs a pipleine test segmentation based on specified folder'''
    ### IMPORT THE DATA TO A DATASET INSTANCE
    dataset = DataImporter(folder).load_npy_data()

    ### FIT NMF DIMENSIONALITY REDUCTION MODEL
    compressor = NMFDataCompressor(cps)
    compressor.fit(dataset, sample_size)
    
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
    
if __name__=='__main__': 
    
    # Number of components in the NMF decomposition
    nmf_components = 20
    
    # Random sampling size in both NMF and LRC fitting
    sample_size = 10_000
    
    ### RUN TEST ON NICKEL SAMPLE
    run_test(
        # Relative path to data folder
        folder='../data/nickel/', 
        cps=nmf_components,
        sample_size=sample_size,
        # Resloution of the domain in um/px
        spatial_resolution=26.12,
    )
    
    ### RUN TEST ON ALUMINIUM SAMPLE
    run_test(
        folder='../data/aluminium/', 
        cps=nmf_components, 
        sample_size=sample_size,
        spatial_resolution=22.10,
    )
    
    ### RUN TEST ON TITANIUM SAMPLE
    run_test(
        folder='../data/titanium/', 
        cps=nmf_components, 
        sample_size=sample_size,
        spatial_resolution=18.33,
    )
    
