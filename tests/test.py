"""
Testing of LRC-MRM grain segmentation package on nickel and aluminium datasets.
"""

from time import perf_counter
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from lib.preprocessing import DataImporter, DataCompressor
from lib.segmentation import (
    fit_lrc_model,
    lrc_mrm_segmentation,
    compare_to_reference, 
    size_dist_plot,
    )

def timeit(method):
    '''
    Timer decorator - measures the running time of a specific function.
    '''
    def timed(*args, **kw):
        print('\n> Starting: {} \n'.format(method.__name__))
        ts = perf_counter()
        result = method(*args, **kw)
        te = perf_counter()
        print('\n> Timer ({}): {:.2f} sec.'.format(method.__name__, te-ts))
        return result
    return timed

@timeit
def run_test(folder, cps, sample_size, spatial_resolution):
    '''Runs a pipleine test segmentation based on specified folder'''
    ### IMPORTING THE DATA TO A DATASET INSTANCE
    dataset = DataImporter(folder).load_npy_data()

    ### DIMENSIONALITY REDUCTION BY NMF
    compressor = DataCompressor(NMF(cps, max_iter=1000))
    compressor.fit(dataset, sample_size)
    dataset['data'] = compressor.transform(dataset.get('data'))
    
    ### TRAINING A MODEL
    lrc_model, metrics = fit_lrc_model(
        dataset,
        model=LogisticRegression(penalty='none'), 
        training_set_size=sample_size,
        test_set_size=sample_size,
    )

    ### SEGMENTING THE DATASET
    dataset = lrc_mrm_segmentation(dataset, lrc_model)
    
    ### SCORING AGAINST REFERENCE MAP
    mutual_info_score = compare_to_reference(dataset)
    
    print('''
---------------------------------------
          \n> Test set accuracy: {:.3f}
          \n> Mutual information score (MIS): {:.3f}
          \n---------------------------------------
          '''.format(
          metrics['test_set_accuracy'],
          mutual_info_score))
        
    ### GRAIN SIZE DISTRIBUTION
    dataset['um_per_px'] = spatial_resolution
    size_dist_plot(dataset)

if __name__=='__main__': 
    
    nmf_components = 5
    sample_size = 10_000
    
    ### NICKEL SAMPLE
    run_test(
        # Data folder.
        folder='../data/nickel/', 
        # Number of components in the NMF decomposition. 
        cps=nmf_components,
        # Random sampling size in the NMF and LRC fitting.
        sample_size=sample_size,
        # Number of um per pixels.
        spatial_resolution=26.12,
    )
    
    ### ALUMINIUM SAMPLE
    run_test(
        folder='../data/aluminium/', 
        cps=nmf_components, 
        sample_size=sample_size,
        spatial_resolution=22.10,
    )
    
    # ### TITANIUM SAMPLE
    run_test(
        folder='../data/titanium/', 
        cps=nmf_components, 
        sample_size=sample_size,
        spatial_resolution=18.33,
    )
    
