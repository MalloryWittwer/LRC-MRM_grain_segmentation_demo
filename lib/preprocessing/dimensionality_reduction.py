import numpy as np

class DataCompressor():
    def __init__(self, compressor):
        '''Supported compressors: NMF() and PCA() objects'''
        self.compressor = compressor

    def fit(self, dataset, sample_size):
        '''Samples the dataset and fits a compressor model'''
        # Get the data from dataset
        data = dataset.get('data')

        # Extract a random sample
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        data_extract = data[idx[:sample_size]]
        
        # Fit the compressor
        self.compressor.fit(data_extract)

    def transform(self, data):
        '''Compresses the data in the input dataset'''
        return self.compressor.transform(data)
        

    