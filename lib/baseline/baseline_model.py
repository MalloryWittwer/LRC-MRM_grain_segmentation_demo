import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BaselineModel():
    '''
    Baseline classifier model used for benchmarking LRC model.

    Training:
    1) Computes the mean of input distance component vectors in trainings set.
    2) Applies rescaling between 0 and 1 and computes the global mean.
    
    Predictions:
    1) Compute mean of input vector, apply rescaling factor from training step
    2) Classifies input as D when value is above global mean, otherwise S.
    '''
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, xtr, ytr):
        '''Fits the model. Input xtr is an array of distance vectors.'''
        mean_vector = np.mean(xtr,1).reshape(-1,1)
        mean_vector_scaled = self.scaler.fit_transform(mean_vector)
        self.global_mean = np.mean(mean_vector_scaled)
   
    def predict(self, xtr):
        '''Performs binary classification of the input distance vectors.'''
        probas = self._operations(xtr)
        preds = (probas<0.5).astype('int')
        return preds
    
    def predict_proba(self, xtr):
        '''Returns the classification probability.'''
        probas = self._operations(xtr)       
        probas = np.vstack((probas, 1-probas)).T
        return probas
    
    def _operations(self, xtr):
        mean_vector = np.mean(xtr, 1).reshape(-1,1)
        probas = np.squeeze(self.scaler.transform(mean_vector))
        probas += (0.5-self.global_mean)
        probas = 1-np.clip(probas, 0, 1)
        return probas