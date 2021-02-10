import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

def fit_lrc_model(dataset, model, training_set_size, test_set_size,
                  with_regularization=False,
                  
                  # training_set = None,
                  
                  ):
    '''Fits a model, computes precision, recall and accuracy'''
    # Get training set by random sampling
    training_set = get_sample_set(dataset, training_set_size)
    
    # Get test set by random sampling
    test_set = get_sample_set(dataset, test_set_size)
    
    
    if with_regularization:
        strengths = [1e-3, 1e-2, 1e-1, 1.0, 10, 100]
        kf = KFold(n_splits=len(strengths))
        accs = []
        for k, (train_index, valid_index) in enumerate(kf.split(training_set['x'])):
            params = {'penalty':'l2', 'C':strengths[k]}
            model.set_params(**params)
            model.fit(training_set['x'][train_index], training_set['y'][train_index])
            valid_preds = model.predict(training_set['x'][valid_index])
            cm_valid = confusion_matrix(training_set['y'][valid_index], valid_preds)
            accs.append(_accuracy(cm_valid))
        best_strength = strengths[np.argmax(accs)]
        params = {'penalty':'l2', 'C':best_strength}
        model.set_params(**params)
        
        print('ACCS: ', accs)
        print('BEST STRENGTH: ', best_strength)
        
    # Fit the model
    model.fit(training_set['x'], training_set['y'])
    
    # Get training predictions
    train_preds = model.predict(training_set['x'])
    
    # Get test predictions
    test_preds = model.predict(test_set['x'])
    
    # Training confusion matrix
    cm_training = confusion_matrix(training_set['y'], train_preds)
    
    # Test confusion matrix
    cm_test = confusion_matrix(test_set['y'], test_preds)
    
    # Accuracy metrics
    metrics = {
        'training_accuracy':_accuracy(cm_training),
        'test_set_accuracy':_accuracy(cm_test),
    }
    
    dataset['lrc_model'] = model
    
    return dataset, metrics

def _shuffler(data, sample_size):
    '''
    Randomly selects a sample of sample_size from the data and returns it 
    with the corresponding indeces.
    '''
    idx = np.arange(data.shape[0])
    
    # np.random.seed(0)
    
    np.random.shuffle(idx)
    sample = data[idx[:sample_size]]
    indeces = idx[:sample_size]
    return sample, indeces

def _get_xymaps(rx, ry):
    '''
    Produces a flattened mesh grid of X and Y coordinates of resolution (rx,ry)
    '''
    # X-map
    xmap = np.empty((rx*ry))
    for k in range(rx):
        xmap[k*ry:(k+1)*ry] = np.array([k]*ry)
    xmap = xmap.reshape((rx, ry))
    xmap = xmap.ravel()
    # Y-map
    ymap = np.empty((rx*ry))
    for k in range(rx):
        ymap[k*ry:(k+1)*ry] = np.arange(ry)
    ymap = ymap.reshape((rx, ry))
    ymap = ymap.ravel()
    return xmap, ymap

def _get_adjacent_sample(rx, ry, data, sample_size, xmap, ymap):
    '''Samples Sbar, the distribution of adjacent pixel feature vectors.'''
    
    # Get set of random data
    X0, idx = _shuffler(data, sample_size)

    # Modify location by 1 pixel
    
    # np.random.seed(0)
    
    modifiedX = xmap[idx] + (np.random.randint(0,2,sample_size)*2-1)
    modifiedX = np.clip(modifiedX.astype('int'), 0, rx-1)
    modifiedY = ymap[idx] + (np.random.randint(0,2,sample_size)*2-1)
    modifiedY = np.clip(modifiedY.astype('int'), 0, ry-1)

    # Find corresponding signal
    X1 = np.empty_like(X0)
    c = 0
    i = np.arange(rx*ry)
    for xc, yc in zip(modifiedX, modifiedY):
        u = np.zeros((rx,ry))
        u[xc,yc] = 1
        num = i[(u.ravel()==1)]
        X1[c] = data[num]
        c += 1
    
    # Compute distance vectors
    Xclose = _vector_similarity(X1,X0)
    
    # Label as 0
    yclose = np.zeros(Xclose.shape[0], dtype=np.uint8)
    
    return Xclose, yclose

def _get_non_adjacent_sample(data, sample_size, xmap, ymap):
    '''Samples Dbar, the distribution of non-adjacent pixels.'''
    
    # Get random set of data and location of selected pixel pairs, twice
    X0, idx = _shuffler(data, sample_size)
    locX_0, locY_0 = xmap[idx], ymap[idx]
    X1, idx = _shuffler(data, sample_size)
    locX_1, locY_1 = xmap[idx], ymap[idx]
    
    # Compute distance vectors
    Xfar = _vector_similarity(X1,X0)
    
    # Filter out adjacent examples
    adjacent_filter = np.abs(locX_0-locX_1) + np.abs(locY_0-locY_1) < 2
    Xfar = Xfar[~adjacent_filter] 
    
    # Label as 1
    yfar = np.ones(Xfar.shape[0], dtype=np.uint8)
    
    return Xfar, yfar

def get_sample_set(dataset, sample_size):
    '''
    Randomly extracts a training or test set from the dataset.
    - Class 0: pairs of adjacent voxels
    - Class 1: pairs of non-adjacent voxels
    '''
    # Collect data from the dataset
    rx, ry = dataset.get('spatial_resol')
    data = dataset.get('data')
    xmap, ymap = _get_xymaps(rx, ry)
    
    # Extract adjacent sample
    Xclose, yclose = _get_adjacent_sample(
        rx, ry, data, sample_size, xmap, ymap)
    
    # Extract non-adjacent sample
    Xfar, yfar = _get_non_adjacent_sample(data, sample_size, xmap, ymap)

    # Stack both samples
    X = np.vstack((Xclose, Xfar)) 
    y = np.hstack((yclose, yfar))
    
    # Shuffle extracted set
    idx = np.arange(0, X.shape[0])
    
    # np.random.seed(0)
    
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    sample_set = {'x':X, 'y':y}
    
    return sample_set

def _vector_similarity(a, b):
    '''Returns distance vector of two input feature vectors'''
    return np.square(np.subtract(a,b))

def _accuracy(cm):
    '''Returns classification accuracy based on confusion matrix'''
    return (cm[0,0]+cm[1,1])/cm.sum()
