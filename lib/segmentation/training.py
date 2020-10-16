import numpy as np
from sklearn.metrics import confusion_matrix

def fit_lrc_model(dataset, model, training_set_size, test_set_size):
    '''Fits a model, computes precision, recall and accuracy'''
    training_set = _get_sample_set(dataset, training_set_size)

    xtr = training_set['x']
    ytr = training_set['y']
    
    model.fit(xtr, ytr)

    train_preds = model.predict(xtr)
    
    test_set  = _get_sample_set(dataset, test_set_size)
    xte = test_set['x']
    yte = test_set['y']
    test_preds = model.predict(xte)
    
    metrics = {
        'training_accuracy':_accuracy(confusion_matrix(ytr, train_preds)),
        'test_set_accuracy':_accuracy(confusion_matrix(yte, test_preds)),
    }
    
    return model, metrics

def _shuffler(data, sample_size):
    '''
    Randomly selects a sample of sample_size from the data and returns it 
    with the corresponding indeces.
    '''
    idx = np.arange(data.shape[0])
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
    Xclose = None
    yclose = None
        
    X0, idx = _shuffler(data, sample_size)

    # Modify location by 1 pixel
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

    xcl = _vector_similarity(X1,X0)
    if Xclose is None:
        Xclose = xcl.copy()
    else:
        Xclose = np.vstack((Xclose, xcl))
            
    yclose = np.zeros(Xclose.shape[0], dtype=np.uint8)
    
    return Xclose, yclose

def _get_non_adjacent_sample(data, sample_size, xmap, ymap):
    '''Samples Dbar, the distribution of non-adjacent pixels.'''
    Xfar = None
    yfar = None
    X0, idx = _shuffler(data, sample_size)
    # Get location of the selected pixel pairs
    locX_0 = xmap[idx]
    locY_0 = ymap[idx]
    X1, idx = _shuffler(data, sample_size)
    # Get location of the selected pixel pairs
    locX_1 = xmap[idx]
    locY_1 = ymap[idx]
    
    # Make sure that distance between pairs is > 1
    adjacent_filter = np.abs(locX_0-locX_1) + np.abs(locY_0-locY_1) < 2
    print(f'************ Filtered out {adjacent_filter.sum()} examples')

    xf = _vector_similarity(X1,X0)
    xf = xf[~adjacent_filter] # filter out adjacent examples (optional)
    if Xfar is None:
        Xfar = xf.copy()
    else:
        Xfar = np.vstack((Xfar, xf))
    yfar = np.ones(Xfar.shape[0], dtype=np.uint8)
    return Xfar, yfar

def _get_sample_set(dataset, sample_size):
    '''
    Extracts a training set from the data.
    - Class 0: pais of adjacent voxels
    - Class 1: random pairs of voxels
    '''
    rx, ry = dataset.get('spatial_resol')
    data = dataset.get('data')
    
    xmap, ymap = _get_xymaps(rx, ry)
    
    # ----------------- # Extract adjacent sample
    Xclose, yclose = _get_adjacent_sample(
        rx, ry, data, sample_size, xmap, ymap)
    
    # ----------------- # Extract random sample
    Xfar, yfar = _get_non_adjacent_sample(data, sample_size, xmap, ymap)

    # ----------------- # Stack both samples
    X = np.vstack((Xclose, Xfar)) 
    y = np.hstack((yclose, yfar))
    
    # Shuffle training set
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    training_set = {'x':X, 'y':y}
    
    return training_set

def _vector_similarity(a, b):
    '''Defines vector similarity'''
    return np.square(np.subtract(a,b))

def _accuracy(cm):
    '''Defines accuracy metric'''
    return (cm[0,0]+cm[1,1])/cm.sum()
