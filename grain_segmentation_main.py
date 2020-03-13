"""
This program presents a fully-autonomous approach to grain segmentation in 
single-phase polycrystals. The program converts a four-dimensional reflectance 
dataset of shape (rx,ry,s0,s1) to a grain map of shape (rx,ry). This version 
includes comparison with a ground truth when provided.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import heapq
import tkinter as tk
from PIL import Image, ImageTk
import imp
imp.reload(tk)

from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from skimage.future.graph import RAG
from skimage.morphology import skeletonize

from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score as nmis
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

def timeit(method):
    '''
    Timer decorator - measures the running time of a specific function.
    '''
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('\n> Timer ({}): {:.2f} sec.'.format(method.__name__, te-ts))
        return result
    return timed

def get_xymaps():
    '''
    Utility - returns 2D maps of (X,Y) coordinates for pixel localization.
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

def show_rag_perso(segmentation, rag):
    '''
    - Visualization of master pixels on the segmented map
    - Color-codes grain boundaries based on edge weights
    '''
    xmap, ymap = get_xymaps()
    xmap = xmap.reshape((rx,ry)).astype('int')
    ymap = ymap.reshape((rx,ry)).astype('int')
    gbs_colored = np.zeros((rx, ry), dtype=np.float32)
    
    # Iterate over the edges
    for nd1, nd2, d in rag.edges(data=True):
        n1, n2 = rag.node[nd1], rag.node[nd2]
        index1 = segmentation[int(n1['xpos']), int(n1['ypos'])]
        index2 = segmentation[int(n2['xpos']), int(n2['ypos'])]
        filt = (segmentation==index1)
        # Color boundary pixels connecting the two nodes
        for xp1, yp1 in zip(xmap[filt], ymap[filt]):
            uniques = np.unique(segmentation[xp1-1:xp1+2, yp1-1:yp1+2])
            if (uniques.size > 1) & (index2 in uniques):
                gbs_colored[xp1, yp1] = d['heap item'][0]

    # Collect all node coordinates
    nodes_pos = np.array([[rag.node[n]['ypos'], 
                           rag.node[n]['xpos']] for n in rag.nodes()])
    
    # Plot a figure
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(gbs_colored, cmap=plt.cm.Blues)
    ax.scatter(nodes_pos[:,0], nodes_pos[:,1], c='red', s=80, marker='s')
    ax.axis('off')
    ax.set_title('EDGES WEIGHTS AND MASTER PIXELS')
    plt.tight_layout()
    plt.show()

def show_confusion_matrix(cm, sc):
    '''
    Visualization of the confusion matrix and accuracy score.
    '''    
    fts = 20 # fontsize
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_title('Accuracy score: {:.3f}'.format(sc), fontsize=fts)
    ax.set_xlabel('Predicted', fontsize=fts)
    ax.set_ylabel('True', fontsize=fts)
    ax.tick_params(axis='both', labelsize=fts)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > (cm.max()/2) else 'black',
                    fontsize=fts)
    plt.tight_layout()
    plt.show()

def show_gradient_map():
    '''
    Visualization of approximate local intensity gradient.
    '''
    dd = data.reshape((rx,ry,data.shape[1]))
    dd = np.transpose(dd, [2,0,1])
    
    func = lambda x:x.reshape(x.shape[0], x.shape[1]*x.shape[2]).T
    
    diag1 = func(dd[:,:-1,:-1])
    diag2 = func(dd[:,1:,1:])
    diag3 = func(dd[:,1:,:-1])
    diag4 = func(dd[:,:-1,1:])

    edges = np.max((
            model.predict_proba(np.abs(diag1-diag2))[:,1], 
            model.predict_proba(np.abs(diag3-diag4))[:,1],
            ), axis=0).reshape((rx-1, ry-1))

    # Plot a figure
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(-edges, cmap=plt.cm.gray)
    ax.axis('off')
    ax.set_title('MODEL PROBABILITY GRADIENT MAP')
    plt.show()

def open_data(folder_name, with_eulers=True):
    '''
    Fetches the data in prepared folders:
        - 'data': required, reflectance dataset
        - 'labels': optional, ground truth segmentation
        - 'eulers': optional, Euler angles map from EBSD
    '''
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR, folder_name)
    try:
        data = np.load(os.path.join(path, 'data.npy')).astype('float')
        rx, ry, s0, s1 = data.shape
        shape = (s0, s1)
    except:
        print('''
              \n> Could not open the data...\n
              - Please check that data.npy is available in specified folder!
              ''')
    try:
        labels = np.load(os.path.join(path, 'labels.npy')) 
    except:
        labels = None
    try:
        eulers = np.load(os.path.join(path, 'eulers.npy')) 
    except:
        eulers = None

    return data, shape, s0, s1, rx, ry, labels, eulers

@timeit
def reduce_dimensions(data, sampling_fraction, components):
    '''
    Applies dimensionality reduction to the dataset using NMF.
    '''
    data_reshaped = data.reshape((rx*ry,s0*s1))
    
    # Extract a random sample
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    sample_size = np.ceil(rx*ry*sampling_fraction).astype('int')   
    data_comp = data_reshaped[idx][:sample_size]
    
    # Fit NMF on the sample, tansform the whole dataset
    compressor = NMF(components)
    compressor.fit(data_comp)
    data_transformed = compressor.transform(data_reshaped)
    
    return data_transformed

def get_training_set(sample_size):
    '''
    Extracts a training set from the data.
    - Class 0: pais of adjacent voxels
    - Class 1: random pairs of voxels
    '''
    xmap, ymap = get_xymaps()
    
    # Extract random sample
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:sample_size]
    X0 = data[idx]
    
    # Modify location by 1 pixel
    modifiedX = xmap[idx] + (np.random.randint(0,2,sample_size)*2-1)
    modifiedX = np.clip(modifiedX.astype('int'), 0, rx-1)
    modifiedY = ymap[idx] + (np.random.randint(0,2,sample_size)*2-1)
    modifiedY = np.clip(modifiedY.astype('int'), 0, ry-1)
    
    # Find the corresponding signal
    c = 0
    i = np.arange(rx*ry)
    X1 = np.empty_like(X0)
    for xc, yc in zip(modifiedX, modifiedY):
        u = np.zeros((rx,ry))
        u[xc,yc] = 1
        num = i[(u.ravel()==1)]
        X1[c] = data[num]
        c += 1
    
    # Get close examples, class 0
    Xclose = np.abs(X1-X0)
    yclose = np.zeros(sample_size)

    # Extract random sample
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:sample_size]
    X2 = data[idx]
    
    # Get far examples, class 1
    Xfar = np.abs(X2-X0)
    yfar = np.ones(Xfar.shape[0])

    X = np.concatenate((Xclose, Xfar), axis=0).astype('float')
    y = np.concatenate((yclose, yfar), axis=0).astype('int')
    
    # Shuffle training set
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    return X, y

@timeit
def train_model(model=LogisticRegression(), sampling_fraction=0.5):
    '''
    Extracts a training set and trains a logistic regression classifier.
    '''
    # Create training set
    xtr, ytrc = get_training_set(
            sample_size=np.ceil(rx*ry*sampling_fraction).astype('int')  
            )
    
    # Fit the model
    model.fit(xtr, ytrc)

    # Visualization
    show_confusion_matrix(
            cm=confusion_matrix(ytrc, model.predict(xtr)),
            sc=model.score(xtr, ytrc),
            ) 
    
    return model

#def merge_decision_function(inp):     
#    return model.predict_proba(np.atleast_2d(inp))[0][1]

@timeit
def hierarchical_merging(verbose=False):
    '''
    Inspired by skimage.future.graph.merge_hierarchical function of Skimage.
    (https://github.com/scikit-image/scikit-image/blob/master/skimage/)
    Performs region-merging segmentation using previously trained classifier
    model to define edge weights. Details in the publication.
    '''
    # Merging decision function
    mdf = lambda vect:model.predict_proba(np.atleast_2d(vect))[0,1]
    
    # Initialize Region Adjacency Graph (RAG)
    xmap, ymap = get_xymaps()
    segments = np.arange(rx*ry).reshape((rx,ry))
    rag = RAG(segments, connectivity=1)
    
    # Initialize nodes
    data_reshaped = data.reshape((rx,ry,data.shape[1]))
    for n in rag:
        rag.nodes[n].update({'labels': [n]})
    for index in np.ndindex(segments.shape):
        current = segments[index]
        rag.nodes[current]['count'] = 1
        rag.nodes[current]['master'] = data_reshaped[index]
        rag.nodes[current]['xpos'] = xmap[current]
        rag.nodes[current]['ypos'] = ymap[current]
    
    # Initialize edges
    edge_heap = []
    for n1, n2, d in rag.edges(data=True):
        master_x = rag.nodes[n1]['master']
        master_y = rag.nodes[n2]['master']
        weight = mdf(np.abs(master_x-master_y))        
        # Push the edge in the heap
        heap_item = [weight, n1, n2, (weight < 0.5)]
        d['heap item'] = heap_item
        heapq.heappush(edge_heap, heap_item)

    if verbose:
        counter = 0
        root = tk.Tk()
        root.geometry('500x500')
        root.title('Live segmentation')
        panel = tk.Label(root)
        panel.pack()
    
    # Start the region-merging algorithm
    while (len(edge_heap) > 0) and (edge_heap[0][0] < 0.5):
        # Pop the smallest edge if weight < 0.5
        smallest_weight, n1, n2, valid = heapq.heappop(edge_heap)
        # Check that the edge is valid
        if valid:
            # Make sure that n1 is the smallest regiom
            if (rag.nodes[n1]['count'] > rag.nodes[n2]['count']):
                n1, n2 = n2, n1
            # Update properties of n2
            rag.nodes[n2]['labels'] = (rag.nodes[n1]['labels'] 
                                       + rag.nodes[n2]['labels'])
            rag.nodes[n2]['count'] = (rag.nodes[n1]['count'] 
                                       + rag.nodes[n2]['count'])
            # Get new neighbors of n2
            n1_nbrs = set(rag.neighbors(n1))
            n2_nbrs = set(rag.neighbors(n2))
            new_neighbors = (n1_nbrs | n2_nbrs) - n2_nbrs - {n1, n2}
            
            # Disable edges of n1 in the heap list
            for nbr in rag.neighbors(n1):
                edge = rag[n1][nbr]
                edge['heap item'][3] = False
            
            # Remove n1 from the graph (edges are still in the heap list)
            rag.remove_node(n1)
            
            # Update new edges of n2
            for nbr in new_neighbors:
                rag.add_edge(n2, nbr)
                edge = rag[n2][nbr]
                master_n2 = rag.nodes[n2]['master']
                master_nbr = rag.nodes[nbr]['master']
                weight = mdf(np.abs(master_n2-master_nbr))
                heap_item = [weight, n2, nbr, (weight < 0.5)]
                edge['heap item'] = heap_item
                # Push edges to the heap
                heapq.heappush(edge_heap, heap_item)
        
            if verbose:
                if counter%1000==0:    
                    label_map = np.arange(segments.max() + 1)
                    for ix, (n, d) in enumerate(rag.nodes(data=True)):
                        for lab in d['labels']:
                            label_map[lab] = ix
        
                    root.title('Live Segmentation (GRAINS: {})'.format(ix))
                            
                    image = label_map[segments]
                    gbs = skeletonize(find_boundaries(image, mode='inner'))
                    image = plt.cm.gray(image)[:,:,:3]
                    image[gbs] = (1,0,0)
                    image = image-image.min()
                    image = image/image.max()
                    image *= 255
                    image = image.astype(np.uint8)
                    image = Image.fromarray(image, mode='RGB')
                    
                    resx, resy = gbs.shape
                    if resx<=resy:
                        w = int(500/resy*resx)
                        image = image.resize((w,500))
                    else:
                        h = int(500/resx*resy)
                        image = image.resize((500,h)) 
                        
                    image = ImageTk.PhotoImage(image)
                    panel.configure(image=image)
                    panel.image = image
                counter += 1
                
        if verbose:
            root.update()
            
    if verbose:
        root.destroy()
    
    # Compute grain map
    label_map = np.arange(segments.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for lab in d['labels']:
            label_map[lab] = ix
    segmentation = label_map[segments]
    
    if verbose:
        gbs = skeletonize(find_boundaries(segmentation, mode='inner'))
        im = label2rgb(segmentation, label2rgb(segmentation), kind='avg')
        im[gbs] = (1,1,1)
        
        fig, ax = plt.subplots(figsize=(12,12))
        ax.imshow(im)
        ax.axis('off')
        ax.set_title('SEGMENTATION')
        plt.show()
        
        show_rag_perso(segmentation, rag)
    
    return segmentation, rag

def compare_to_reference(segmentation, reference, eulers):
    '''
    Optional function called when reference segmentation is provided.
    '''    
    # Compute grain boundaries on segmentation and reference
    gbsseg = skeletonize(find_boundaries(segmentation, mode='inner'))
    gbsref = skeletonize(find_boundaries(reference, mode='inner'))
    
    # Assign region color based on Euler angles and mark boundaries
    lab = label2rgb(segmentation, eulers, kind='avg')
    wrgbseg = label2rgb(segmentation, lab, kind='avg')
    wrgbseg[gbsseg] = (1,1,1)
    wrgbref = label2rgb(reference, lab, kind='avg')
    wrgbref[gbsref] = (1,1,1)
    
    # Plot a figure
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(121)
    ax.imshow(wrgbseg)
    ax.axis('off')
    ax.set_title('SEGMENTATION')
    ax = fig.add_subplot(122)
    ax.imshow(wrgbref)
    ax.axis('off')
    ax.set_title('REFERENCE')
    plt.tight_layout()
    plt.show()
    
    # Mutual information score between segmentation and reference
    score = nmis(segmentation.ravel(), reference.ravel())
    print('-------------------------------------------')
    print('\n> Mutual info score: {:.3f}\n'.format(score))
    print('-------------------------------------------')

def segmentation_pipeline(folder_name):
    '''
    Master function calling all steps of the segmentation workflow in order.
    '''
    global data, shape, s0, s1, rx, ry, labels, eulers, model, rag, segmentation
    
    # DATA IMPORTATION
    data, shape, s0, s1, rx, ry, labels, eulers = open_data(folder_name)
    # DIMENSIONALITY REDUCTION
    data = reduce_dimensions(data, sampling_fraction=0.5, components=20)
    
    # MODEL TRAINING
    model = train_model(LogisticRegression(C=1), sampling_fraction=0.5)
    
    # PROBABILITY GRADIENT MAP
    show_gradient_map()
    
    # REGION-MERGING SEGMENTATION
    segmentation, rag = hierarchical_merging(verbose=True)
    
    # COMPARISON WITH REFERENCE
    if (eulers is not None) & (labels is not None):
        compare_to_reference(segmentation, labels, eulers)
    
if __name__=='__main__':
    segmentation_pipeline(folder_name='data_nickel')
    