# -*- coding: utf-8 -*-
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
from sklearn.metrics import normalized_mutual_info_score

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('\n> Timer ({}): {:.2f} sec.'.format(method.__name__, te-ts))
        return result
    return timed

def open_data(folder_name):
    
    ROOT_DIR = 'C:/Users/mallo/Spyder_work/GrainSeg/Grain_Segmentation_github/'
    
    path = os.path.join(ROOT_DIR, folder_name)
    
    data = np.load(os.path.join(path, 'data.npy')).astype('float')
    rx, ry, s0, s1 = data.shape
    shape = (s0, s1)
    labels = np.load(os.path.join(path, 'labels.npy'))
    gbs = np.load(os.path.join(path, 'gbs.npy'))
    eulers = np.load(os.path.join(path, 'eulers.npy'))
    
    return data, shape, s0, s1, rx, ry, labels, gbs, eulers

@timeit
def reduce_dimensions(data, sampling_fraction, components):
    data_reshaped = data.reshape((rx*ry,s0*s1))
    data_reshaped = data_reshaped/255.0
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    sample_size = np.ceil(rx*ry*sampling_fraction).astype('int')   
    data_comp = data_reshaped[idx][:sample_size]
    compressor = NMF(components)
    compressor.fit(data_comp)
    data_transformed = compressor.transform(data_reshaped)
    
    return data_transformed

def get_training_batch(sample_size):
    
    # Compute X-map
    xmap = np.empty((rx*ry))
    for k in range(rx):
        xmap[k*ry:(k+1)*ry] = np.array([k]*ry)
    xmap = xmap.reshape((rx, ry))
    xmap = xmap.ravel()
    
    # Compute Y-map
    ymap = np.empty((rx*ry))
    for k in range(rx):
        ymap[k*ry:(k+1)*ry] = np.arange(ry)
    ymap = ymap.reshape((rx, ry))
    ymap = ymap.ravel()
    
    # Extract random sample
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:sample_size]
    X0 = data[idx]
    
    # Modify location by 1 pixel
    modifiedX = (xmap[idx] + (np.random.randint(0,2,sample_size)*2-1)).astype('int')
    modifiedX = np.clip(modifiedX, 0, rx-1)

    modifiedY = (ymap[idx] + (np.random.randint(0,2,sample_size)*2-1)).astype('int')
    modifiedY = np.clip(modifiedY, 0, ry-1)
    
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
    
    # Get close examples, label 0
    Xclose = np.abs(X1-X0) #np.concatenate((X0, X1), axis=1)
    yclose = np.zeros(sample_size)

    # Extract random sample
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:sample_size]
    X2 = data[idx]
    
    # Get far examples, label 1
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
def train_model(model, sampling_fraction):
    sample_size = np.ceil(rx*ry*sampling_fraction).astype('int')   
    xtr, ytrc = get_training_batch(sample_size)
    model.fit(xtr, ytrc)
    
    score = model.score(xtr, ytrc)
    
    print('\n- Accuracy score: %f\n' % score)
    
    return model

def merge_decision_function(inp):     
    return model.predict_proba(np.atleast_2d(inp))[0][1]

@timeit
def hierarchical_merging(verbose=False):
    
    global data, rx, ry
    
    segments = np.arange(rx*ry).reshape((rx,ry))
    
    rag = RAG(segments, connectivity=1)
    
    for n in rag:
        rag.nodes[n].update({'labels': [n]})
    
    data_reshaped = data.reshape((rx,ry,data.shape[1]))
    for index in np.ndindex(segments.shape):
        current = segments[index]
        rag.nodes[current]['count'] = 1
        rag.nodes[current]['master'] = data_reshaped[index]
    
    edge_heap = []
    for n1, n2, d in rag.edges(data=True):
        
        master_x = rag.nodes[n1]['master']
        master_y = rag.nodes[n2]['master']
        
        weight = merge_decision_function(np.abs(master_x-master_y))
        
        valid_state = (weight < 0.5)
        
        heap_item = [weight, n1, n2, valid_state]
        d['heap item'] = heap_item
        
        heapq.heappush(edge_heap, heap_item)
    
    # -------------------------------------------------------------------------
    if verbose:
        counter = 0
        root = tk.Tk()
        root.geometry('500x500')
        root.title('Live segmentation')
        panel = tk.Label(root)
        panel.pack()
    # -------------------------------------------------------------------------
    
    while (len(edge_heap) > 0) and (edge_heap[0][0] < 0.5):
        
        smallest_weight, n1, n2, valid = heapq.heappop(edge_heap)

        if valid:
            
            if (rag.nodes[n1]['count'] > rag.nodes[n2]['count']):
                n1, n2 = n2, n1

            rag.nodes[n2]['labels'] = rag.nodes[n1]['labels'] + rag.nodes[n2]['labels']
            rag.nodes[n2]['count'] = rag.nodes[n1]['count'] + rag.nodes[n2]['count']
            
            n1_nbrs = set(rag.neighbors(n1))
            n2_nbrs = set(rag.neighbors(n2))
            new_neighbors = (n1_nbrs | n2_nbrs) - n2_nbrs - {n1, n2}
            
            for nbr in rag.neighbors(n1):
                edge = rag[n1][nbr]
                edge['heap item'][3] = False
            
            rag.remove_node(n1)

            for nbr in new_neighbors:
                rag.add_edge(n2, nbr)
                edge = rag[n2][nbr]
                master_n2 = rag.nodes[n2]['master']
                master_nbr = rag.nodes[nbr]['master']
                weight = merge_decision_function(np.abs(master_n2-master_nbr))
                valid_state = (weight < 0.5)
                heap_item = [weight, n2, nbr, valid_state]
                edge['heap item'] = heap_item
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
                    
                    rx, ry = gbs.shape
                    if rx<=ry:
                        w = int(500/ry*rx)
                        image = image.resize((w,500))
                    else:
                        h = int(500/rx*ry)
                        image = image.resize((500,h)) 
                        
                    image = ImageTk.PhotoImage(image)
                    panel.configure(image=image)
                    panel.image = image
                counter += 1
                
        if verbose:
            root.update()
            
    if verbose:
        root.destroy()
    
    label_map = np.arange(segments.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for lab in d['labels']:
            label_map[lab] = ix
    
    segmentation = label_map[segments]
    
    if verbose:
        gbs = skeletonize(find_boundaries(segmentation, mode='inner'))
        im = label2rgb(segmentation, label2rgb(segmentation), kind='avg')
        im[gbs] = (1,1,1)
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(im)
        ax.axis('off')
        ax.set_title('SEGMENTATION')
        plt.show()
    
    return segmentation

def compare_to_reference(segmentation, reference, eulers):
    gbsseg = skeletonize(find_boundaries(segmentation, mode='inner'))
    gbsref = skeletonize(find_boundaries(reference, mode='inner'))
    
    lab = label2rgb(segmentation, eulers, kind='avg')
    
    wrgbseg = label2rgb(segmentation, lab, kind='avg')
    wrgbseg[gbsseg] = (1,1,1)
    
    wrgbref = label2rgb(reference, lab, kind='avg')
    wrgbref[gbsref] = (1,1,1)

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

    score = normalized_mutual_info_score(segmentation.ravel(), reference.ravel())
    
    print('-------------------------------------------')
    print('\n> Mutual info score: {:.3f}\n'.format(score))
    print('-------------------------------------------')

if __name__=='__main__':
    
    # DATA IMPORTATION
    data, shape, s0, s1, rx, ry, labels, gbs, eulers = open_data(
#            'data_nickel'
            'data_aluminium'
            )
    
    # DIMENSIONALITY REDUCTION
    data = reduce_dimensions(data, sampling_fraction=0.5, components=20)
    
    # MODEL TRAINING
    model = LogisticRegression(C=1)
    model = train_model(model, sampling_fraction=0.5)
    
    # REGION-MERGING SEGMENTATION
    segmentation = hierarchical_merging(verbose=False)
    
    # COMPARISON WITH REFERENCE
    compare_to_reference(segmentation, labels, eulers)
