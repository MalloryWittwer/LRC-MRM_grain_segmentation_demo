
import numpy as np
from skimage.future.graph import RAG
import heapq
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize
from lib.segmentation.training import _vector_similarity

def lrc_mrm_segmentation(dataset, model, p_limit=0.5):
    '''
    Inspired by skimage.future.graph.merge_hierarchical function of Skimage.
    (https://github.com/scikit-image/scikit-image/blob/master/skimage/)
    Performs region-merging segmentation using previously trained classifier
    model to define edge weights. Details in the publication.
    '''
    rx, ry = dataset.get('spatial_resol')
    data = dataset.get('data')
    
    # Merging decision function
    mdf = lambda vect:model.predict_proba(np.atleast_2d(vect))[0,1]
    
    # Initialize the RAG graph
    rag, edge_heap, segments = _initialize_graph(rx, ry, data, model, p_limit)
    
    # Start the region-merging algorithm
    while (len(edge_heap) > 0) and (edge_heap[0][0] < p_limit):
        # Pop the smallest edge if weight < p_limit
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
                weight = mdf(_vector_similarity(master_n2, master_nbr))
                heap_item = [weight, n2, nbr, (weight < p_limit)]
                edge['heap item'] = heap_item
                # Push edges to the heap
                heapq.heappush(edge_heap, heap_item)
    
    # Compute grain map
    label_map = np.arange(segments.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for lab in d['labels']:
            label_map[lab] = ix
    segmentation = label_map[segments]
    
    # Grain boundary map
    gbs = skeletonize(find_boundaries(segmentation, mode='inner'))
    
    # Return updated dataset
    dataset['segmentation'] = segmentation.ravel()
    dataset['boundaries'] = gbs.ravel()
    
    return dataset

def _initialize_graph(rx, ry, data, model, p_limit=0.5):
    '''Initializes the Region Adjacency Graph (RAG).'''
    
    # Merging decision function
    mdf = lambda vect:model.predict_proba(np.atleast_2d(vect))[0,1]
    
    # Initialize RAG
    xmap, ymap = _get_xymaps(rx, ry)
    segments = np.arange(rx*ry).reshape((rx,ry))
    rag = RAG(segments)
    
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
        weight = mdf(_vector_similarity(master_x, master_y))        
        # Push the edge in the heap
        heap_item = [weight, n1, n2, (weight < p_limit)]
        d['heap item'] = heap_item
        heapq.heappush(edge_heap, heap_item)
    
    return rag, edge_heap, segments

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