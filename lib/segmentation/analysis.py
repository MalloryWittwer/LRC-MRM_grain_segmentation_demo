import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize
from sklearn.metrics import normalized_mutual_info_score as nmis

def compare_to_reference(dataset):
    '''
    Shows a comparison between segmentation and reference, returns MIS score.
    '''
    # Collect data from dataset
    rx, ry = dataset.get('spatial_resol')
    segmentation = dataset.get('segmentation').reshape((rx,ry))
    reference = dataset.get('labels').reshape((rx,ry))
    eulers = dataset.get('eulers').reshape((rx,ry,3))
    
    # Compute grain boundaries on segmentation and reference
    gbsseg = skeletonize(find_boundaries(segmentation, mode='inner'))
    gbsref = skeletonize(find_boundaries(reference, mode='inner'))
    
    # Assign coloring based in IPF-Z    
    rot_mat_inv, _ = _eulers_to_orientation(eulers.reshape((rx*ry,3)))
    ipfZ = _ipf_Z_map(rot_mat_inv).reshape((rx,ry,3))
    
    np.save('C:/Users/mallo/Desktop/ipfZ.npy', ipfZ)
    
    # ### SHOWING ROW IPF-Z
    # fig, ax = plt.subplots(figsize=(8,8))
    # ax.imshow(ipfZ)
    # ax.axis('off')
    # plt.show()
    
    lab = label2rgb(segmentation, ipfZ, kind='avg')
    rgb_seg = label2rgb(segmentation, lab, kind='avg')
    rgb_seg[gbsseg] = (1,1,1)
    rgb_ref = label2rgb(reference, lab, kind='avg')
    rgb_ref[gbsref] = (1,1,1)
    
    # Mutual information score between segmentation and reference
    mis_score = nmis(segmentation.ravel(), reference.ravel())

    # Plot a figure
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(121)
    ax.imshow((rgb_seg*255).astype(np.uint8))
    ax.axis('off')
    ax.set_title('Output segmentation', weight='bold')
    ax = fig.add_subplot(122)
    ax.imshow((rgb_ref*255).astype(np.uint8))
    ax.axis('off')
    ax.set_title('Reference', weight='bold')
    plt.tight_layout()
    plt.show()

    return mis_score

def reconstruction_analysis(dataset, data):
    
    rx, ry = dataset.get('spatial_resol')
    s0, s1 = dataset.get('angular_resol')
    rag = dataset.get('rag')
    um_per_pix = dataset.get('um_per_px')
    compressor = dataset.get('compressor')
    comps = compressor.compressor.components_
    compressed_data = dataset.get('data')
    
    reconstructed_data = np.matmul(compressed_data, comps)
    
    print('DONE') # Bugs from here...
    error_map = np.mean(np.square(reconstructed_data-data), axis=1)
    error_map = error_map.reshape((rx,ry))#[rx//4+50:rx//2+50,ry//4:ry//2]
    seg = dataset.get('segmentation').reshape((rx,ry))#[rx//4+50:rx//2+50,ry//4:ry//2]
    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    ax.imshow(error_map)
    ax.axis('off')
    plt.show()
    
    data_reshaped = data.reshape((rx,ry,s0*s1))
    errs = []
    sizes = []
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        m = d.get('master')
        x, y = int(d.get('xpos')), int(d.get('ypos'))
        orig = data_reshaped[x,y]
        reco = np.matmul(m, comps)#.reshape((s0, s1))
        recon_error = np.mean(np.square(reco-orig))
        errs.append(recon_error)
        sizes.append(d.get('count'))

    errs = np.array(errs)
    sizes = np.sqrt(np.array(sizes)*um_per_pix**2)
    
    ### Size partitioning histogram
    
    size_threshold = 50
    small_grain_errors = errs[sizes<size_threshold]
    large_grain_errors = errs[sizes>size_threshold]
    
    med = np.median(errs)
    print('MED: ', med)
    print('AVG: ', np.mean(errs))
    
    error_quantile = med
    f_small = small_grain_errors[small_grain_errors>error_quantile]
    f_large = large_grain_errors[large_grain_errors>error_quantile]
    
    print('f small: ', f_small.shape)
    print('f large: ', f_large.shape)
    print('small grains: ', small_grain_errors.shape)
    print('large grains: ', large_grain_errors.shape)
    
    f = len(f_small) / (len(f_large) + len(f_small))
    print(f'Fraction of anomalies in small grains: {f}')
    n = len(small_grain_errors) / (len(large_grain_errors) + len(small_grain_errors))
    print(f'Fraction of anomalies in all grains: {n}')
    
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    sns.distplot(large_grain_errors, bins=20, kde=True, ax=ax, label='> 100 px')
    sns.distplot(small_grain_errors, bins=20, kde=True, ax=ax, label='< 100 px')
    plt.legend()
    plt.show()
    
    ###

    bin_width = 100
    
    binned_errs = []
    binned_stds = []
    labs = []
    for k in range(0, 500+bin_width, bin_width):
        filt = (sizes < k+bin_width) & (sizes > k)
        binned_errs.append(errs[filt].mean())
        binned_stds.append(errs[filt].std())
        labs.append('{}-{}'.format(k, k+bin_width))
    
    filt = (sizes > 500)
    binned_errs.append(errs[filt].mean())
    binned_stds.append(errs[filt].std())
    labs.append('>500')

    bnr = (binned_errs-min(errs))/max(errs-min(errs))
    fig, ax = plt.subplots(figsize=(4,3), dpi=200)
    ax.set_xlabel('Region size (microns)')
    # ax.set_ylabel('NMF reconstruction error')
    ax.bar(labs, binned_errs, 
           color=plt.cm.viridis(bnr),
           yerr=binned_stds, capsize=3)
    ax.set_xticklabels(labs, rotation=30)
    # ax.set_yticks([])
    plt.show()
    
    df = pd.DataFrame(data=seg.ravel(), columns=['grain'])
    df['grainID'] = df['grain'].astype('str')
    groups = df.groupby('grainID').count().sort_values('grain', ascending=False)
    small_grains = np.squeeze(groups.values < 10)
    small_grains = groups.index[small_grains].values.astype('int')
    
    mask = np.zeros(seg.shape, dtype=np.uint8)
    for idx in small_grains:
        mask[seg==idx] = 1
    
    gbs = dataset.get('boundaries').astype('int').reshape((rx,ry))#[rx//4+50:rx//2+50,ry//4:ry//2]
    
    composite = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    composite[(gbs==0)] = np.array([0,111,158])
    composite[gbs==1] = np.array([255,255,255])
    composite[mask==1] = np.array([255,0,0])
    
    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    ax.imshow(composite)
    ax.axis('off')
    plt.show()
    
    return sizes, errs

def size_dist_plot(dataset, limdown=0, limup=1000, gsize=100):
    '''
    Plots cumulative grain size distribution of segmentation and 
    compares with reference.
    '''
    rx, ry = dataset.get('spatial_resol')
    segmentation = dataset.get('segmentation').reshape((rx,ry))
    um_per_pix = dataset.get('um_per_px')
    unit = 'microns' if um_per_pix else 'px'
    
    sizes = np.array([
        np.sum(segmentation==k) for k in range(np.max(segmentation)) \
            if (np.sum(segmentation==k) >= limdown)
    ])
    sizes = np.sqrt(sizes*um_per_pix**2)
    
    labels = dataset.get('labels')
    labels = labels.reshape((rx,ry))
    sizesLabels = np.array([
        (labels==k).sum() for k in range(np.max(labels)) \
            if (np.sum(labels==k) >= limdown)
    ])
    sizesLabels = np.sqrt(sizesLabels*um_per_pix**2)

    fig, ax = plt.subplots(figsize=(3,3), dpi=200)
    sns.distplot(sizes, ax=ax, label='LRC-MRM', hist=False, bins=gsize,
                  kde_kws={'gridsize':gsize,
                          'cumulative':True},
                  hist_kws={'density':True,
                            'cumulative':True,
                            'edgecolor':'black', 
                            'linewidth':1},
                  )
    sns.distplot(sizesLabels, ax=ax, label='Reference', hist=False, bins=gsize,
                  kde_kws={'gridsize':gsize,
                          'cumulative':True},
                  hist_kws={'density':True,
                            'cumulative':True,
                            'edgecolor':'black', 
                            'linewidth':1},
                  )
    ax.set_xlim(limdown, limup)
    ax.set_ylim(0,1)
    ax.set_ylabel('Cumulative fraction')
    ax.set_xlabel(f'Grain size ({unit})')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.legend(loc=4)
    plt.show()
    
def _eulers_to_orientation(eulers_list):
    '''Converts a set of euler angles to a set of rotation matrices and inv.'''
    i1  = eulers_list[:,0]
    i2  = eulers_list[:,1]
    i3  = eulers_list[:,2]       
    i1c = np.cos(i1)
    i1s = np.sin(i1)
    i2c = np.cos(i2)
    i2s = np.sin(i2)
    i3c = np.cos(i3)
    i3s = np.sin(i3)
    x00 = i1c*i2c*i3c-i1s*i3s
    x01 = -i3c*i1s-i1c*i2c*i3s
    x02 = i1c*i2s
    x10 = i1c*i3s+i2c*i3c*i1s
    x11 = i1c*i3c-i2c*i1s*i3s
    x12 = i1s*i2s
    x20 = -i3c*i2s
    x21 = i2s*i3s
    x22 = i2c
    c0 = np.concatenate((np.expand_dims(x00,-1), 
                         np.expand_dims(x01,-1), 
                         np.expand_dims(x02,-1)), axis=1)
    c1 = np.concatenate((np.expand_dims(x10,-1), 
                         np.expand_dims(x11,-1), 
                         np.expand_dims(x12,-1)), axis=1)
    c2 = np.concatenate((np.expand_dims(x20,-1), 
                         np.expand_dims(x21,-1), 
                         np.expand_dims(x22,-1)), axis=1)
    rot_mat = np.concatenate((np.expand_dims(c0,-1), 
                              np.expand_dims(c1,-1),
                              np.expand_dims(c2,-1)), axis=1)
    rot_mat = rot_mat[:,:,0]
    rot_mat_inv = np.linalg.inv(np.reshape(rot_mat , [rot_mat.shape[0], 3, 3]))
    return rot_mat_inv, rot_mat

def _colorize(indeces):
    '''Returns a color based on the Miller indeces of a plane'''
    a = np.abs(np.subtract(indeces[:,2], indeces[:,1]))
    b = np.abs(np.subtract(indeces[:,1], indeces[:,0]))
    c = indeces[:,0]
    rgb = np.concatenate((np.expand_dims(a, -1), 
                          np.expand_dims(b, -1), 
                          np.expand_dims(c, -1)), axis=1)
    maxes = np.max(rgb, axis=1)
    a = a/maxes
    b = b/maxes
    c = c/maxes
    rgb    = np.concatenate((np.expand_dims(a, -1), 
                             np.expand_dims(b, -1), 
                             np.expand_dims(c, -1)), axis=1)
    return rgb

def _symmetrize(indeces):
    indeces = np.abs(indeces)
    norms = (np.square(indeces[:,0]) +
             np.square(indeces[:,1]) + 
             np.square(indeces[:,2]))
    indeces = np.concatenate((np.expand_dims(indeces[:,0]/norms, -1), 
                              np.expand_dims(indeces[:,1]/norms, -1),
                              np.expand_dims(indeces[:,2]/norms, -1)), axis=1)
    indeces = np.sort(indeces, axis=1)
    return indeces

def _ipf_Z_map(rot_mat_inv):
    '''Returns the IPF (Z) map based on an inverted rotation matrix'''
    output = np.concatenate((_symmetrize(rot_mat_inv[:,:,0]),
                             _symmetrize(rot_mat_inv[:,:,1]),
                             _symmetrize(rot_mat_inv[:,:,2])), axis=1)
    rgbz_pred = _colorize(output[:,6:9])
    return rgbz_pred