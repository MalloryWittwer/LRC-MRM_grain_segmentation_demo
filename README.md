# A Parameter-free Grain Segmentation Method based on Directional Reflectance Microscopy 

Companion code to our 2021 [publication](https://doi.org/10.1016/j.matchar.2021.110978) in *Materials Characterization*.

## Installation

We recommend installing Python > 3.6 and the dependencies listed in requirements.txt in a fresh environment.

## Testing the code

To test the code the experimental DRM datasets presented in the publication, please first 

Download the **data** folder from [this Mendeley Dataset](http://dx.doi.org/10.17632/t4wvpy29fz.1) and place it next to the **test** and **lib** folders. The **/data** folder contains DRM datasets and segmentation ground truths for **Ni**, **Al** and **Ti**. In each subfolder:

- **drm_data.npy**:	The DRM dataset in the form of a 4D matrix of shape (x, y, theta, phi) and type UINT-8.

- **euler_angles.py**:	The registered Euler angles map (shape (x, y, 3), type float32) corresponding to the segmentation domain, measured by EBSD.

- **reference.py**:	A reference segmentation of the domain determined by Matlab MTEX based on the EBSD dataset (misorientation threshold of 5 degrees).

To reproduce the results in our publication, run **test_LRC-MRM.py** from the **tests** folder. The execution produces a segmented grain map of the DRM dataset visualized against the reference.

For any inquiry, please contact Mallory Wittwer at mallory.wittwer@gmail.com.
