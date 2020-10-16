# Grain Identification in Polycrystals by Linear Classification and Region Merging of Directional Reflectance Signals 

Corresponding publication is available from [**Link to publication**]

--------

To test the code the experimental DRM datasets presented in the publication, please first download the data from [**Link to datasets**] and add the data folder to the code repository next to the **test** and **lib** folders. Each folder in the **/data** directory contains the necessary data for each studied sample (**Ni**, **Al** and **Ti**) in three separate NPY files:

- **data.npy**	Contains the DRM dataset in the form of a 4D matrix of shape (x, y, theta, phi) and type UINT-8.

- **eulers.py**	Contains the registered Euler angles map (shape (x, y, 3), type float32) corresponding to the segmentation domain, as measured by EBSD.

- **labels.py**	Contains a reference segmentation of the domain determined by the Matlab MTEX grain segmentation algorithm applied to the EBSD data with a minimum misorientation angle threshold of 5 degrees.

Segmentation results presented and discussed in the publication can then be reproduced by running the test file **test_LRC-MRM.py** located in the **/tests** folder. The output of the test is a segmented grain map resulting from the LRC-MRM pipeline visualized against the reference and complemented with the corresponding grain size distributions.

For any question or inquiry, please contact the corresponding author.

