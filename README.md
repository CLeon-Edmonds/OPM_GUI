# OPM-MEG Coregistration GUI

## Overview

The OPM-MEG Coregistration GUI is a software tool designed to accurately align Optically-Pumped Magnetometer (OPM) MEG sensor positions with MRI anatomical data. 

This GUI simplifies the coregistration process for OPM-MEG by:
- Providing an intuitive interface for loading and visualising data
- Guiding users through each step of the coregistration workflow
- Automating complex transformation calculations
- Providing quantitative error metrics
- Generating the necessary transformation files for MEG analysis

Please note that this is only optimised for MacOS. Future iterations will provide fully optimised functions for LINUX/Windows.

## Required Input Files

To use this application, you need:

1. **Inside MSR scan** (.PLY file): 3D scan of the subject's head while wearing the OPM sensor helmet inside the magnetically shielded room (MSR) (this would be the LiDAR scan). Esnure 7 ppoints from helmet are visible in scan and were not deleted after rendering outlier point clouds.

2. **Outside MSR scan** (.PLY file): 3D scan of the subject's head outside the MSR, typically with fiducial markers visible. (This would also be a LiDAR scan). Make sure fiducials are clearly visible, avoid holes in these areas.

3. **MRI scalp surface** (.STL file): Surface mesh of the subject's scalp extracted from MRI data (typically using FreeSurfer), either from your pipeline or detaisl below on how to achieve it. 

4. **OPM-MEG data** (.FIF file): The OPM-MEG recording data that requires coregistration.

## Instructions

### Preparation

1. Process MRI data with FreeSurfer:
   ```
   recon-all -i your_t1.nii -s subject_name -all
   ```
(Skip this section if you have completed this via your pipeline)

2. Generate scalp surface STL file:
   ```
   mris_convert $SUBJECTS_DIR/subject_name/surf/lh.seghead.surf scalp.stl
   ```
   
   If the seghead surface is not available, use:
   ```
   mri_watershed -surf $SUBJECTS_DIR/subject_name/mri/T1.mgz $SUBJECTS_DIR/subject_name/surf/scalp
   mris_convert $SUBJECTS_DIR/subject_name/surf/scalp scalp.stl
   ```

3. Acquire OPM data (.fif file)

4. Obtain 3D scans of the subject's head both inside and outside the MSR (PLY files)
  - Instructions for scan adquisition can be found here: {I will add that later}

### Coregistration Workflow

1. **Load Files**
   - Click the "Select" button for each file type
   - Choose the appropriate file for each category
   - Click "Load" when all files are selected

2. **Inside MSR Registration (Step 1)**
   - The GUI will display the inside MSR scan
   - Select the 7 helmet label points in sequence using Shift+Left Click, starting on the right side of the       participant (your left on the screen).
   - These points correspond to known positions on the helmet
   - Click "Confirm Points" when all 7 points are selected
   - If needed, use "Fit Points" to optimize the selection. This can be repeated until all points are under 3mm and displayed in green.
   - Click "Continue" to proceed to the next step

3. **Outside MSR Registration (Step 2)**
   - The GUI will display the outside MSR scan
   - Select the 3 fiducial points in this order:
     1. Right pre-auricular point
     2. Left pre-auricular point
     3. Nasion
   - Click "Confirm Points" when all 3 fiducials are selected
   - Click "Continue" to proceed to the next step

4. **MRI-to-Head Registration (Step 3)**
   - The GUI will perform the registration automatically
   - This step aligns the MRI scalp surface with the head scans
   - The system will display error metrics for:
     - Helmet points (ideally <3mm)
     - Fiducial points (ideally <3mm)(Not fully achived in this version, currently at <6mm)
   - A visualisation of the alignment will be shown
   - It is likely that this step will take up to 1 minute to complete. Do not kill your terminal or force otehr buttons as this will interrupt the process and the transformation will not be completed. 

5. **Save Results (Step 4)**
   - Once registration is complete, click "Save Results"
   - Select a directory to save the transformation files
   - The system will save:
     - Updated MEG data with device-to-head transformation
     - MRI-to-head transformation file
     - Text files with reference point coordinates
   - BIDS implementation will be added soon.

## Expected Accuracy

- **Helmet Points**: Mean error should be <2mm, with maximum error <3mm. Currently the mean erros  is 1.61.
- **Fiducial Points**: Ideally <3mm, though helmet point accuracy is more critical for OPM-MEG source localisation. Still working on this but currently at <6mm.

## System Requirements and Technical Details

### Software Requirements

- **Python Version**: Python 3.8 or above (please note python 12.0 and above might not have ML support and therefore might not work correctly).
- **Required Packages**:
  - `numpy`: For numerical computations and matrix transformations
  - `mne`: For MEG data handling and transformation file I/O
  - `PyQt5`: For the graphical user interface
  - `vtk`: For 3D visualization of point clouds and meshes
  - `open3d`: For point cloud processing and registration algorithms
  - `copy`: For deep copying of data structures
  - `os`, `sys`: For file operations and system functions (only for MacOS)

These dependencies can be installed via pip:
```bash
pip install numpy mne PyQt5 vtk open3d
```

Alternatively, using conda:
```bash
conda install -c conda-forge numpy mne pyqt vtk open3d
```

### Hardware Requirements

- A computer with at least 8GB RAM
- Graphics card supporting OpenGL 3.3 or higher (for 3D visualization)
- At least 1GB of free disk space

### Mathematical Background

The coregistration process relies on several mathematical transformations:

1. **Kabsch Algorithm**: This algorithm calculates the optimal rigid transformation (rotation and translation) to align two sets of corresponding points. Given two point sets P and Q, it:
   
   - Centers both point sets by subtracting their respective centroids
   - Computes the cross-covariance matrix H = P'Q (where P' and Q are the centered point sets)
   - Performs singular value decomposition (SVD) on H = USV^T
   - Computes the rotation matrix R = VU^T
   - Ensures a proper rotation by checking det(R) > 0
   - Computes the translation t = centroid_Q - R·centroid_P
   - The final transformation matrix combines R and t

   This algorithm minimises the root mean square deviation (RMSD) between the corresponding points.

2. **Iterative Closest Point (ICP)**: Used for refining the alignment between point clouds when there are not enough fiducial points or when additional precision is needed.

3. **Spatial Interpolation**: Used in the refined alignment to distribute corrections smoothly across space, preserving local geometry while improving global alignment.

### Output Files

The coregistration process generates several important output files:

1. **Transformed MEG Data File** (*.fif): Contains the original MEG data with the added device to head transformation matrix. This file can be directly used in source estimation tools like MNE-Python.

3. **Trans File** (*_trans.fif): Contains the MRI-to-head transformation matrix, which is needed for source localisation with MRI data.

4. **Inside MSR Points** (inside_msr_points.txt): Text file containing the coordinates of the selected helmet points.

5. **Fiducial Points** (fiducial_points.txt): Text file with coordinates of the fiducial points selected on the outside MSR scan.

### Implementation Notes

The transformation pipeline is structured to maximize accuracy while being robust to scanning variations:

1. The helmet point registration (X1 transform) aligns the inside MSR scan with a known helmet reference frame.

2. The fiducial point registration (X2 transform) establishes a standard head coordinate system.

3. The combined transformation (X21 = X2 * X1) maps from the MEG device space to the standardised head space.

4. The MRI registration (X3 transform) maps from the MRI reference frame to the standardised head coordinates.

This multi step approach allows for more precise control over each transformation component and enables detailed error metrics at each stage. The software prioritizes helmet point accuracy over fiducial accuracy, as sensor positions relative to brain anatomy are the most critical factor in OPM-MEG source localization.

## Troubleshooting

- If alignments appear incorrect, try clearing points and reselecting them with greater care. Buttons for clearing all points or clearing last point are avaialble and functional. 
- Ensure fiducial points are marked in the correct order, right/left/naison (starting on the right of the participant, your left on the screen).
- Verify that your MRI derived scalp mesh represents the head surface accurately
- For persistent issues with fiducial errors, check that the anatomical landmarks are consistently defined between the MRI and the physical measurements
- Contact the author (Christopher Leon) at cle464@student.bham.ac.uk, christopherleonhunt@gmail.com, {insert Milan's email too} or the project's supervisor Dr Anna Kowalczyk at a.kowalczky@bham.ac.uk.

## Technical Notes

- The co-registration methodology is based on research from the University of York and Aalto University. This can be found here: https://vcs.ynic.york.ac.uk/ynic-public/yorc
- The algorithm uses a multistep transformation pipeline:
  1. Inside MSR to helmet coordinate transformation
  2. Outside MSR to standard coordinate transformation
  3. Helmet to head coordinate alignment
  4. MRI to head registration
- The alignment quality is mathematically verified to ensure accurate sensor localisation

### Critical Helmet Coordinate System

The coregistration relies on accurate specification of the known helmet points in 3D space. The default values are configured for the York OPM system, but these may need to be adjusted for different OPM-MEG setups.

**Important**: You should verify that the helmet reference point coordinates match their specific system before proceeding with coregistration.

The helmet reference point coordinates are defined in the following methods:
- `confirmInsideMSR` method
- `computeHeadToHelmetTransform` method
- `finalizeCoregistration` method

In all these methods, look for the following array definition:
```python
# Known positions of stickers in helmet reference frame
sticker_pillars = np.zeros([7, 3])
sticker_pillars[0] = [102.325, 0.221, 16.345]
sticker_pillars[1] = [92.079, 66.226, -27.207]
sticker_pillars[2] = [67.431, 113.778, -7.799]
sticker_pillars[3] = [-0.117, 138.956, -5.576]
sticker_pillars[4] = [-67.431, 113.778, -7.799]
sticker_pillars[5] = [-92.079, 66.226, -27.207]
sticker_pillars[6] = [-102.325, 0.221, 16.345]
```

If your helmet coordinate system differs from these values, you should update all instances of this array with your system's specific coordinates.

### Known Coordinate Systems

- **Helmet Coordinate System**: Defined by the 7 reference points as specified above. This is system-specific and must be calibrated for your OPM array.

- **Head Coordinate System**: Follows the convention used in neuroimaging where:
  - Origin: Midpoint between the left and right pre-auricular points
  - X-axis: Points through the nasion
  - Y-axis: Points through the left pre-auricular point
  - Z-axis: Points up, perpendicular to the XY plane

- **MRI Coordinate System**: Typically RAS (Right-Anterior-Superior) as used by FreeSurfer and most neuroimaging software.

Ensuring consistency between these coordinate systems is essential for accurate co-registration.

### Modifying the Code for Different Systems

If you need to adapt the software for a different OPM-MEG system:

1. Update the `known_sensor_positions` variable in the `__init__` method
2. Update all instances of `sticker_pillars` in the methods mentioned above
3. Verify that your fiducial definition matches the expected order (RPA, LPA, NAS)
4. Consider adjusting the error thresholds in the `finalizeCoregistration` method if your system has different precision requirements
5. Please do not edit the main, and either download a copy or create a branch for your use.
   
## References

- Zetter et al. (2019). Optical Co-registration of MRI and On-scalp MEG. Scientific Reports, 9(1), 5490.
- FieldTrip documentation: Coregistration of optically-pumped magnetometer (OPM) data: https://www.fieldtriptoolbox.org/tutorial/source/coregistration_opm/#coregistration-of-optically-pumped-magnetometer-opm-data
- University of York, York Neuroimaging Centre: Richard Aveyard, Alex Wade, Joe Lyons
