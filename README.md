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

4. **OPM-MEG data** (.FIF file): The OPM-MEG recording data that requires co-registration.

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

5. **Save Results (Step 4)**
   - Once registration is complete, click "Save Results"
   - Select a directory to save the transformation files
   - The system will save:
     - Updated MEG data with device-to-head transformation
     - MRI-to-head transformation file
     - Text files with reference point coordinates

## Expected Accuracy

- **Helmet Points**: Mean error should be <2mm, with maximum error <3mm. Currently the mean erros  is 1.61.
- **Fiducial Points**: Ideally <3mm, though helmet point accuracy is more critical for OPM-MEG source localisation. Still working on this but currently at <6mm.

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

## References

- Zetter et al. (2019). Optical Co-registration of MRI and On-scalp MEG. Scientific Reports, 9(1), 5490.
- FieldTrip documentation: Coregistration of optically-pumped magnetometer (OPM) data: https://www.fieldtriptoolbox.org/tutorial/source/coregistration_opm/#coregistration-of-optically-pumped-magnetometer-opm-data
- University of York, York Neuroimaging Centre: Richard Aveyard, Alex Wade, Joe Lyons
