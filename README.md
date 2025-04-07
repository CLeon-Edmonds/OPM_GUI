# OPM-MEG Co-Registration GUI

## Overview

This software provides a graphical user interface for co-registering Optically Pumped Magnetometer (OPM) MEG data with structural MRI scans. It facilitates the alignment of sensor positions with the subject's anatomy, which is essential for accurate source localisation in MEG analysis.

## Prerequisites

Before using this software, you will need:

1. A structural T1-weighted MRI scan of the subject
2. FreeSurfer installed and configured on your system
3. OPM-MEG data in .fif format
4. 3D scans of the subject's head both inside and outside the Magnetically Shielded Room (MSR)
5. Python 3.6+ with the dependencies installed (see installation instructions below)

### Installation of Dependencies

Please ensure all of the following dependencies are installed. Please note that the GUI might not run correctly with Python 3.12 and above as there is limited ML capability for the GUI to run correctly. Furthermore, this GUI currently only runs on MacOS. Further iterations will provide compatabilty for Linux/Windows.

To install the necessary dependencies, add this to your command line:

```bash
# Install required Python packages
pip install numpy
pip install mne
pip install PyQt5
pip install vtk
pip install open3d

# Additional dependencies may be required on some systems
pip install scipy matplotlib
```

For macOS users using Homebrew:
```bash
brew install vtk
```

## Input Files Required

The software requires the following input files:

1. **Inside MSR Scan** (.ply format): 3D scan of the subject's head captured inside the MSR with the OPM sensors in place. Please ensure that this scan has all of the 7 kknow points from your helmet (the stickers) visible. It is importatnt that these have not been deleted during rendering.
2. **Outside MSR Scan** (.ply format): 3D scan of the subject's head captured outside the MSR. Likewise, this should include all fiducials clearly with no holes.
3. **Scalp Surface** (.stl format): Generated from the subject's MRI using FreeSurfer (instructions below)
4. **OPM Data** (.fif format): The MEG data file that requires co-registration

## FreeSurfer Processing Guide

Before using this software, you must process the MRI data using FreeSurfer. Follow these detailed steps:

1. Set up FreeSurfer environment variables (add to your .bashrc or .bash_profile for permanence):
   ```bash
   export FREESURFER_HOME=/path/to/freesurfer
   source $FREESURFER_HOME/SetUpFreeSurfer.sh
   export SUBJECTS_DIR=/path/to/your/subjects/directory
   ```

2. Convert your MRI to NIfTI format if it isn't already:
   ```bash
   mri_convert your_dicom_directory output_file.nii.gz
   ```

3. Run the FreeSurfer reconstruction:
   ```bash
   recon-all -i your_t1.nii -s subject_name -all
   ```
   This process typically takes 8-12 hours to complete. You can check progress with:
   ```bash
   tail -f $SUBJECTS_DIR/subject_name/scripts/recon-all.log
   ```

4. Generate the scalp surface file using one of these methods:

   a. If using standard FreeSurfer 6.0+:
   ```bash
   #Generate the outer skin surface
   mri_watershed -surf $SUBJECTS_DIR/subject_name/mri/T1.mgz $SUBJECTS_DIR/subject_name/surf/skin
   
   #Convert to STL format
   mris_convert $SUBJECTS_DIR/subject_name/surf/lh.skin $SUBJECTS_DIR/subject_name/surf/lh.skin.stl
   ```

   b. Alternative method using seghead if available:
   ```bash
   #Check if seghead exists
   ls $SUBJECTS_DIR/subject_name/surf/lh.seghead.surf
   
   #Convert to STL
   mris_convert $SUBJECTS_DIR/subject_name/surf/lh.seghead.surf $SUBJECTS_DIR/subject_name/surf/scalp.stl
   ```

   c. If using the newer head segmentation in FreeSurfer 7.0+:
   ```bash
   #Run the head segmentation
   mri_segment --seg $SUBJECTS_DIR/subject_name/mri/T1.mgz --wm $SUBJECTS_DIR/subject_name/mri/wm.mgz --subcort $SUBJECTS_DIR/subject_name/mri/aseg.mgz --surf $SUBJECTS_DIR/subject_name/surf --subject $SUBJECTS_DIR/subject_name/mri/norm.mgz
   
   #Generate the outer skin surface
   mri_tessellate $SUBJECTS_DIR/subject_name/mri/skin.mgz 1 $SUBJECTS_DIR/subject_name/surf/skin
   
   #Convert to STL
   mris_convert $SUBJECTS_DIR/subject_name/surf/skin $SUBJECTS_DIR/subject_name/surf/scalp.stl
   ```

5. Verify the scalp extraction quality using:
   ```bash
   freeview -v $SUBJECTS_DIR/subject_name/mri/T1.mgz -f $SUBJECTS_DIR/subject_name/surf/scalp.stl:edgecolor=red
   ```

## Co-Registration Workflow

The software implements a step by step co-registration process:

### 1. Load Data Files
Select all required files using the buttons on the left panel.

### 2. Inside MSR Registration
- The Inside MSR scan will be displayed.
- Mark the 7 helmet label points from left to right using Shift+Left Click.
- These points correspond to known positions on the OPM helmet.
- Use the "Fit Points" button to optimise the points if needed. You may do this a few times until all points are under 3mm.
- Click "Confirm Points" to calculate the transformation.
- Click "Continue" when satisfied.

### 3. Outside MSR Registration
- The Outside MSR scan will be displayed.
- Mark 3 fiducial points in the following order:
  1. Right pre-auricular point (RPA)
  2. Left pre-auricular point (LPA)
  3. Nasion
- Click "Confirm Points" to store the fiducials.
- Click "Continue" to proceed.

### 4. MRI Registration
- The software will automatically compute the head-to-standard transformation.
- It will then calculate the MRI-to-head transformation.
- Both the scalp model and head scan will be visualised together to verify alignment.
- Please note that this process may take time. Do not quit the GUI or force it as the transformation may not be accurate and it will not save. Pay attention to the progress bar at the bottom left.

### 5. Save Results
- Click "Save Results" to save the transformation files.
- The software will save the following files:
  - Updated MEG .fif file with the device-to-head transformation
  - MRI-to-head transformation file (.fif format)
  - Text files with the selected points for reference

## Mathematical Background

The software computes three key transformations using rigid body mathematics and optimisation techniques:

1. **X₁: Inside MSR to Helmet Transformation**
   - Maps the points selected on the inside MSR scan to known positions in the helmet coordinate system.
   - Uses a rigid body transformation computed via Singular Value Decomposition (SVD).
   
   Given source points P from the inside MSR scan and target points Q in the helmet coordinate system:
   
   a) Center both point sets:
      ```
      P̄ = 1/n ∑ Pᵢ
      Q̄ = 1/n ∑ Qᵢ
      Pᶜ = Pᵢ - P̄
      Qᶜ = Qᵢ - Q̄
      ```
   
   b) Compute the cross-covariance matrix H:
      ```
      H = Pᶜᵀ × Qᶜ
      ```
   
   c) Apply SVD to get the rotation matrix R:
      ```
      [U, S, Vᵀ] = SVD(H)
      R = V × Uᵀ
      ```
      
      If det(R) < 0, then:
      ```
      V'ₙ = -Vₙ
      R = V' × Uᵀ
      ```
   
   d) Compute the translation vector t:
      ```
      t = Q̄ - R × P̄
      ```
   
   e) Construct the transformation matrix X₁:
      ```
      X₁ = [R | t]
           [0 | 1]
      ```

2. **X₂: Head to Standard Coordinate System Transformation**
   - Maps the fiducial points (RPA,LPA, and NAS) to a standard coordinate system.
   
   a) Define the origin as the midpoint between the pre-auricular points:
      ```
      origin = (RPA + LPA)/2
      ```
   
   b) Define the x-axis as pointing from origin to nasion (normalized):
      ```
      x = (NAS - origin)/‖NAS - origin‖
      ```
   
   c) Define the temporary y-axis as pointing from origin to LPA:
      ```
      y_tmp = (LPA - origin)
      ```
   
   d) Define the z-axis as orthogonal to the x-y plane:
      ```
      z = x × y_tmp/‖x × y_tmp‖
      ```
   
   e) Define the final y-axis to ensure orthogonality:
      ```
      y = z × x
      ```
   
   f) Construct the rotation matrix using these orthogonal axes:
      ```
      R = [x y z]
      ```
   
   g) Construct the transformation matrix X₂:
      ```
      X₂ = [R | -R×origin]
           [0 |    1    ]
      ```

3. **X₃: MRI to Head Transformation**
   - Aligns the MRI scalp surface with the head scan from outside the MSR.
   - Initially aligned using fiducial points then refined using Iterative Closest Point (ICP).
   
   The ICP algorithm iteratively refines the transformation by:
   
   a) For each point in the source cloud, find the closest point in the target cloud.
   
   b) Estimate the transformation that minimizes:
      ```
      E(R,t) = 1/N ∑ᵢ ‖Rpᵢ + t - qᵢ‖²
      ```
      where pᵢ are points from the source cloud and qᵢ are their corresponding closest points in the target cloud.
   
   c) Apply the transformation to the source cloud.
   
   d) Repeat until convergence (change in error < threshold).

The final transformations computed are:
- **X₂₁ = X₂ × X₁**: Combined transformation (device to head)
- **X₃**: MRI to head transformation

The software implements weighted fiducial refinement to minimise fiducial registration error (FRE):
```
FRE = 1/n ∑ᵢ ‖X₃×fᵢᴹᴿᴵ - fᵢᴴᴱᴬᴰ‖
```
where fᵢᴹᴿᴵ are fiducial points in MRI space and fᵢᴴᴱᴬᴰ are fiducial points in head space. The goal is to keep all errors under 3mm.



## Output Files

After successful co-registration, the software generates:

1. **Updated MEG File**: Original .fif file updated with the device-to-head transformation
2. **Transformation File**: MRI-to-head transformation in .fif format
3. **Inside MSR Points**: Text file listing the selected helmet points
4. **Fiducial Points**: Text file listing the selected fiducial points

These files can be used directly with MNE-Python and other MEG analysis tools.

## Troubleshooting

- **Poor Alignment**: If the visualised alignment looks incorrect, try:
  - Selecting different points and rerunning the registration
  - Ensuring the fiducial points are accurately placed
  - Using the "Fit Points" function to optimise point selection

- **High Registration Errors**: If errors exceed 3mm:
  - The software will attempt automatic refinement
  - You may need to manually improve fiducial point selection

- **FreeSurfer Issues**: If you encounter problems with FreeSurfer:
  - Ensure FreeSurfer is correctly installed and configured
  - Check that $SUBJECTS_DIR is correctly set
  - Verify that the MRI data is of sufficient quality for FreeSurfer processing

## Acknowledgements

This software was developed with inspiration and methods derived from the OPM research programme at York Neuroimaging Centre, University of York, in particualr: Richard Aveyard, Alex Wade and Joe Lyons.

The co-registration algorithms incorporate techniques developed through collaborative efforts in the MEG community, with particular thanks to the MNE-Python team for their excellent software ecosystem that this tool builds upon.


## Contact

For support or queries, please contact:

Christopher Leon Edmonds-Hunt: cle464@student.bham.ac.uk, christopherleonhunt@gmail.com
Milan Nedoma: {ask for Milans email}
Anna Kowalczyk (Supervisor): a.kowalczyk@bham.ac.uk
