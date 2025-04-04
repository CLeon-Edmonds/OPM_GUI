# OPM-Coregistration
This is a source code that will run on MNE python. THe aim of the app is to ease OPM-MEg corregistration using a 3D LiDAR scanner for source localisation. 

This current version seems tobe working fine. The IPC works well and matches the points automatically. The GUI is working fine but will require some improvements. Specifically around the laoding bar and teh order of some of the buttons. 

The transformation matrix seems to be working fine. There are still minimal errors with the fiducials, as it marks the erros around 8mm, which is still too high. 

The pipeline works with no errors, but there is plenty debugging at each stage to notify if there were to be a problem. 

Currently i am using a scalp model from FreeSurfer, which seems to have been working fine. When overlapping the models, we will notice theat they don't fully overlap, but it seems to be that this is due to the hair of the participant. 
