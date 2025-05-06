import mne

# Load the fiducials file
fids = mne.io.read_fiducials('/Volumes/CHRISDRIVE1/Test/derivatives/FreeSurfer/sub-01/bem/sub-01-fiducials.fif')

# Print fiducial points
for fid in fids[0]:
    print(f"{fid['ident']}: {fid['r']}  ({fid['kind']})")