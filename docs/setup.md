# Setup and Viewer Guide

The project uses a local Python viewer for skeleton inspection, pose editing,
skinning-weight visualisation, and video export.

## Create The Environment

From the repository root, create and activate the course environment once:

```bash
mamba env create -f env.yml
mamba activate gf5
```

## Run The Viewer

Start the local viewer from the repository root:

```bash
python viewer/asset_viewer.py
```

If the default port is already in use, choose another port:

```bash
python viewer/asset_viewer.py --port 8090
```

For a remote or headless session, suppress automatic browser opening:

```bash
python viewer/asset_viewer.py --no-open-browser
```

Then open the printed `localhost` URL in a browser.

## Files You Will Edit

For Part 1, implement forward kinematics in:

```text
viewer/student_submission/part1_fk.py
```

For Part 2, implement one-hot and LBS skinning in:

```text
viewer/student_submission/part2_skinning.py
```

You will also use these files to understand how data flows through the viewer:

- `viewer/asset_viewer.py`
- `viewer/motion_sequences.py`
- `viewer/smpl_support.py`
- `viewer/skeleton_profiles.py`

## Main Viewer Controls

- `Motion` and `Animate`: choose and play a built-in or saved motion clip.
- `Show Skeleton` and `Show Mesh`: inspect the rig and visible character
  surface.
- `Selected Joint` and `Joint Editor`: rotate individual joints and debug the
  hierarchy.
- `Timeline`: capture keyframes for the custom Part 1 motion.
- `Export Motion Video`: render evidence videos for the interim submission.

## SMPL Model Files

SMPL model files are not included in the repository. If you have access to
them, place them under:

```text
assets/smpl/models/
```

or start the viewer with an explicit model path:

```bash
python viewer/asset_viewer.py --smpl-model /path/to/smpl/model.pkl
```

## Next Step

Once the viewer runs, continue to [Part 1](part1.md).
