# GF5: Animating 3D Characters

[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-0969da)](https://cambridgecvcourses.github.io/IIA-Project-GF5/)

This repository contains the codebase for the IIA Project GF5: Animating 3D Characters. Start with the [project page](https://cambridgecvcourses.github.io/IIA-Project-GF5/) for the brief, instructions, and
interim report, then use this repo to run the viewer and complete the coding
tasks.

The project is organised into three parts:

- Part 1: forward kinematics on a rigid block character
- Part 2: skinning weights and linear blend skinning using SMPL
- Part 3: later group character-animation brief

## Project Handouts

- [Project Overview](https://cambridgecvcourses.github.io/IIA-Project-GF5/)
- [Part 1: Forward Kinematics](https://cambridgecvcourses.github.io/IIA-Project-GF5/part1.html)
- [Part 2: Skinning and LBS](https://cambridgecvcourses.github.io/IIA-Project-GF5/part2.html)
- [Interim Report](https://cambridgecvcourses.github.io/IIA-Project-GF5/interim.html)
- [Part 3: Coming Soon](https://cambridgecvcourses.github.io/IIA-Project-GF5/part3.html)

## Running the Viewer

You will need to run the viewer locally. Create the Python environment from the shipped environment file:

```bash
mamba env create -f env.yml
mamba activate gf5
```

From the repository root:

```bash
python viewer/asset_viewer.py
```

This opens the viewer page in your browser automatically.
To use a different port, for example when another viewer is already running:

```bash
python viewer/asset_viewer.py --port 8090
```

## Repository Layout

- `docs/`: handout source documents
- `viewer/`: the animation viewer and implementation stubs
- `assets/blocky/`: the block character assets
- `assets/smpl/`: local SMPL model files, if installed
- `libraries/`: local pose and motion libraries created in the viewer
- `exports/`: local rendered videos
