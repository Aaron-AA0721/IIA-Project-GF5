# GF5: Animating 3D Characters

[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-0969da)](https://cambridgecvcourses.github.io/IIA-Project-GF5/)

This repository contains the code, assets, and handout source for an
undergraduate project on 3D character animation. The rendered student handouts
are available on the project page.

The project is organised into three parts:

- Part 1: forward kinematics on a rigid block character
- Part 2: skinning weights and linear blend skinning using SMPL
- Part 3: later group character-animation brief

## Handouts

- [Project Overview](docs/project_overview.md)
- [Part 1: Forward Kinematics](docs/part1.md)
- [Part 2: Skinning and LBS](docs/part2.md)
- [Interim Report](docs/interim.md)
- [Part 3: Coming Soon](docs/part3_placeholder.md)

The project page is the main student-facing version of the Parts 1 and 2
handouts. The Markdown files in `docs/` are included here as source documents.

## Running the Viewer

Create the Python environment from the shipped environment file:

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
