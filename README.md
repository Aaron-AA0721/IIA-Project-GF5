# GF5: Animating 3D Characters

This repository contains the teaching assets and student handouts for an
undergraduate project on 3D character animation.

The material is being written in Markdown first so it can evolve alongside the
code and assets. The project is organised into three parts:

- Part 1: forward kinematics on a rigid block character
- Part 2: skinning weights and linear blend skinning using SMPL
- Part 3: later group character-animation brief

## Reading Order

- [Student Website](docs/index.html)
- [Project Overview](docs/project_overview.md)
- [Part 1: Forward Kinematics](docs/part1.md)
- [Part 2: Skinning and LBS](docs/part2.md)
- [Interim Report](docs/interim.md)
- [Part 3: Coming Soon](docs/part3.html)

For the Parts 1 and 2 student release, start from the static website at
`docs/index.html`. It can be opened directly in a browser or served from the
`docs/` folder by a simple static host.

The website is generated from the Markdown handouts. After editing Markdown in
`docs/`, regenerate the HTML pages with:

```bash
python docs/build_site.py
```

On GitHub, `.github/workflows/build-docs-site.yml` runs the same generator when
Markdown changes are pushed to `main`, uploads the generated `docs/` directory,
and deploys it with GitHub Pages. Configure GitHub Pages to use GitHub Actions
as the publishing source.

## Repository Layout

- `docs/`: student-facing handouts and project notes
- `viewer/`: the runnable animation viewer and implementation stubs
- `assets/blocky/`: the block character assets
- `assets/smpl/`: local SMPL model files, if installed
- `libraries/`: local pose and motion libraries created in the viewer
- `exports/`: local rendered videos

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

This opens the local viewer page in your browser automatically.
To use a different port, for example when another viewer is already running:

```bash
python viewer/asset_viewer.py --port 8090
```

## Current Approach

- Keep the runnable code stable under `viewer/`
- Put teaching documents under `docs/`
- Use the same viewer framework to bridge FK, skinning, motion preview, and
  final-project evidence export
