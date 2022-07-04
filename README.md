# (Offline) Lake Bathymetry Scanning
This project aims to divide a lake area into several regions, optimize the region size and shape and use path planning to calculate a WGS 84 conform path. This is one step of my universities [RoBiMo project](https://tu-freiberg.de/en/robimo) to automatically scan the subsurface of a lake by a [boat drone](https://www.youtube.com/watch?v=qbDoeSv3BPs).
See an example scan in 3D at [sketchfab.com](https://sketchfab.com/3d-models/riesenstein-scientific-diving-center-freiberg-5f30ea70c20e447eb5e121b51e5ae3f7)!

### Motivation & Conditions:

A small boat drone with a bathymetric scanner can only move a certain distance until its battery is empty and needs a refill.
Dividing the lake into regions with a defined number of tiles is one step. Rearranging the grid around every drone's start point by the DARP algorithm in used to find an optimal solution considered the distance of every tile inside the lake area.

After finding the optimal regions a path planning algorithm has to find a way with the lowest number of turns and the highest number of the longest possible line segments. This way through every region will be exportable as [WGS 84 (EPSG:4326)](https://en.wikipedia.org/wiki/World_Geodetic_System) path for usage in automatic path finding programs.  


### DARP: Divide Areas Algorithm for Optimal Multi-Robot Coverage Path Planning

This is a fork of the [DARP Python Project](https://github.com/alice-st/DARP) with its Java source the original [DARP Java Project](https://github.com/athakapo/DARP).

Look up the original project for further details, how the algorithm works and all links.

## Current Status

This README needs work.
Will extend this doc to the current development state soon...


In the meantime:
Install the environment via Anaconda (Conda) or Mamba (conda_environment.yaml). 

Try using the "start_grid_generation_notebook" Jupyter Notebook and draw regions (as Polygons) inside a area of interest.
Start the grid generation and get the Spanning Tree Coverage (STC) Tiles.

If you don't need the notebook start the get_grid.py file and read a lake area from a geojson file (provided in the "dams_single_geojsons" folder).

After that you can use the get_darp_working.py file to calculate all paths for all the grid tile groups.

For displaying the results start the display_results.py script. It opens all calculated steps and the end result as HTML file in your browser.
