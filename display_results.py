import sys
from pathlib import Path
import os
import pandas
from tqdm.auto import tqdm
import geopandas as gpd
import pandas as pd
import numpy as np
import math
from multiprocessing import Process, Queue, cpu_count
import queue
from shapely.geometry import Point, box, MultiPolygon, Polygon, GeometryCollection, shape
import shapely
from shapely.ops import unary_union
from shapely.validation import make_valid
import webbrowser
import matplotlib.pyplot as plt
from setting_helpers import load_yaml_config_file
import folium


def search_newest_file_in_folder(path_to_folder, unique_search_string):
    search_pattern = [unique_search_string]
    files_in_folder = os.listdir(Path(path_to_folder).resolve())
    files = [Path(path_to_folder, nm).resolve() for ps in search_pattern for nm in files_in_folder if ps in nm]
    last_file_with_suffix = max(files, key=os.path.getctime)
    return Path(last_file_with_suffix).name.removesuffix('.geojson')


if __name__ == '__main__':
    settings = load_yaml_config_file('./settings/settings_talsperre_malter.yaml')

    path_per_tilegroup_file = search_newest_file_in_folder(Path('geodataframes'), 'path_per_tilegroup')
    paths_gdf = gpd.read_file(filename=f'./geodataframes/{path_per_tilegroup_file}.geojson')

    subcells_and_lines_collection_file = search_newest_file_in_folder(Path('geodataframes'), 'subcells_and_lines_collection')
    subcells_and_lines_collection_gdf = gpd.read_file(filename=f'./geodataframes/{subcells_and_lines_collection_file}.geojson')

    paths_gdf.set_geometry('geometry').set_crs(crs=4326)
    subcells_and_lines_collection_gdf.set_geometry('geometry').set_crs(crs=4326)

    # draw loaded map in browser and save
    paths_map = paths_gdf.explore('path_length_meter', scheme='quantiles', cmap='jet')  # YlGn,jet,Spectral, PuBu, legend=True, scheme='quantiles'
    subcells_map = subcells_and_lines_collection_gdf.explore(column='tiles_group_identifier', cmap='Spectral')

    for sp in settings['real_start_points']:
        folium.Marker([sp[1], sp[0]], popup="<i>Startpoint</i>").add_to(paths_map)
    paths_map.save(f'htmls/paths_only.html')
    path = 'file:///' + os.getcwd() + '/htmls/paths_only.html'
    webbrowser.open(path)

    for sp in settings['real_start_points']:
        folium.Marker([sp[1], sp[0]], popup="<i>Startpoint</i>").add_to(subcells_map)
    subcells_map.save(f'htmls/subcells_with_paths.html')
    path = 'file:///' + os.getcwd() + '/htmls/subcells_with_paths.html'
    webbrowser.open(path)

    # draw loaded map in browser and save
    # fol_map = grid_gdf.explore('covered_area', cmap='Spectral')  # YlGn,jet, PuBu, legend=True, scheme='quantiles'
    # for sp in settings['real_start_points']:
    #     folium.Marker([sp[1], sp[0]], popup="<i>Startpoint</i>").add_to(fol_map)
    # fol_map.save(f'htmls/{last_file_no_suffix}.html')
    # path = f'file:///{os.getcwd()}/htmls/{last_file_no_suffix}.html'
    # webbrowser.open(path)

    sys.exit(0)
