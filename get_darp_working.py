import sys
import geopandas as gpd
from pathlib import Path
import webbrowser
import os
import time
from gridding_helpers import get_biggest_area_polygon, check_real_start_points, get_long_lat_diff
from path_planning_pre_calculation import generate_numpy_contour_array, get_random_start_points_list, \
    generate_stc_geodataframe
from setting_helpers import load_yaml_config_file
from MultiRobotPathPlanner import MultiRobotPathPlanner
import folium
import pandas
import numpy as np
from shapely.ops import unary_union, linemerge
from shapely.validation import make_valid

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def generate_file_name(filename: str):
    f_name = f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_{str(filename)}'
    # Replace all characters in dict
    b = {' ': '', '.geojson': ''}
    for x, y in b.items():
        f_name = f_name.replace(x, y)
    return f_name


def newest_grid_file_in_folder(path_to_folder):
    search_pattern = ['grid']
    files_in_folder = os.listdir(Path(path_to_folder).resolve())
    files = [Path(path_to_folder, nm).resolve() for ps in search_pattern for nm in files_in_folder if ps in nm]
    last_file_with_suffix = max(files, key=os.path.getctime)
    return Path(last_file_with_suffix).name.removesuffix('.geojson')


if __name__ == '__main__':

    settings = load_yaml_config_file('./settings/settings_talsperre_malter.yaml')

    last_file_no_suffix = newest_grid_file_in_folder(Path('geodataframes'))
    grid_gdf = gpd.read_file(filename=f'./geodataframes/{last_file_no_suffix}.geojson')
    print(f'Loaded grid file: ./geodataframes/{last_file_no_suffix}.geojson')

    list_real_start_points_coords = settings['real_start_points']
    export_file_name = generate_file_name(settings['geojson_file_name'])

    if check_real_start_points(settings['geojson_file_name'], settings['real_start_points']):

        measure_start = time.time()

        gdf_subcells_and_lines_collection = gpd.GeoDataFrame()
        gdf_path_per_multipoly = gpd.GeoDataFrame()

        for idx, geoserie in grid_gdf.iterrows():

            dict_tile_data = {"tile_width": geoserie.tile_width,
                              "tile_height": geoserie.tile_height}

            # post gridding numpy contour bool array generation
            np_bool_array, gdf_numpy_positions = generate_numpy_contour_array(geoserie.geometry, dict_tile_data)
            relevant_tiles_count = np.count_nonzero(np_bool_array)

            # TODO: search for start points within given area array
            start_points = get_random_start_points_list(5, np_bool_array)
            dict_darp_startparameters = {}
            for i, point_tuple in enumerate(start_points):
                dict_darp_startparameters[i] = {'row': point_tuple[0],
                                                'col': point_tuple[1],
                                                'tiles_count': 150}

            # settings['darp_random_seed_value']
            handle = MultiRobotPathPlanner(np_bool_array, settings['darp_max_iter'], settings['darp_cc_variation'],
                                           settings['darp_random_level'], settings['darp_dynamic_tiles_threshold'],
                                           dict_darp_startparameters, settings['darp_random_seed_value'],
                                           settings['darp_trigger_importance'], False,
                                           settings['trigger_image_export_final_assignment_matrix'],
                                           settings['trigger_video_export_assignment_matrix_changes'],
                                           f'{export_file_name}_{str(geoserie.tiles_group_identifier)}')  # TODO a real name for every grid of tile_size x
            if handle.darp_success:
                gdf_path_one_multipoly = generate_stc_geodataframe(gdf_numpy_positions, handle.darp_instance.A,
                                                                   handle.best_case.paths,
                                                                   geoserie.tiles_group_identifier)

                # filter for lines only and try to unify them
                list_all_startpoints = gdf_path_one_multipoly[gdf_path_one_multipoly['line']].assigned_startpoint.unique()

                for idx_point, startpoint in enumerate(list_all_startpoints):
                    temp = gdf_path_one_multipoly.query(f'line == True and assigned_startpoint == {startpoint}')
                    path_multilinestring = linemerge(temp.geometry.to_list())
                    # path_multilinestring = make_valid(unary_union(merged_lines))
                    print("Unified path lines!")

                    data = [{'tiles_group_identifier': str(geoserie.tiles_group_identifier),
                             'assigned_startpoint': startpoint,
                             'sensor_line_length_meter': geoserie.sensor_line_length_meter,
                             'path_length_meter': path_multilinestring.length,
                             'geometry': path_multilinestring}]

                    gdf_one_multipoly_path = gpd.GeoDataFrame(data, crs=4326).set_geometry('geometry')
                    gdf_path_per_multipoly = gpd.GeoDataFrame(pandas.concat([gdf_path_per_multipoly,
                                                                             gdf_one_multipoly_path], axis=0,
                                                                            ignore_index=True),
                                                              crs=4326)
                # finally append everything
                gdf_subcells_and_lines_collection = gpd.GeoDataFrame(pandas.concat([gdf_subcells_and_lines_collection,
                                                                                    gdf_path_one_multipoly], axis=0,
                                                                                   ignore_index=True),
                                                                     crs=4326)

        # save results to file before drawing
        gdf_subcells_and_lines_collection.to_file(
            filename=f'./geodataframes/{export_file_name}_subcells_and_lines_collection.geojson', driver="GeoJSON")
        gdf_path_per_multipoly.to_file(filename=f'./geodataframes/{export_file_name}_path_per_tilegroup.geojson',
                                       driver="GeoJSON")

        measure_end = time.time()
        print("Elapsed time path generation (with darp): ", str((measure_end - measure_start) / 60), "min")

        sys.exit(0)

    else:
        print("start points don't match the given lake area")

        sys.exit(1)
