import sys
import geopandas as gpd
from pathlib import Path
import webbrowser
import os
import time
from gridding_helpers import get_biggest_area_polygon, check_real_start_points, get_long_lat_diff
from path_planning_pre_calculation import generate_numpy_contour_array, get_random_start_points_list, generate_stc_geodataframe
from setting_helpers import load_yaml_config_file
from MultiRobotPathPlanner import MultiRobotPathPlanner
import folium
import numpy as np
from shapely.ops import unary_union
from shapely.validation import make_valid


def generate_file_name(filename: str):
    f_name = f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_{str(filename)}'
    # Replace all characters in dict
    b = {' ': '', '.geojson': ''}
    for x, y in b.items():
        f_name = f_name.replace(x, y)
    return f_name


def newest_file_in_folder(path_to_folder):
    files = []
    for one_file in Path(path_to_folder).iterdir():
        if one_file.suffix == '.geojson':
            files.append(one_file)
    last_file_with_suffix = max(files, key=os.path.getctime)
    return Path(last_file_with_suffix).name.removesuffix('.geojson')


if __name__ == '__main__':

    settings = load_yaml_config_file('./settings/settings_talsperre_malter.yaml')

    last_file_no_suffix = newest_file_in_folder(Path('geodataframes'))
    grid_gdf = gpd.read_file(filename=f'./geodataframes/{last_file_no_suffix}.geojson')

    # draw loaded map in browser and save
    # fol_map = grid_gdf.explore('covered_area', cmap='Spectral')  # YlGn,jet, PuBu, legend=True, scheme='quantiles'
    # for sp in settings['real_start_points']:
    #     folium.Marker([sp[1], sp[0]], popup="<i>Startpoint</i>").add_to(fol_map)
    # fol_map.save(f'htmls/{last_file_no_suffix}.html')
    # path = 'file:///' + os.getcwd() + '/htmls/' + last_file_no_suffix + '.html'
    # webbrowser.open(path)

    if check_real_start_points(settings['geojson_file_name'], settings['real_start_points']):

        # load series of multipolygons with specific tile size out of geodataframe
        different_area_sizes = grid_gdf.covered_area.unique()
        max_val = max(different_area_sizes)
        biggest_area_series = grid_gdf.loc[grid_gdf['covered_area'] == max_val]

        dict_tile_data = {"tile_width": biggest_area_series.tile_width.values[0],
                          "tile_height": biggest_area_series.tile_height.values[0]}
        dict_offset = {'offset_longitude': biggest_area_series.offset_longitude.values[0],
                       'offset_latitude': biggest_area_series.offset_latitude.values[0]}

        # transform max_distance_per_robot into max_tiles_per_robot for DARP
        list_start_point_coords = settings['real_start_points']
        area_polygon = get_biggest_area_polygon(settings['geojson_file_name'])

        w_diff, h_diff = get_long_lat_diff(settings['max_distance_per_robot'], area_polygon.centroid.y)
        # w_diff / settings['max_distance_per_robot']
        max_tiles_per_robot = settings['max_distance_per_robot']  # TODO

        export_file_name = generate_file_name(settings['geojson_file_name'])

        for multipoly in biggest_area_series.geometry.to_list():

            #######################################
            # bla_gdf = gpd.GeoDataFrame(geometry=[make_valid(unary_union(multipoly))], crs=4326).set_geometry('geometry')
            # biggest_map = bla_gdf.explore(cmap='Spectral')
            # for sp in settings['real_start_points']:
            #     folium.Marker([sp[1], sp[0]], popup="<i>Startpoint</i>").add_to(biggest_map)
            # biggest_map.save(f'htmls/biggest_multipoly_second.html')
            # path = 'file:///' + os.getcwd() + '/htmls/biggest_multipoly_second.html'
            # webbrowser.open(path)
            ########################################

            # post gridding numpy contour bool array generation
            np_bool_array, gdf_numpy_positions = generate_numpy_contour_array(multipoly, dict_tile_data)
            relevant_tiles_count = np.count_nonzero(np_bool_array) - len(list_start_point_coords)

            # TODO: search for start points within given area array
            start_points = get_random_start_points_list(5, np_bool_array)
            dict_darp_startparameters = {0: {'row': start_points[0][0],
                                             'col': start_points[0][1],
                                             'tiles_count': 500},
                                         1: {'row': start_points[1][0],
                                             'col': start_points[1][1],
                                             'tiles_count': 500},
                                         2: {'row': start_points[2][0],
                                             'col': start_points[2][1],
                                             'tiles_count': 500},
                                         3: {'row': start_points[3][0],
                                             'col': start_points[3][1],
                                             'tiles_count': 500},
                                         4: {'row': start_points[4][0],
                                             'col': start_points[4][1],
                                             'tiles_count': 500}}

            # settings['darp_random_seed_value']
            handle = MultiRobotPathPlanner(np_bool_array, settings['darp_max_iter'], settings['darp_cc_variation'],
                                           settings['darp_random_level'], settings['darp_dynamic_tiles_threshold'],
                                           dict_darp_startparameters, settings['darp_random_seed_value'],
                                           settings['darp_trigger_importance'], False,
                                           settings['trigger_image_export_final_assignment_matrix'],
                                           settings['trigger_video_export_assignment_matrix_changes'],
                                           export_file_name)  # TODO a real name for every grid of tile_size x
            if handle.darp_success:
                gdf_path = generate_stc_geodataframe(gdf_numpy_positions, handle.darp_instance.A, handle.best_case.paths)

                # draw loaded map in browser and save
                fol_map = gdf_path.explore('assigned_drone', cmap='Spectral')  # YlGn,jet, PuBu, legend=True, scheme='quantiles'

                for sp in settings['real_start_points']:
                    folium.Marker([sp[1], sp[0]], popup="<i>Startpoint</i>").add_to(fol_map)
                fol_map.save(f'htmls/subcells_with_paths.html')
                path = 'file:///' + os.getcwd() + '/htmls/subcells_with_paths.html'
                webbrowser.open(path)
                pass

    else:
        print("start points don't match the given lake area")

    sys.exit(0)
