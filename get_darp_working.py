import sys
import geopandas as gpd
from pathlib import Path
import webbrowser
import os
import time
from gridding_helpers import generate_numpy_contour_array, get_biggest_area_polygon, get_random_start_points_list
from setting_helpers import load_yaml_config_file
from MultiRobotPathPlanner import MultiRobotPathPlanner


def generate_file_name(filename: str):
    export_file_name = f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_{str(filename)}'
    # Replace all characters in dict
    b = {' ': '', '.geojson': ''}
    for x, y in b.items():
        export_file_name = export_file_name.replace(x, y)
    return export_file_name


def newest_file_in_folder(path_to_folder):
    files = []
    for one_file in Path(path_to_folder).iterdir():
        if one_file.suffix == '.geojson':
            files.append(one_file)
    last_file_with_suffix = max(files, key=os.path.getctime)
    return Path(last_file_with_suffix).name.removesuffix('.geojson')


if __name__ == '__main__':

    settings = load_yaml_config_file('./settings/settings_talsperre_bautzen.yaml')

    last_file_no_suffix = newest_file_in_folder(Path('geodataframes'))
    grid_gdf = gpd.read_file(filename=f'./geodataframes/{last_file_no_suffix}.geojson')

    # draw loaded map in browser and save
    # fol_map = grid_gdf.explore('covered_area', cmap='PuBu', scheme='quantiles')  # jet, legend=True
    # fol_map.save(f'htmls/{last_file_no_suffix}.html')
    # path = 'file:///' + os.getcwd() + '/htmls/' + last_file_no_suffix + '.html'
    # webbrowser.open(path)

    # load series of multipolygons with specific tile size out of geodataframe
    different_area_sizes = grid_gdf.covered_area.unique()
    max_val = max(different_area_sizes)

    biggest_areas = grid_gdf.loc[grid_gdf['covered_area'] == max_val]
    tile_width = biggest_areas.tile_width[0]
    tile_height = biggest_areas.tile_height[0]

    dict_something = {"tile_width": tile_width, "tile_height": tile_height}

    # post numpy contour bool array generation
    list_np_bool_contours = generate_numpy_contour_array(biggest_areas.geometry.to_list(),
                                                         dict_something)

    # distance calculation here, depending on the tile size

    area_polygon = get_biggest_area_polygon(settings['geojson_file_name'])
    bla = settings['real_start_points']
    export_file_name = generate_file_name(settings['geojson_file_name'])

    for np_bool_array in list_np_bool_contours:
        start_points = get_random_start_points_list(5, np_bool_array)
        handle = MultiRobotPathPlanner(np_bool_array, settings['darp_max_iter'], settings['darp_cc_variation'],
                                       settings['darp_random_level'], settings['darp_dynamic_tiles_threshold'],
                                       settings['max_distance_per_robot'], settings['darp_random_seed_value'],
                                       settings['darp_trigger_importance'], start_points,
                                       False, settings['trigger_image_export_final_assignment_matrix'],
                                       settings['trigger_video_export_assignment_matrix_changes'],
                                       export_file_name)  # TODO a real name for every grid of tile_size x

    sys.exit(0)
