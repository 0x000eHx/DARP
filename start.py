import sys
from MultiRobotPathPlanner import MultiRobotPathPlanner
from gridding import check_edge_length, check_real_start_points, find_grid
import geopandas as gpd
import numpy as np
import yaml
import time


def write_yaml_config_file(str_filepath):
    start_settings = {'geojson_file_name': 'Talsperre Malter.geojson',
                      'grid_edge_length_meter': [3, 6, 12],  # edge lengths in meter (can contain one or more values)
                      'real_start_points': [
                          [13.653522254079629, 50.92603465830493],  # long, lat values; shapely x, y values
                          [13.6500293945372, 50.91945111878728],
                          [13.654725066304804, 50.921186253206045],
                          [13.664545589728833, 50.907868824583616]
                      ],
                      'polygon_threshold': 15,  # group of polygons with number below this value will be consider irrelevant
                      'max_distance_per_robot': 10000,  # in meter
                      'trigger_image_export_final_assignment_matrix': True,
                      'trigger_video_export_assignment_matrix_changes': True,
                      'darp_max_iter': 100000,
                      'darp_dynamic_tiles_threshold': 500,
                      'darp_cc_variation': 0.01,
                      'darp_random_level': 0.0001,
                      'darp_random_seed_value': 1234,
                      'darp_trigger_importance': False
                      }

    with open(str_filepath, 'w') as f:
        yaml.dump(start_settings, f, sort_keys=False, default_flow_style=False)


def load_yaml_config_file(str_filepath):
    # load the settings
    with open(str_filepath, 'r') as f:
        data = yaml.safe_load(f)
        return data


def generate_file_name(filename: str):
    export_file_name = f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_{str(filename)}'
    # Replace all characters in dict
    b = {' ': '', '.geojson': ''}
    for x, y in b.items():
        export_file_name = export_file_name.replace(x, y)
    return export_file_name


if __name__ == '__main__':
    settings_yaml_filepath = './settings/settings_talsperre_malter.yaml'
    write_yaml_config_file(settings_yaml_filepath)
    settings = load_yaml_config_file(settings_yaml_filepath)

    if check_edge_length(settings['grid_edge_length_meter']) and check_real_start_points(settings['geojson_file_name'],
                                                                                         settings['real_start_points']):
        # find biggest grid of highest value in grid_edge_length_meter
        grid_gdf = find_grid(str(settings['geojson_file_name']), settings['grid_edge_length_meter'], settings['polygon_threshold'])

        # save candidate for best result
        if grid_gdf is not None:
            file_name = generate_file_name(settings['geojson_file_name'])

            grid_gdf.to_file(filename=file_name + ".geojson", driver="GeoJSON")

    else:
        print("check_edge_length() or check_real_start_points() failed!")
        sys.exit(13)

    # search for smaller edge length tiles around the area we just concluded to cover the biggest area



    sys.exit(0)
