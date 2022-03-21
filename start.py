import sys
from MultiRobotPathPlanner import MultiRobotPathPlanner
from gridding import check_edge_length, check_real_start_points, find_biggest_grid
import numpy as np
import yaml


# from yaml.loader import SafeLoader


def write_yaml_config_file(str_filepath):
    start_settings = {'dam_file_name': 'Talsperre Malter.geojson',
                      'grid_edge_length_meter': [3, 6, 12],  # edge lengths in meter (can contain one or more values)
                      'max_iter': 100000,
                      'cc_variation': 0.01,
                      'random_level': 0.0001,
                      'random_seed_value': 1234,
                      'dynamic_tiles_threshold': 500,
                      'max_tiles_per_robot': 10000,
                      'trigger_importance': False,
                      'trigger_pygame_visualisation': False,
                      'trigger_image_export_final_assignment_matrix': True,
                      'trigger_video_export_assignment_matrix_changes': True,
                      'real_start_points': [
                          [13.653522254079629, 50.92603465830493],  # long, lat values
                          [13.6500293945372, 50.91945111878728],    # geopandas/shapely (Point) x, y value
                          [13.654725066304804, 50.921186253206045],
                          [13.664545589728833, 50.907868824583616]
                      ]
                      }

    with open(str_filepath, 'w') as f:
        yaml.dump(start_settings, f, sort_keys=False, default_flow_style=False)


def load_yaml_config_file(str_filepath):
    # load the settings
    with open(str_filepath, 'r') as f:
        data = yaml.safe_load(f)
        return data


if __name__ == '__main__':
    settings_yaml_filepath = './settings/settings_talsperre_malter.yaml'
    # write_yaml_config_file(settings_yaml_filepath)
    settings = load_yaml_config_file(settings_yaml_filepath)

    if check_edge_length(settings['grid_edge_length_meter']) and check_real_start_points(settings['dam_file_name'], settings['real_start_points']):
        # find biggest grid of highest value in grid_edge_length_meter
        list_of_generated_files = find_biggest_grid(str(settings['dam_file_name']), settings['grid_edge_length_meter'])
    else:
        print("something went terribly wrong!")
        sys.exit(13)

    # grid_bool = get_grid_array(dam_file_name, grid_edge_length_meter, multiprocessing=True)

    # start_points = [(600, 338), (547, 298), (527, 370), (446, 324), (323, 244), (643, 410)]
    # get_random_start_points_list(3, grid_bool)
    # [(359, 114), (416, 37), (216, 178)] and [0.4, 0.3, 0.3] -> overflow maxiter
    # [(269, 158), (529, 281), (564, 304)] and portions [0.4, 0.3, 0.3] --> good ones
    # [(230, 180), (243, 178), (212, 176)] damned start points, portions [0.3, 0.2, 0.5]
    # [(63, 217), (113, 195), (722, 326)] good ones with 20k tiles
    # [(60, 244), (237, 185), (651, 464), (678, 378), (667, 412)]
    # [(166, 212), (334, 157), (587, 337), (251, 301), (550, 327), (247, 258)]
    # [(600, 338), (547, 298), (527, 370), (446, 324), (323, 244), (643, 410)] with 10000 takes long
    # [(727, 533), (587, 391), (206, 176)] at 20000 tiles take a long time

    # MultiRobotPathPlanner(grid_bool, np.uintc(max_iter), CCvariation, randomLevel, np.uintc(dcells),
    #                       max_tiles_per_robot, seed_value, importance, start_points, visualize,
    #                       image_export_final_assignment_matrix, dam_file_name, video_export_assignment_matrix_changes)

    sys.exit(0)
