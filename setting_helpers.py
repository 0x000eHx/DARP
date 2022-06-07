import yaml


def write_yaml_config_file(str_filepath):
    start_settings = {'geojson_file_name': 'Talsperre Malter.geojson',
                      'sensor_line_length_meter': [5, 15],  # scanner line lengths in meter for grid generation (can contain one or more values)
                      'real_start_points': [
                          [13.653522254079629, 50.92603465830493],  # shapely x (longitude), y (latitude) values
                          [13.6500293945372, 50.91945111878728],    # position in this list will be priority
                          [13.654725066304804, 50.921186253206045], # top coords get chosen first
                          [13.664545589728833, 50.907868824583616]
                      ],
                      'polygon_threshold': [5, 4],  # always keep the same count of numbers here as in sensor_line_length_meter
                      # polygon groups with given number below this value will be considered irrelevant
                      # index is equivalent to index of edge length
                      'max_distance_per_task': 10000,  # in meter
                      'trigger_image_export_final_assignment_matrix': False,  # recommended only for debugging purposes
                      'trigger_video_export_assignment_matrix_changes': True,  # recommended only for debugging purposes
                      'darp_max_iter': 100000,
                      'darp_dynamic_tiles_threshold': 500,  # if darp hits iter max, will increase by 10 until this threshold reached
                      'darp_cc_variation': 0.01,
                      'darp_random_level': 0.0001,
                      'darp_random_seed_value': None,
                      'darp_trigger_importance': False
                      }

    with open(str_filepath, 'w') as f:
        yaml.dump(start_settings, f, sort_keys=False, default_flow_style=False)


def load_yaml_config_file(str_filepath):
    # load the settings
    with open(str_filepath, 'r') as f:
        data = yaml.safe_load(f)
        return data
