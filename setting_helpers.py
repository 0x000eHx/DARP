import yaml


def write_yaml_config_file(str_filepath):
    start_settings = {'geojson_file_name': 'Talsperre Bautzen.geojson',
                      'grid_edge_length_meter': [8, 16],  # edge lengths in meter (can contain one or more values)
                      'real_start_points': [
                          # [13.653522254079629, 50.92603465830493],  # long, lat values; shapely x, y values
                          # [13.6500293945372, 50.91945111878728],
                          # [13.654725066304804, 50.921186253206045],
                          # [13.664545589728833, 50.907868824583616]
                      ],
                      'polygon_threshold': [10, 15],
                      # group of polygons with number below this value will be consider irrelevant
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
