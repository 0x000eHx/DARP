import sys
from gridding_helpers import check_edge_length_polygon_threshold, check_real_start_points, find_grid, get_biggest_area_polygon
import time
from setting_helpers import load_yaml_config_file, write_yaml_config_file


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

    if check_edge_length_polygon_threshold(settings['grid_edge_length_meter'], settings['polygon_threshold']):

        area_polygon = get_biggest_area_polygon(settings['geojson_file_name'])

        # find biggest grid of highest value in grid_edge_length_meter
        grid_gdf = find_grid(area_polygon,
                             settings['grid_edge_length_meter'],
                             settings['polygon_threshold'])

        # save best results
        file_name = generate_file_name(settings['geojson_file_name'])
        grid_gdf.to_file(filename=f'./geodataframes/{file_name}.geojson', driver="GeoJSON")

    else:
        print("check_edge_length_polygon_threshold() or check_real_start_points() failed!")
        sys.exit(13)

    print("Successfully finished grid generation and saved geometry to file!\n", f'./geodataframes/{file_name}.geojson')
    sys.exit(0)
