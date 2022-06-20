import sys
from gridding_helpers import generate_file_name, generate_grid, get_biggest_area_polygon, Grid_Generation_Task_Manager
import time
from setting_helpers import load_yaml_config_file, write_yaml_config_file


if __name__ == '__main__':
    settings_yaml_filepath = './settings/settings_talsperre_malter.yaml'
    write_yaml_config_file(settings_yaml_filepath)
    settings = load_yaml_config_file(settings_yaml_filepath)

    # find the Shapely Geometry (Multipolygon) of interest
    area_polygon = get_biggest_area_polygon(settings['geojson_file_name'])

    # create a task manager
    task_manager = Grid_Generation_Task_Manager(settings['sensor_line_length_meter'], settings['polygon_threshold'], area_polygon, settings['geojson_file_name'])
    # Grid generation will be generic without any specification
    # the widest scanner line will get used to create the biggest grid
    # use the Notebook if you wanna specify the regions per scanner line width

    measure_start = time.time()

    # find biggest grid of highest value in sensor_line_length_meter
    grid_gdf = generate_grid(task_manager)

    if not grid_gdf.empty:
        # save best results
        file_name = generate_file_name(settings['geojson_file_name'])
        grid_gdf.to_file(filename=f'./geodataframes/{file_name}_grid.geojson', driver="GeoJSON")
        print("Successfully finished grid generation and saved geometry to file!\n",
              f'./geodataframes/{file_name}_grid.geojson')

    measure_end = time.time()
    print("Elapsed time grid generation: ", (measure_end - measure_start), "sec")

    sys.exit(0)
