import geopandas as gpd
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import shapely
import numpy as np
from fiona.crs import from_epsg
import math
from multiprocessing import Process, Queue, cpu_count
import queue


def get_lat_long_decimal_of_cell_by_meter(meter_distance: int, startpoint_lat_long_tuple: tuple):
    earth_radius = 6378137  # earth radius, sphere

    # Coordinate offsets in radians
    new_lat_radians = meter_distance / earth_radius
    new_long_radians = meter_distance / (earth_radius * math.cos(math.pi * startpoint_lat_long_tuple[0] / 180))

    # OffsetPosition, decimal degrees
    new_lat_decimal = startpoint_lat_long_tuple[0] + new_lat_radians * 180 / math.pi
    new_long_decimal = startpoint_lat_long_tuple[1] + new_long_radians * 180 / math.pi

    # calc difference between startpoint lat/long value and cell width [meter_distance] meter
    lat_difference = startpoint_lat_long_tuple[0] - new_lat_decimal
    long_difference = startpoint_lat_long_tuple[1] - new_long_decimal

    return abs(lat_difference), abs(long_difference)  # latitude, longitude


def which_row_cells_within_area_boundaries(area, r, cell_height, c, cell_width):
    grid_row = []
    for x0 in c:
        x1 = x0 - cell_width
        y1 = r + cell_height
        new_cell = shapely.geometry.box(x0, r, x1, y1)
        if new_cell.within(area):
            grid_row.append(new_cell)
        else:
            pass
    return grid_row


def worker(input_queue, output_queue):
    for idx, func, args in iter(input_queue.get, 'STOP'):
        result = func(*args)
        output_queue.put([idx, result])


def multi_processing_geometry_boundary_check(selected_area, selected_gdf):
    xmin, ymin, xmax, ymax = selected_gdf.total_bounds

    # cell tile is one meter in lat/long, use centroid of area x, y values from geopandas for reference
    # latitude (x diff), longitude (y diff)
    cell_height, cell_width = get_lat_long_decimal_of_cell_by_meter(1, (selected_area.centroid.y, selected_area.centroid.x))

    rows = np.arange(ymin, ymax + cell_height, cell_height)
    columns = np.arange(xmin, xmax + cell_width, cell_width)

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    num_of_processes = cpu_count() - 1
    grid_cells_unsorted = []
    for idx, row in enumerate(rows):
        one_task = [idx, which_row_cells_within_area_boundaries, (selected_area, row, cell_height, columns, cell_width)]
        task_queue.put(one_task)

    # Start worker processes
    for i in range(num_of_processes):
        Process(target=worker, args=(task_queue, done_queue)).start()

    for row in tqdm(rows):
        try:
            grid_cells_unsorted.append(done_queue.get())
        except queue.Empty as e:
            print(e)
        except queue.Full as e:
            print(e)

    # Tell child processes to stop
    for i in range(num_of_processes):
        task_queue.put('STOP')

    grid_cells_sorted = sorted(grid_cells_unsorted, key=lambda x: x[0])  # sort by index, first entry

    grid = []
    for i in grid_cells_sorted:  # extend grid with Polygon Lists  keep relevant ndarrays
        if len(i[1]) > 0:
            grid.extend(i[1])

    return grid


def serial_processing_geometry_boundary_check(selected_area, selected_gdf):
    xmin, ymin, xmax, ymax = selected_gdf.total_bounds

    # cell tile is one meter in lat/long, use centroid of area x, y values from geopandas for reference
    # latitude (x diff), longitude (y diff)
    cell_height, cell_width = get_lat_long_decimal_of_cell_by_meter(1, (selected_area.centroid.y, selected_area.centroid.x))

    grid = []
    for x0 in tqdm(np.arange(xmin, xmax + cell_width, cell_width)):
        for y0 in np.arange(ymin, ymax + cell_height, cell_height):
            x1 = x0 - cell_width
            y1 = y0 + cell_height
            new_cell = shapely.geometry.box(x0, y0, x1, y1)
            if new_cell.within(selected_area):
                grid.append(new_cell)
            else:
                pass
    return grid


if __name__ == '__main__':

    talsperre_geojsonfile = Path("Talsperren_einzeln_geojson_files", "Talsperre Malter.geojson")
    gdf_talsperre = gpd.read_file(talsperre_geojsonfile)

    talsperre_gdf_exploded = gdf_talsperre.geometry.explode().tolist()
    max_area_talsperre = max(talsperre_gdf_exploded, key=lambda a: a.area)

    trigger_multiprocessing = True

    if trigger_multiprocessing:
        grid_cells = multi_processing_geometry_boundary_check(max_area_talsperre, gdf_talsperre)

    else:
        grid_cells = serial_processing_geometry_boundary_check(max_area_talsperre, gdf_talsperre)

    # plot talsperre
    fig, ax = plt.subplots(figsize=(16, 16))

    gdf_talsperre.plot(ax=ax, markersize=.1, figsize=(16, 16), cmap='jet')
    cell_df = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=from_epsg(4326))
    cell_df.plot(ax=ax, facecolor="none", edgecolor='grey')

    plt.show()

    pass
