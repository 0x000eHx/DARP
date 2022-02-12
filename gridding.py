import geopandas as gpd
from pathlib import Path
from tqdm.auto import tqdm
import shapely
import numpy as np
import math
from multiprocessing import Process, Queue, cpu_count
import queue
from numba import njit, jit, prange


def get_long_diff_in_meter(grid_size_meter: float, startpoint_lat_long_tuple: tuple):

    earth_radius = 6378137  # earth radius, sphere

    # Coordinate offsets in radians
    new_lat_radians = grid_size_meter / earth_radius
    new_long_radians = grid_size_meter / (earth_radius * math.cos(math.pi * startpoint_lat_long_tuple[0] / 180))

    # OffsetPosition, decimal degrees
    new_lat_decimal = startpoint_lat_long_tuple[0] + new_lat_radians * 180 / math.pi
    new_long_decimal = startpoint_lat_long_tuple[1] + new_long_radians * 180 / math.pi

    # calc difference between startpoint lat/long value and cell width [grid_size_meter] meter
    lat_difference = startpoint_lat_long_tuple[0] - new_lat_decimal
    long_difference = startpoint_lat_long_tuple[1] - new_long_decimal

    return abs(lat_difference), abs(long_difference)  # latitude, longitude


def which_row_cells_within_area_boundaries(area, r, cell_height, c, cell_width):

    grid_row = np.full(len(c), False)
    for idx, x0 in enumerate(c):
        x1 = x0 - cell_width
        y1 = r + cell_height
        new_cell = shapely.geometry.box(x0, r, x1, y1)
        if new_cell.within(area):
            grid_row[idx] = True
        else:
            pass
    return grid_row


def worker(input_queue, output_queue):
    for idx, func, args in iter(input_queue.get, 'STOP'):
        result = func(*args)
        output_queue.put([idx, result])


def processing_geometry_boundary_check(grid_size, selected_area, selected_gdf, multiprocessing=True):

    xmin, ymin, xmax, ymax = selected_gdf.total_bounds

    # cell tile is one meter in lat/long, use centroid of area x, y values from geopandas for reference
    # latitude (x diff), longitude (y diff)
    cell_height, cell_width = get_long_diff_in_meter(grid_size, (selected_area.centroid.y, selected_area.centroid.x))

    rows = sorted(np.arange(ymin, ymax + cell_height, cell_height), reverse=True)  # scan from top to bottom
    columns = np.arange(xmin, xmax + cell_width, cell_width)  # scan from left to right

    if multiprocessing:
        # Create queues
        task_queue = Queue()
        done_queue = Queue()

        num_of_processes = cpu_count() - 1
        grid_cells_unsorted = []

        # create tasks and push them into queue
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

        grid = np.array([i[1] for i in grid_cells_sorted])  # np.array easily creates 1 axe array out of list

    else:
        # serial processing, cause time doesn't matter
        grid = np.full((len(rows), len(columns)), False)
        for xidx, x0 in enumerate(tqdm(columns)):
            for yidx, y0 in enumerate(rows):
                x1 = x0 - cell_width
                y1 = y0 + cell_height
                new_cell = shapely.geometry.box(x0, y0, x1, y1)
                if new_cell.within(selected_area):
                    grid[yidx][xidx] = True
                else:
                    pass
    return grid


@jit()  # @njit(parallel=True)
def numba_gridding(grid_size, selected_area, selected_gdf):

    xmin, ymin, xmax, ymax = selected_gdf.total_bounds

    # cell tile is one meter in lat/long, use centroid of area x, y values from geopandas for reference
    # latitude (x diff), longitude (y diff)
    cell_height, cell_width = get_long_diff_in_meter(grid_size, (selected_area.centroid.y, selected_area.centroid.x))

    rows = sorted(np.arange(ymin, ymax + cell_height, cell_height), reverse=True)  # scan from top to bottom
    columns = np.arange(xmin, xmax + cell_width, cell_width)  # scan from left to right

    grid = np.full((len(rows), len(columns)), False, dtype=bool)

    xid = 0
    for x0 in prange(columns):
        yid = 0
        for y0 in prange(rows):
            x1 = x0 - cell_width
            y1 = y0 + cell_height
            new_cell = shapely.geometry.box(x0, y0, x1, y1)
            if new_cell.within(selected_area):
                grid[yid][xid] = True
            else:
                pass
            yid += yid
        xid += xid

    return grid


def get_grid_array(dam_file_name: str, grid_size_meter: float, multiprocessing=True):

    dam_geojson_filepath = Path("dams_single_geojsons", dam_file_name)
    gdf_dam = gpd.read_file(dam_geojson_filepath)

    gdf_dam_exploded = gdf_dam.geometry.explode().tolist()
    dam_biggest_water_area = max(gdf_dam_exploded, key=lambda a: a.area)

    # TODO check grid_size_meter for appropriate value
    if multiprocessing:
        grid = processing_geometry_boundary_check(grid_size_meter, dam_biggest_water_area, gdf_dam, multiprocessing=multiprocessing)
        # numba_gridding(grid_size_meter, dam_biggest_water_area, gdf_dam)
    else:
        grid = processing_geometry_boundary_check(grid_size_meter, dam_biggest_water_area, gdf_dam, multiprocessing=multiprocessing)

    return grid

