import time
import copy
import psutil
import geopandas as gpd
import numpy as np
import pandas
from tqdm.auto import tqdm
from multiprocessing import Process, Queue
import queue
from shapely.geometry import Polygon, MultiPolygon, LineString, box, MultiLineString
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely import speedups

if speedups.available:
    speedups.enable()


def calc_path_A_to_B():
    path_length_meter = 0
    line = LineString()

    return line, path_length_meter


def get_start_points_from_coords(list_start_point_coords: list, numpy_bool_array: np.ndarray):
    list_start_points = []


def generate_stc_geodataframe(input_gdf: gpd.GeoDataFrame, assignment_matrix: np.ndarray, paths, tiles_group_identifier):
    print("Start dividing every polygon from grid generation into 4 subcells for usage in STC.")
    task_queue = Queue()
    done_queue = Queue()

    num_of_processes = 2  # psutil.cpu_count(logical=False)  # only physical available core count for this task

    # get big polygons from input_gdf and divide them into 4 parts for STC path planning
    # keep the old numpy_array cell positions alive in new subcells
    list_subcells_dicts = []
    task_counter = 0
    for idx, serie in enumerate(input_gdf.itertuples()):
        task_counter += 1
        column_idx = serie.column_idx  # directly use hashable entries from input_gdf
        row_idx = serie.row_idx
        assigned_startpoint = assignment_matrix[row_idx, column_idx]
        one_task = [idx, divide_polygon, (row_idx, column_idx, serie.geometry, assigned_startpoint, tiles_group_identifier)]
        task_queue.put(one_task)

    # Start worker processes
    for i in range(num_of_processes):
        Process(target=worker, args=(task_queue, done_queue)).start()

    for _ in tqdm(range(task_counter)):
        try:
            ix, list_of_dicts = done_queue.get()
            list_subcells_dicts.extend(list_of_dicts)

        except queue.Empty as e:
            print(e)

    # Tell child processes to stop
    for i in range(num_of_processes):
        task_queue.put('STOP')

    task_queue.close()
    done_queue.close()

    print("Start going through subcells and keep their relative position inside DARP numpy array.",
          "Creating LineStrings from subcell centroids.")
    measure_start = time.time()
    # create path from lines (centroid for centroid) and keep the assigned_startpoint for the geodataframe
    task_queue = Queue()
    done_queue = Queue()
    process_count = 1  # psutil.cpu_count(logical=False)  # only physical available core count for this task

    list_geoseries = []
    task_counter = 0
    for ix_startpoint, segment in enumerate(paths):
        for line_tuples in segment:
            one_task = [task_counter, generate_linestring_geoseries, (line_tuples, ix_startpoint, tiles_group_identifier)]
            task_queue.put(one_task)
            task_counter += 1

    # if a lot of work needs to be done then copy the original list_subcells_dicts and give every process its own
    list_of_deepcopys = [list_subcells_dicts]
    for i in range(process_count - 1):
        list_of_deepcopys.append(copy.deepcopy(list_subcells_dicts))

    # start search_worker with their own copy of list_subcells_dicts
    for i in range(process_count):
        # using search worker generates a lot of memory usage, cause of copys, so increase value with care
        Process(target=search_worker, args=(task_queue, done_queue, list_of_deepcopys[i])).start()

    for _ in tqdm(range(task_counter)):
        try:
            idx, geoseries = done_queue.get()
            if not geoseries.empty:
                list_geoseries.append(geoseries)
                # append, not extend: new geoseries is just one line with assigned start point etc

        except queue.Empty as e:
            print(e)

    # Tell child processes to stop
    for i in range(process_count):
        task_queue.put('STOP')

    task_queue.close()
    done_queue.close()
    measure_end = time.time()
    print("Measured time LineString path generation: ", (measure_end - measure_start), " sec")

    gdf_subcells = gpd.GeoDataFrame(list_subcells_dicts, crs=4326).set_geometry('geometry')
    gdf_trajectory_paths = gpd.GeoDataFrame(list_geoseries, crs=4326).set_geometry('geometry')

    gdf_collection = gpd.GeoDataFrame(pandas.concat([gdf_subcells, gdf_trajectory_paths], axis=0, ignore_index=True),
                                      crs=gdf_trajectory_paths.crs)

    return gdf_collection


def generate_linestring_geoseries(list_of_subcells_dicts, line_tuples, assigned_startpoint, tiles_group_identifier):
    # p1 row index, p1 column index, p2 row index, p2 column index = line_tuples
    p1_r_i, p1_c_i, p2_r_i, p2_c_i = line_tuples

    p1 = next((item['geometry'] for item in list_of_subcells_dicts if
               (item["column_idx"] == p1_c_i and item["row_idx"] == p1_r_i)), None)
    p2 = next((item['geometry'] for item in list_of_subcells_dicts if
               (item["column_idx"] == p2_c_i and item["row_idx"] == p2_r_i)), None)

    if p1 is not None and p2 is not None:

        data = {'tiles_group_identifier': tiles_group_identifier,
                'row_idx': np.nan,
                'column_idx': np.nan,
                'assigned_startpoint': assigned_startpoint,
                'poly': False,
                'line': True,
                'geometry': LineString([p1.centroid, p2.centroid])}
        geoseries = gpd.GeoSeries(data, crs=4326)

    else:
        geoseries = gpd.GeoSeries()

    return geoseries


def divide_polygon(row_idx, column_idx, poly: Polygon, assigned_startpoint, tiles_group_identifier):
    minx, miny, maxx, maxy = poly.bounds

    l_x = (maxx - minx) / 2
    l_y = (maxy - miny) / 2

    data = [{'tiles_group_identifier': tiles_group_identifier,
             'row_idx': row_idx * 2,
             'column_idx': column_idx * 2,
             'assigned_startpoint': assigned_startpoint,
             'poly': True,
             'line': False,
             'geometry': box(minx, miny + l_y, maxx - l_x, maxy)},
            {'tiles_group_identifier': tiles_group_identifier,
             'row_idx': row_idx * 2,
             'column_idx': column_idx * 2 + 1,
             'assigned_startpoint': assigned_startpoint,
             'poly': True,
             'line': False,
             'geometry': box(minx + l_x, miny + l_y, maxx, maxy)},
            {'tiles_group_identifier': tiles_group_identifier,
             'row_idx': row_idx * 2 + 1,
             'column_idx': column_idx * 2,
             'assigned_startpoint': assigned_startpoint,
             'poly': True,
             'line': False,
             'geometry': box(minx, miny, maxx - l_x, maxy - l_y)},
            {'tiles_group_identifier': tiles_group_identifier,
             'row_idx': row_idx * 2 + 1,
             'column_idx': column_idx * 2 + 1,
             'assigned_startpoint': assigned_startpoint,
             'poly': True,
             'line': False,
             'geometry': box(minx + l_x, miny, maxx, maxy - l_y)}]

    return data


def generate_numpy_contour_array(multipoly: MultiPolygon, dict_tile_width_height):
    print("Generate numpy contour bool_area_array from given MultiPolygon!")
    union_area = make_valid(unary_union(multipoly))
    minx, miny, maxx, maxy = union_area.bounds

    # scan columns from left to right
    columns_range = np.arange(minx, maxx + dict_tile_width_height['tile_width'], dict_tile_width_height['tile_width'])
    rows_range = np.arange(miny, maxy + dict_tile_width_height['tile_height'], dict_tile_width_height['tile_height'])
    # scan rows from top to bottom
    rows_range = np.flip(rows_range)
    np_bool_grid = np.full(shape=(rows_range.shape[0], columns_range.shape[0]), fill_value=False, dtype=bool)

    list_numpy_contour_positions_geoseries = []

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    num_of_processes = 2  # psutil.cpu_count(logical=False)  # only physical available core count for this task
    multipoly_list = list(multipoly.geoms)
    # create tasks and push them into queue
    for idx, poly in enumerate(multipoly_list):
        one_task = [idx, check_poly_pos, (rows_range, columns_range, poly.centroid.y, poly.centroid.x,
                                          dict_tile_width_height['tile_height'],
                                          dict_tile_width_height['tile_width'])]
        task_queue.put(one_task)

    # Start worker processes
    for i in range(num_of_processes):
        Process(target=worker, args=(task_queue, done_queue)).start()

    for _ in tqdm(multipoly_list):
        try:
            ix, (row_idx, col_idx) = done_queue.get()
            np_bool_grid[row_idx, col_idx] = True

            data = {'row_idx': row_idx,
                    'column_idx': col_idx,
                    'geometry': multipoly_list[ix]}
            list_numpy_contour_positions_geoseries.append(gpd.GeoSeries(data, crs=4326))

        except queue.Empty as e:
            print(e)

    # Tell child processes to stop
    for i in range(num_of_processes):
        task_queue.put('STOP')

    print("Area contour numpy array (bool_area_array) generated from tiles.")

    gdf_numpy_positions = gpd.GeoDataFrame(list_numpy_contour_positions_geoseries, crs=4326).set_geometry('geometry')

    print("GeoDataFrame with tile positions inside numpy contour bool_array created.")

    return np_bool_grid, gdf_numpy_positions


def check_poly_pos(rows_range, columns_range, y, x, tile_height, tile_width):
    poly_i_r = 0  # row
    poly_i_c = 0  # column

    # rows_range -> scan from top to bottom, from max to min
    for i_r, row in enumerate(rows_range):
        if row > y > row - tile_height:
            poly_i_r = i_r
            break

    # columns_range -> scan from left to right, from min to max
    for i_c, column in enumerate(columns_range):
        if column < x < column + tile_width:
            poly_i_c = i_c
            break

    return poly_i_r, poly_i_c


def get_random_start_points_list(number_of_start_points: int, area_bool: np.ndarray):
    start_coordinates = set()  # if a set, then no duplicates, but unordered and unindexed
    rows, cols = area_bool.shape
    available_cells = np.count_nonzero(area_bool)

    while True:
        random_row = np.random.randint(0, rows)
        random_col = np.random.randint(0, cols)
        if area_bool[random_row, random_col]:
            start_coordinates.add((random_row, random_col))
        if len(start_coordinates) >= available_cells or number_of_start_points == len(start_coordinates):
            break

    return list(start_coordinates)  # back to list, because we need an index later


def worker(input_queue, output_queue):
    """
    Necessary worker for python multiprocessing
    Has an Index, if needed...
    """
    for idx, func, args in iter(input_queue.get, 'STOP'):
        result = func(*args)
        output_queue.put([idx, result])


def search_worker(input_queue, output_queue, list_too_search):
    """
    Necessary worker for python multiprocessing
    Has an Index, if needed...
    """
    for idx, func, args in iter(input_queue.get, 'STOP'):
        result = func(list_too_search, *args)
        output_queue.put([idx, result])
