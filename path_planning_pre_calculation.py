import time

import geopandas as gpd
import numpy as np
import pandas
from tqdm.auto import tqdm
from multiprocessing import Process, Queue, cpu_count
import queue
from shapely.geometry import Polygon, MultiPolygon, LineString, box, MultiLineString
from shapely.ops import unary_union
from shapely.validation import make_valid


def calc_path_A_to_B():
    path_length_meter = 0
    line = LineString()

    return line, path_length_meter


def get_start_points_from_coords(list_start_point_coords: list, numpy_bool_array: np.ndarray):
    list_start_points = []


def generate_stc_geodataframe(input_gdf: gpd.GeoDataFrame, assignment_matrix: np.ndarray, paths):
    new_geoseries_list = []

    for serie in input_gdf.itertuples():
        x = serie.np_x
        y = serie.np_y
        drone = assignment_matrix[x, y]
        new_polys_geoseries = divide_polygon(x, y, serie.geometry, drone)
        new_geoseries_list.extend(new_polys_geoseries)

    gdf_trajectory = gpd.GeoDataFrame(new_geoseries_list, crs=4326).set_geometry('geometry')

    list_MultiLineStrings = []
    for ix, drone in enumerate(paths):
        lines_list = []
        for line in drone:
            np1x, np1y, np2x, np2y = line

            # query as tiny bit (1 or 2 ms) faster than
            # gdf_trajectory[gdf_trajectory['np_x'] == np1x][gdf_trajectory['np_y'] == np1y].reset_index()
            point1 = gdf_trajectory.query(f'np_x == {np1x} and np_y == {np1y}').reset_index().geometry.centroid[0]
            point2 = gdf_trajectory.query(f'np_x == {np2x} and np_y == {np2y}').reset_index().geometry.centroid[0]

            lines_list.append(LineString([point1, point2]))

        data = {'np_x': np.nan,
                'np_y': np.nan,
                'assigned_drone': ix,
                'poly': False,
                'line': True,
                'geometry': MultiLineString(lines_list)}
        list_MultiLineStrings.append(gpd.GeoSeries(data, crs=4326))

    gdf_lines = gpd.GeoDataFrame(list_MultiLineStrings, crs=4326).set_geometry('geometry')

    gdf = gpd.GeoDataFrame(pandas.concat([gdf_trajectory, gdf_lines], axis=0, ignore_index=True),
                           crs=gdf_lines.crs)

    return gdf


def divide_polygon(pos_x, pos_y, poly: Polygon, assigned_drone):
    minx, miny, maxx, maxy = poly.bounds

    l_x = (maxx - minx) / 2
    l_y = (maxy - miny) / 2

    data = [{'np_x': pos_x * 2,
             'np_y': pos_y * 2,
             'assigned_drone': assigned_drone,
             'poly': True,
             'line': False,
             'geometry': box(minx, miny + l_y, maxx - l_x, maxy)},
            {'np_x': pos_x * 2,
             'np_y': pos_y * 2 + 1,
             'assigned_drone': assigned_drone,
             'poly': True,
             'line': False,
             'geometry': box(minx + l_x, miny + l_y, maxx, maxy)},
            {'np_x': pos_x * 2 + 1,
             'np_y': pos_y * 2,
             'assigned_drone': assigned_drone,
             'poly': True,
             'line': False,
             'geometry': box(minx, miny, maxx - l_x, maxy - l_y)},
            {'np_x': pos_x * 2 + 1,
             'np_y': pos_y * 2 + 1,
             'assigned_drone': assigned_drone,
             'poly': True,
             'line': False,
             'geometry': box(minx + l_x, miny, maxx, maxy - l_y)}]

    return gpd.GeoSeries(data, crs=4326)


def generate_numpy_contour_array(multipoly: MultiPolygon, dict_tile_width_height, dict_multipoly_offset):
    print("Generate numpy contour bool_area_array from given number of multipolygons")
    union_area = make_valid(unary_union(multipoly))
    minx, miny, maxx, maxy = union_area.bounds

    columns_range = np.arange(minx + dict_multipoly_offset['offset_longitude'],
                              maxx + dict_multipoly_offset['offset_longitude'] + dict_tile_width_height['tile_width'],
                              dict_tile_width_height['tile_width'])  # scan from left to right
    rows_range = np.arange(miny + dict_multipoly_offset['offset_latitude'],
                           maxy + dict_multipoly_offset['offset_latitude'] + dict_tile_width_height['tile_height'],
                           dict_tile_width_height['tile_height'])
    rows_range = np.flip(rows_range)  # scan from top to bottom
    np_bool_grid = np.full(shape=(rows_range.shape[0], columns_range.shape[0]), fill_value=False, dtype=bool)

    list_numpy_contour_positions_geoseries = []

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    num_of_processes = cpu_count() - 1

    # create tasks and push them into queue
    for idx, poly in enumerate(multipoly.geoms):
        one_task = [idx, check_poly_pos, (rows_range, columns_range, poly.centroid.y, poly.centroid.x,
                                          dict_tile_width_height['tile_height'],
                                          dict_tile_width_height['tile_width'])]
        task_queue.put(one_task)

    # Start worker processes
    for i in range(num_of_processes):
        Process(target=worker, args=(task_queue, done_queue)).start()

    for _ in tqdm(multipoly.geoms):
        try:
            ix, (row_idx, col_idx) = done_queue.get()
            np_bool_grid[row_idx, col_idx] = True

            data = {'np_x': row_idx,
                    'np_y': col_idx,
                    'geometry': multipoly.geoms[ix]}
            list_numpy_contour_positions_geoseries.append(gpd.GeoSeries(data, crs=4326))

        except queue.Empty as e:
            print(e)
        except queue.Full as e:
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

    for i_r, row in enumerate(rows_range):
        if row > y > row - tile_height:
            poly_i_r = i_r
            break
    for i_c, column in enumerate(columns_range):
        if column < x < column + tile_width:
            poly_i_c = i_c
            break

    return poly_i_r, poly_i_c


def get_random_start_points_list(number_of_start_points: int, area_bool: np.ndarray):
    start_coordinates = set()  # if a set, then no duplicates, but unordered and unindexed
    rows, cols = area_bool.shape

    while True:
        random_row = np.random.randint(0, rows)
        random_col = np.random.randint(0, cols)

        if area_bool[random_row, random_col]:
            start_coordinates.add((random_row, random_col))

        if len(start_coordinates) == number_of_start_points:
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
