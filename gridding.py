import sys
from pathlib import Path
from tqdm.auto import tqdm
import geopandas as gpd
import numpy as np
import math
from multiprocessing import Process, Queue, cpu_count
import queue
from shapely.geometry import Point, box, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import matplotlib.pyplot as plt


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

    return list(start_coordinates)  # back to list, cause we need a index later


def get_long_lat_diff(grid_size_meter: float, startpoint_latitude: float):
    """

    :param grid_size_meter:
    :param startpoint_latitude:
    :return: width, height difference in longitude, latitude
    """
    earth_radius = 6378137  # earth radius, sphere

    # Coordinate offsets in radians
    new_lat_radians = grid_size_meter / earth_radius
    new_long_radians = grid_size_meter / (earth_radius * math.cos(math.pi * startpoint_latitude / 180))

    # OffsetPosition, decimal degrees
    # new_lat_decimal = startpoint_latitude + new_lat_radians * 180 / math.pi
    # new_long_decimal = startpoint_long + new_long_radians * 180 / math.pi

    # difference only, equals grid_size_meter in lat/long
    lat_difference = new_lat_radians * 180 / math.pi
    long_difference = new_long_radians * 180 / math.pi

    return abs(long_difference), abs(lat_difference)


def which_row_cells_within_area_boundaries(area, r, tile_height, c, tile_width) -> list[Polygon]:

    row_list_of_Polygons = []
    for idx, x0 in enumerate(c):
        x1 = x0 + tile_width
        y1 = r + tile_height
        new_Polygon = box(x0, r, x1, y1)
        if new_Polygon.within(area):
            row_list_of_Polygons.append(new_Polygon)
    return row_list_of_Polygons


def worker(input_queue, output_queue):
    for idx, func, args in iter(input_queue.get, 'STOP'):
        result = func(*args)
        output_queue.put([idx, result])


def processing_geometry_boundary_check(offset: tuple,  # (longitude_offset, latitude_offset)
                                       grid_edge_length_meter: float,
                                       selected_area,
                                       multiprocessing=True):

    # latitude (x diff), longitude (y diff)
    tile_width, tile_height = get_long_lat_diff(grid_edge_length_meter, selected_area.centroid.y)
    xmin, ymin, xmax, ymax = selected_area.bounds

    # offset tuple (long, lat)
    rows = sorted(np.arange(ymin + offset[1], ymax + offset[1] + tile_height, tile_height), reverse=True)  # scan from top to bottom
    columns = np.arange(xmin + offset[0], xmax + offset[0] + tile_width, tile_width)  # scan from left to right

    multipolygon_from_list_Polygons_selected_area = None

    if multiprocessing:
        # Create queues
        task_queue = Queue()
        done_queue = Queue()

        num_of_processes = cpu_count() - 1
        list_Polygons_selected_area = []

        # create tasks and push them into queue
        for idx, row in enumerate(rows):
            one_task = [idx, which_row_cells_within_area_boundaries, (selected_area, row, tile_height, columns, tile_width)]
            task_queue.put(one_task)

        # Start worker processes
        for i in range(num_of_processes):
            Process(target=worker, args=(task_queue, done_queue)).start()

        for _ in tqdm(rows):
            try:
                list_row_Polygons = done_queue.get()
                if len(list_row_Polygons[1]) > 0:
                    list_Polygons_selected_area.extend(list_row_Polygons[1])
            except queue.Empty as e:
                print(e)
            except queue.Full as e:
                print(e)

        # Tell child processes to stop
        for i in range(num_of_processes):
            task_queue.put('STOP')

        multipolygon_from_list_Polygons_selected_area = MultiPolygon(list_Polygons_selected_area)

    #else:
    #    # serial processing, cause time doesn't matter
    #    grid = np.full((len(rows), len(columns)), False)
    #    for xidx, x0 in enumerate(tqdm(columns)):
    #        for yidx, y0 in enumerate(rows):
    #            x1 = x0 - cell_width
    #            y1 = y0 + cell_height
    #            if box(x0, y0, x1, y1).within(selected_area):
    #                grid[yidx][xidx] = True
    #            else:
    #                pass

    return multipolygon_from_list_Polygons_selected_area


def check_real_start_points(geopandas_area_file, start_points):
    biggest_area = get_biggest_area_in_geojson(str(geopandas_area_file))

    start_points_list = []
    for p in start_points:
        start_points_list.append(p)

    for p in start_points_list:
        if not Point(p[0], p[1]).within(biggest_area):
            print("Given real start point", str(p), "is not inside given area", str(geopandas_area_file), "\nTry again")
            return False

    # no disturbances?
    return True


def check_edge_length(grid_edge_length):
    edge_length_array = np.array(grid_edge_length)
    for i in edge_length_array:
        # check if negative values
        if i <= 0:
            print("No negative value or zero possible for grid edge length. Abort!")
            return False
        elif i > 50:
            print("Grid edge length value bigger than 50m. Depending on the area size you won't get results.")

        # are the edge lengths a multiple of 2
        if i > edge_length_array[0] and not i % edge_length_array[0] == 0:
            print("First or second value in edge_length is not a factor of 2 of first value")
            return False

    return True


def get_biggest_area_in_geojson(dam_file_name):
    dam_geojson_filepath = Path("dams_single_geojsons", dam_file_name)
    gdf_dam = gpd.read_file(dam_geojson_filepath)

    gdf_dam_exploded = gdf_dam.geometry.explode().tolist()
    biggest_area = max(gdf_dam_exploded, key=lambda a: a.area)

    return biggest_area


def generate_offset_list(num_offsets: int, area_polygon, grid_edge_length: float):
    """
    Create a list of offsets within the boundaries of chosen grid edge length.
    :param num_offsets: number of offsets to calculate extra to origin
    :param area_polygon: geopandas area polygon
    :param grid_edge_length: edge length of the square to search for offset within
    :return: list of tuples (longitude, latitude) which first entry no offset, rest num_offsets * offset
    """
    cell_width, cell_height = get_long_lat_diff(grid_edge_length, area_polygon.centroid.y)
    offset = []
    rng = np.random.default_rng(seed=42)

    # how many offsets except none
    if num_offsets <= 0:
        num_offsets = 4

    # generate random values between centroid
    random_long_offsets = rng.uniform(low=0,
                                      high=cell_width,
                                      size=num_offsets)
    random_lat_offsets = rng.uniform(low=0,
                                     high=cell_height,
                                     size=num_offsets)
    # fill offset list
    offset.append((0, 0))
    for i in range(num_offsets):
        offset.append((random_long_offsets[i], random_lat_offsets[i]))

    return offset


def find_biggest_grid(selected_area_filename, grid_edge_lengths):

    area_polygon = get_biggest_area_in_geojson(selected_area_filename)

    # search for biggest tiles first
    # find offset for biggest tile size in grid_edge_lengths (should always be its last entry grid_edge_lengths[-1])
    offsets = generate_offset_list(4, area_polygon, grid_edge_lengths[-1])

    list_of_grids = []
    # offset is always in (long, lat)
    for off in offsets:
        grid_multipolygon = processing_geometry_boundary_check(off, grid_edge_lengths[-1], area_polygon)
        if grid_multipolygon is None:
            print("No grid could be found for biggest edge length", grid_edge_lengths[-1], " meter and", str(off), "offset")
        else:
            list_of_grids.append((off, grid_multipolygon))

    # search through list_of_grids for biggest covered area
    biggest_area_multipolygon_idx = 0
    if len(list_of_grids) > 0:
        for idx, (o, multipoly) in enumerate(list_of_grids):
            if unary_union(multipoly).area > unary_union(list_of_grids[biggest_area_multipolygon_idx][1]).area:
                biggest_area_multipolygon_idx = idx
        biggest_area_multipolygon = (list_of_grids[biggest_area_multipolygon_idx][0], make_valid(unary_union(list_of_grids[biggest_area_multipolygon_idx][1])))

        if len(biggest_area_multipolygon[1].geoms) > 1:
            print(len(biggest_area_multipolygon[1].geoms), "members inside biggest_area_multipolygon.multipolygon.geoms")

        # gs = gpd.GeoSeries(, crs="EPSG:4326")
        temp = {'grid_edge_length': [grid_edge_lengths[-1]] * len(biggest_area_multipolygon[1].geoms),
                'offset_longitude': [biggest_area_multipolygon[0][0]] * len(biggest_area_multipolygon[1].geoms),
                'offset_latitude': [biggest_area_multipolygon[0][1]] * len(biggest_area_multipolygon[1].geoms),
                'geometry': biggest_area_multipolygon[1]}
        gdf = gpd.GeoDataFrame(temp, crs="EPSG:4326")

        # just for demo
        fig, ax = plt.subplots(figsize=(20, 20))
        talsperre_geojsonfile = Path("dams_single_geojsons", "Talsperre Malter.geojson")
        gdf_talsperre = gpd.read_file(talsperre_geojsonfile)
        gdf_talsperre.plot(ax=ax, markersize=.05, figsize=(20, 20), color='blue', edgecolor='black')  # cmap='jet'
        gdf.plot(ax=ax, cmap='PuBu', scheme='quantiles')  # edgecolor='white',
        # gdf.explore("geometry", legend=False)
        plt.show()

    else:
        print("Didn't find a grid! No Multipolygon available")

    # transfer to geopandas dataframe

    # search for smaller edge length tiles around the area we just concluded to cover the biggest area

    # save candidate for best result
    # gpd.GeoDataFrame.to_file("my_file.geojson", driver="GeoJSON")

    pass
