from pathlib import Path
import uuid
import pandas
from tqdm.auto import tqdm
import geopandas as gpd
import numpy as np
import math
from multiprocessing import Process, Queue, cpu_count
import queue
from shapely.geometry import Point, box, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid


def get_start_points_from_coords(list_start_point_coords: list, numpy_bool_array: np.ndarray):
    list_start_points = []


def generate_numpy_contour_array(multipoly: MultiPolygon, dict_tile_width_height, dict_multipoly_offset):

    union_area = make_valid(unary_union(multipoly))
    minx, miny, maxx, maxy = union_area.bounds

    tiles_longitude = round((maxx - minx) / dict_tile_width_height['tile_width'])  # colums
    tiles_latitude = round((maxy - miny) / dict_tile_width_height['tile_height'])  # rows

    np_bool_grid = np.full(shape=(int(tiles_latitude), int(tiles_longitude)), fill_value=False, dtype=bool)

    columns_range = np.arange(minx + dict_multipoly_offset['offset_latitude'],
                              maxx + dict_multipoly_offset['offset_latitude'] + dict_tile_width_height['tile_width'],
                              dict_tile_width_height['tile_width'])  # scan from left to right
    rows_range = np.arange(miny + dict_multipoly_offset['offset_longitude'],
                           maxy + dict_multipoly_offset['offset_longitude'] + dict_tile_width_height['tile_height'],
                           dict_tile_width_height['tile_height'])
    rows_range = np.flip(rows_range)  # scan from top to bottom

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
        except queue.Empty as e:
            print(e)
        except queue.Full as e:
            print(e)

    # Tell child processes to stop
    for i in range(num_of_processes):
        task_queue.put('STOP')

    return np_bool_grid


def check_poly_pos(rows_range, columns_range, y, x, tile_height, tile_width):
    poly_i_r = 0
    poly_i_c = 0

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

    return list(start_coordinates)  # back to list, cause we need a index later


def get_long_lat_diff(square_edge_length_meter: float, startpoint_latitude: float):
    """

    :param square_edge_length_meter:
    :param startpoint_latitude:
    :return: width, height difference in longitude, latitude
    """
    earth_radius = 6378137  # earth radius, sphere

    # Coordinate offsets in radians
    new_lat_radians = square_edge_length_meter / earth_radius
    new_long_radians = square_edge_length_meter / (earth_radius * math.cos(math.pi * startpoint_latitude / 180))

    # OffsetPosition, decimal degrees
    # new_lat_decimal = startpoint_latitude + new_lat_radians * 180 / math.pi
    # new_long_decimal = startpoint_long + new_long_radians * 180 / math.pi

    # difference only, equals square_edge_length_meter in lat/long
    lat_difference = new_lat_radians * 180 / math.pi
    long_difference = new_long_radians * 180 / math.pi

    return abs(long_difference), abs(lat_difference)


def which_row_cells_within_area_boundaries(area, r, tile_height, c, tile_width, list_union_geo_coll=None) -> list:
    row_list_of_Polygons = []
    for idx, x0 in enumerate(c):
        x1 = x0 + tile_width
        y1 = r + tile_height
        new_Polygon = box(x0, r, x1, y1)
        if list_union_geo_coll is not None:
            am_i_a_good_polygon = True
            for union_geo_coll in list_union_geo_coll:
                if not new_Polygon.within(area):
                    am_i_a_good_polygon = False
                    break
                if union_geo_coll.covers(new_Polygon) and not new_Polygon.touches(union_geo_coll):
                    am_i_a_good_polygon = False
                    # if new_Polygon.overlaps(union_geo_coll):
                    #     am_i_a_good_polygon = True
            if am_i_a_good_polygon:
                row_list_of_Polygons.append(new_Polygon)
        else:
            if new_Polygon.within(area):
                row_list_of_Polygons.append(new_Polygon)
    return row_list_of_Polygons


def worker(input_queue, output_queue):
    """
    Necessary worker for python multiprocessing
    Has an Index, if needed...
    """
    for idx, func, args in iter(input_queue.get, 'STOP'):
        result = func(*args)
        output_queue.put([idx, result])


def processing_geometry_boundary_check(offset: tuple,  # (longitude_offset, latitude_offset)
                                       square_size_long_lat: dict,
                                       selected_area,
                                       list_known_geo_coll_of_single_polys: list):
    xmin, ymin, xmax, ymax = selected_area.bounds

    # offset tuple (long, lat)
    rows = np.arange(ymin + offset[1], ymax + offset[1] + square_size_long_lat['tile_height'],
                     square_size_long_lat['tile_height'])
    rows = np.flip(rows)  # scan from top to bottom
    columns = np.arange(xmin + offset[0], xmax + offset[0] + square_size_long_lat['tile_width'],
                        square_size_long_lat['tile_width'])  # scan from left to right

    # Create queues for task input and result output
    task_queue = Queue()
    done_queue = Queue()

    num_of_processes = cpu_count() - 1
    list_Polygons_selected_area = []

    # create tasks and push them into queue
    if len(list_known_geo_coll_of_single_polys) > 0:
        list_valid_union_geo_colls = []
        for geo_coll in list_known_geo_coll_of_single_polys:
            list_valid_union_geo_colls.append(make_valid(unary_union(geo_coll)))

        for idx, row in enumerate(rows):
            one_task = [idx, which_row_cells_within_area_boundaries,
                        (selected_area, row, square_size_long_lat['tile_height'], columns,
                         square_size_long_lat['tile_width'], list_valid_union_geo_colls)]
            task_queue.put(one_task)
    else:
        for idx, row in enumerate(rows):
            one_task = [idx, which_row_cells_within_area_boundaries,
                        (selected_area, row, square_size_long_lat['tile_height'], columns,
                         square_size_long_lat['tile_width'])]
            task_queue.put(one_task)

    # Start worker processes
    for i in range(num_of_processes):
        Process(target=worker, args=(task_queue, done_queue)).start()

    for _ in tqdm(rows):
        try:
            list_row_Polygons = done_queue.get()
            if len(list_row_Polygons[1]) > 0:
                list_Polygons_selected_area.extend(list_row_Polygons[1])  # extend and not append to unbox received list
        except queue.Empty as e:
            print(e)
        except queue.Full as e:
            print(e)

    # Tell child processes to stop
    for i in range(num_of_processes):
        task_queue.put('STOP')

    return MultiPolygon(list_Polygons_selected_area)


def check_real_start_points(geopandas_area_file, start_points):
    biggest_area = get_biggest_area_polygon(str(geopandas_area_file))

    start_points_list = []
    for p in start_points:
        start_points_list.append(p)

    for p in start_points_list:
        if not Point(p[0], p[1]).within(biggest_area):
            print("Given real start point", str(p), "is not inside given area", str(geopandas_area_file), "\nTry again")
            return False

    # no disturbances?
    return True


def check_edge_length_polygon_threshold(grid_edge_lengths, polys_threshold):
    """

    :param grid_edge_lengths:
    :param polys_threshold:
    :return:
    """
    if isinstance(grid_edge_lengths, list):
        grid_edge_lengths = sorted(grid_edge_lengths, reverse=True)  # from greatest to smallest value
    if isinstance(polys_threshold, list):
        polys_threshold = sorted(polys_threshold, reverse=True)  # from greatest to smallest value

    edge_length_array = np.array(grid_edge_lengths)
    poly_threshold_array = np.array(polys_threshold)

    if edge_length_array.shape != poly_threshold_array.shape:
        print("Number defined edges don't match number polygon_threshold. Abort!")
        return False

    for i in edge_length_array:
        # check if negative values
        if i <= 0:
            print("No negative edge length value", str(i), "allowed. Abort!")
            return False

        elif i > 50:
            print("Grid edge length value bigger than 50m. Depending on the area size you might not get results.")

        # are the edge lengths divider from another?
        if i > np.amin(edge_length_array) and not i % np.amin(edge_length_array) == 0:
            print("Edge_length value", i, "doesn't match,"
                                          "cause it is not the exponentiation of a half of the greatest value",
                  str(np.amin(edge_length_array)), "\nThe tiles won't align!")
            return False

    # check if polygon_threshold list contains a negative value and replace it with a reasonable entry
    for poly_thresh in poly_threshold_array:
        if poly_thresh <= 0:
            print("polygon_threshold", poly_thresh, "<= 0: invalid entry! Abort!")
            return False

    return True


def get_biggest_area_polygon(dam_file_name):
    dam_geojson_filepath = Path("dams_single_geojsons", dam_file_name)
    gdf_dam = gpd.read_file(dam_geojson_filepath)

    gdf_dam_exploded = gdf_dam.geometry.explode(index_parts=True)  # no index_parts / .tolist()
    biggest_area = max(gdf_dam_exploded, key=lambda a: a.area)

    return biggest_area


def generate_square_edges_long_lat(grid_edge_lengths, selected_area) -> dict:
    list_long_lat_tuples = {}
    edge_length_max = max(grid_edge_lengths)

    # latitude == width (y diff), longitude == height (x diff)
    tile_width_max, tile_height_max = get_long_lat_diff(edge_length_max, selected_area.centroid.y)

    for edge_length_meter in grid_edge_lengths:
        if edge_length_max % edge_length_meter == 0:
            divider = int(edge_length_max / edge_length_meter)
            tile_width = tile_width_max / divider
            tile_height = tile_height_max / divider
            list_long_lat_tuples[f'{edge_length_meter}'] = {"tile_width": tile_width,
                                                            "tile_height": tile_height}
    return list_long_lat_tuples


def generate_offset_list(num_offsets: int, grid_edge_length: dict):
    """
    Create a list of offsets within the boundaries of chosen grid edge length.
    :param num_offsets: number of offsets to calculate extra to origin
    :param grid_edge_length: edge length of the square to search for offset within
    :return: list of tuples (longitude, latitude) which first entry no offset, rest num_offsets * offset
    """

    cell_width, cell_height = grid_edge_length["tile_width"], grid_edge_length["tile_height"]
    offset = []
    rng = np.random.default_rng()

    # how many offsets except none
    if num_offsets <= 0:
        num_offsets = 5

    # generate random values between centroid
    random_long_offsets = rng.uniform(low=0,
                                      high=cell_width,
                                      size=num_offsets)
    random_lat_offsets = rng.uniform(low=0,
                                     high=cell_height,
                                     size=num_offsets)
    # fill offset list
    for i in range(num_offsets):
        offset.append((random_long_offsets[i], random_lat_offsets[i]))

    return offset


def keep_only_relevant_geo_coll_of_single_polygon_geo_coll(coll_single_polyons, polygon_threshold):
    """
    Check if single polygons inside Collection form a group and the groups joined area is below
     polygon_threshold * area of one polygon

    :param coll_single_polyons: Must be Multipolygon of many single Polygon

    :param polygon_threshold: Number of Polygons forming a group which area minimum will get considered relevant

    :return: List of geometry collection polygon groups which don't align and were considered relevant
    """
    # remove Collection regions which are considered too small by polygon_threshold
    # need to unify at intersecting/touching edges and divide polygons with make_valid(unary_union()) -> see shapely doc
    coll_valid_unions = make_valid(unary_union(coll_single_polyons))
    area_one_polygon = coll_single_polyons.geoms[0].area
    relevant_union_coll = []
    if isinstance(coll_valid_unions, Polygon):
        if coll_valid_unions.area >= (area_one_polygon * polygon_threshold):
            relevant_union_coll.append(coll_valid_unions)
    else:
        for one_single_poly in tqdm(coll_valid_unions.geoms):
            if one_single_poly.area >= (area_one_polygon * polygon_threshold):
                relevant_union_coll.append(one_single_poly)

    # Create queues for task input and result output
    task_queue = Queue()
    done_queue = Queue()
    num_of_processes = cpu_count() - 1

    # after relevancy check for grouped polygons go for single polygons inside Group of Polys
    list_known_geo_coll_of_single_polys = []

    for idx, one_relevant_coll in enumerate(relevant_union_coll):
        one_task = [idx, keep_relevent_coll_helper, (coll_single_polyons, one_relevant_coll)]
        task_queue.put(one_task)

    # Start worker processes
    for _ in range(num_of_processes):
        Process(target=worker, args=(task_queue, done_queue)).start()

    for _ in tqdm(relevant_union_coll):
        try:
            idx, list_polys = done_queue.get()
            if len(list_polys) > 0:
                list_known_geo_coll_of_single_polys.append(MultiPolygon(list_polys))

        except queue.Empty as e:
            print(e)
        except queue.Full as e:
            print(e)

    # Tell child processes to stop
    for _ in range(num_of_processes):
        task_queue.put('STOP')

    return list_known_geo_coll_of_single_polys


def keep_relevent_coll_helper(coll_single_polyons, one_relevant_coll):
    polygon_list = []
    for one_single_poly in coll_single_polyons.geoms:
        # which single Polygon is actually inside a relevant Multipolygon
        if one_relevant_coll.covers(one_single_poly):
            polygon_list.append(one_single_poly)
    return polygon_list


def create_geodataframe_dict(best_offset, dict_square_edge_length_long_lat, list_known_geo_coll_of_single_polys):
    gdf_dict = {'hash': [str(uuid.uuid4()) for _ in list_known_geo_coll_of_single_polys],
                'offset_longitude': [best_offset[0]] * len(list_known_geo_coll_of_single_polys),
                'offset_latitude': [best_offset[1]] * len(list_known_geo_coll_of_single_polys),
                'tile_width': [dict_square_edge_length_long_lat['tile_width']] * len(
                    list_known_geo_coll_of_single_polys),
                'tile_height': [dict_square_edge_length_long_lat['tile_height']] * len(
                    list_known_geo_coll_of_single_polys),
                'covered_area': [unary_union(x).area for x in list_known_geo_coll_of_single_polys],
                'geometry': list_known_geo_coll_of_single_polys}
    return gdf_dict


def find_tile_groups_of_given_edge_length(area_polygon,
                                          dict_square_edge_length_long_lat: dict,
                                          polygon_threshold,
                                          known_tiles_gdf: gpd.GeoDataFrame):
    # there is no know geometry data available
    if known_tiles_gdf.empty:
        # find offset for biggest tile size first
        offsets = generate_offset_list(5, dict_square_edge_length_long_lat)

        list_biggest_grids = []
        # offset is always in (long, lat)
        for off in offsets:
            grid_geo_coll = processing_geometry_boundary_check(off, dict_square_edge_length_long_lat, area_polygon, [])
            if grid_geo_coll is None:
                print("No grid could be found for biggest square edge length", str(dict_square_edge_length_long_lat),
                      "meter and", str(off), "offset")
            else:
                list_biggest_grids.append((off, grid_geo_coll))

        # search through list_biggest_grids for biggest covered area
        if len(list_biggest_grids) > 0:
            # make unary_union of all determined Polygon and compare the areas, the biggest area wins
            best_offset, geo_coll_single_polyons = max(list_biggest_grids,
                                                       key=lambda a: make_valid(unary_union(a[1])).area)
            if len(geo_coll_single_polyons.geoms) > 1:
                print(len(geo_coll_single_polyons.geoms), "Polygons found in given area!")

            # relevancy check of joined polygons group
            list_relevant_geo_colls = keep_only_relevant_geo_coll_of_single_polygon_geo_coll(
                geo_coll_single_polyons, polygon_threshold)

            gdf_dict = create_geodataframe_dict(best_offset,
                                                dict_square_edge_length_long_lat,
                                                list_relevant_geo_colls)

            gdf_biggest_tile_size = gpd.GeoDataFrame(gdf_dict, crs=4326).set_geometry('geometry')
            return gdf_biggest_tile_size
        else:
            return gpd.GeoDataFrame()

    # there is some known geometry data available
    if not known_tiles_gdf.empty:
        # determine best_offset for aligned tiles, is one hashable column inside known_tiles_gdf
        best_offset = (known_tiles_gdf.head(1).offset_longitude[0], known_tiles_gdf.head(1).offset_latitude[0])
        # TODO offset of smallest tiles doesn't match biggest tiles in final geodataframe, I smell a bug somewhere

        list_known_geo_coll_of_single_polys = []  # will be list of series (of Polygons)
        # extract all GeoSeries from known_tiles_gdf
        for row in known_tiles_gdf.itertuples(index=False):
            list_known_geo_coll_of_single_polys.append(row.geometry)  # geometry is a (hashable) column in row

        # get all Polygon of dict_square_edge_length_long_lat inside area_polygon
        grid_geo_coll = processing_geometry_boundary_check(best_offset,
                                                           dict_square_edge_length_long_lat,
                                                           area_polygon,
                                                           list_known_geo_coll_of_single_polys)

        if grid_geo_coll.is_empty:
            print("No grid could be found for biggest square edge length", dict_square_edge_length_long_lat,
                  "meter and", str(best_offset), "offset")
            return gpd.GeoDataFrame()
        else:
            # group the polygons and reject groups of polygons (by area) which are too small
            list_relevant_geo_colls = keep_only_relevant_geo_coll_of_single_polygon_geo_coll(grid_geo_coll,
                                                                                             polygon_threshold)

            # generate GeoDataframe
            gdf_dict = create_geodataframe_dict(best_offset,
                                                dict_square_edge_length_long_lat,
                                                list_relevant_geo_colls)
            gdf = gpd.GeoDataFrame(gdf_dict, crs=4326).set_geometry('geometry')  # , crs=4326
            return gdf


def find_grid(area_polygon, grid_edge_lengths: list, polygon_threshold: list):

    # generate tile width and height by calculating the biggest tile size to go for and divide it into the other tile sizes
    # hopefully clears out a mismatch in long/lat max and min values between biggest tile size and smallest
    square_edges_long_lat = generate_square_edges_long_lat(grid_edge_lengths, area_polygon)

    # search for biggest tiles first
    gdf_collection = gpd.GeoDataFrame()

    grid_edge_lengths = sorted(grid_edge_lengths, reverse=True)  # from greatest to smallest value
    polygon_threshold = sorted(polygon_threshold, reverse=True)  # from greatest to smallest value

    for idx, edge_length in enumerate(grid_edge_lengths):
        gdf_one_tile_size = find_tile_groups_of_given_edge_length(area_polygon,
                                                                  square_edges_long_lat[f'{edge_length}'],
                                                                  polygon_threshold[idx],
                                                                  gdf_collection)
        if gdf_one_tile_size.empty:
            print("Didn't find a grid with", grid_edge_lengths[idx],
                  "square edge length!\nContinuing with smaller one...")
        else:
            gdf_collection = gpd.GeoDataFrame(pandas.concat([gdf_collection,
                                                             gdf_one_tile_size],
                                                            axis=0,
                                                            ignore_index=True),
                                              crs=gdf_one_tile_size.crs)  # , verify_integrity=True

    return gdf_collection
