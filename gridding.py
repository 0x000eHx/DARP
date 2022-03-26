import webbrowser
from pathlib import Path

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
                if new_Polygon.within(union_geo_coll):
                    am_i_a_good_polygon = False
                    # if new_Polygon.crosses(union_geo_coll) or new_Polygon.touches(union_geo_coll):
                    #     am_i_a_good_polygon = True
            if am_i_a_good_polygon:
                row_list_of_Polygons.append(new_Polygon)
        else:
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
                                       list_known_geo_coll_of_single_polys=None):
    # latitude (x diff), longitude (y diff)
    tile_width, tile_height = get_long_lat_diff(grid_edge_length_meter, selected_area.centroid.y)
    xmin, ymin, xmax, ymax = selected_area.bounds

    # offset tuple (long, lat)
    # scan from top to bottom
    rows = np.arange(ymin + offset[1], ymax + offset[1] + tile_height, tile_height)
    rows = np.flip(rows)
    columns = np.arange(xmin + offset[0], xmax + offset[0] + tile_width, tile_width)  # scan from left to right

    # Create queues for task input and result output
    task_queue = Queue()
    done_queue = Queue()

    num_of_processes = cpu_count() - 1
    list_Polygons_selected_area = []

    # create tasks and push them into queue
    if list_known_geo_coll_of_single_polys is not None:
        list_valid_union_geo_colls = []
        for geo_coll in list_known_geo_coll_of_single_polys:
            list_valid_union_geo_colls.append(make_valid(unary_union(geo_coll)))

        for idx, row in enumerate(rows):
            one_task = [idx, which_row_cells_within_area_boundaries,
                        (selected_area, row, tile_height, columns, tile_width, list_valid_union_geo_colls)]
            task_queue.put(one_task)
    else:
        for idx, row in enumerate(rows):
            one_task = [idx, which_row_cells_within_area_boundaries,
                        (selected_area, row, tile_height, columns, tile_width)]
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
        for one_single_poly in coll_valid_unions.geoms:
            if one_single_poly.area >= (area_one_polygon * polygon_threshold):
                relevant_union_coll.append(one_single_poly)

    # after relevancy check for grouped polygons go for single polygons inside Group of Polys
    list_known_geo_coll_of_single_polys = []
    for one_relevant_coll in relevant_union_coll:
        # check the grouped polygons after size check for  their single Polygons and group them by
        polygon_list = []
        for one_single_poly in tqdm(coll_single_polyons.geoms):
            # which single Polygon is actually inside a relevant Multipolygon
            if one_relevant_coll.covers(one_single_poly):
                polygon_list.append(one_single_poly)
        if len(polygon_list) > 0:
            list_known_geo_coll_of_single_polys.append(MultiPolygon(polygon_list))
    return list_known_geo_coll_of_single_polys


def find_tile_groups_of_given_edge_length(area_polygon,
                                          given_grid_edge_length: float,
                                          polygon_threshold,
                                          known_tiles_gdf: gpd.GeoDataFrame):
    # there is no know geometry data available
    if known_tiles_gdf.empty:
        # find offset for biggest tile size first
        offsets = generate_offset_list(5, area_polygon, given_grid_edge_length)

        list_biggest_grids = []
        # offset is always in (long, lat)
        for off in offsets:
            grid_geo_coll = processing_geometry_boundary_check(off, given_grid_edge_length, area_polygon)
            if grid_geo_coll is None:
                print("No grid could be found for biggest square edge length", given_grid_edge_length, "meter and",
                      str(off), "offset")
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
            list_known_geo_coll_of_single_polys = keep_only_relevant_geo_coll_of_single_polygon_geo_coll(
                geo_coll_single_polyons, polygon_threshold)

            gdf_dict = {'offset_longitude': best_offset[0] * len(list_known_geo_coll_of_single_polys),
                        'offset_latitude': best_offset[1] * len(list_known_geo_coll_of_single_polys),
                        'grid_edge_length': [given_grid_edge_length] * len(list_known_geo_coll_of_single_polys),
                        'covered_area': [unary_union(x).area for x in list_known_geo_coll_of_single_polys],
                        'geometry': list_known_geo_coll_of_single_polys}
            gdf_biggest = gpd.GeoDataFrame(gdf_dict, crs=4326).set_geometry('geometry')
            return gdf_biggest
        else:
            return gpd.GeoDataFrame()

    # there is some known geometry data available
    if not known_tiles_gdf.empty:
        # determine best_offset for aligned tiles, is one hashable column inside known_tiles_gdf
        best_offset = (known_tiles_gdf.head(1).offset_longitude[0], known_tiles_gdf.head(1).offset_latitude[0])
        list_known_geo_coll_of_single_polys = []  # will be list of series (of Polygons)
        # extract all GeoSeries from known_tiles_gdf
        for row in known_tiles_gdf.itertuples(index=False):
            list_known_geo_coll_of_single_polys.append(row.geometry)  # geometry is a (hashable) column in row

        # get all Polygon of given_grid_edge_length inside area_polygon
        grid_geo_coll = processing_geometry_boundary_check(best_offset,
                                                           given_grid_edge_length,
                                                           area_polygon,
                                                           list_known_geo_coll_of_single_polys)

        if grid_geo_coll.is_empty:
            print("No grid could be found for biggest square edge length", given_grid_edge_length, "meter and",
                  str(best_offset), "offset")
            return gpd.GeoDataFrame()
        else:
            # group the polygons and reject groups of polygons (by area) which are too small
            list_relevant_geo_colls = keep_only_relevant_geo_coll_of_single_polygon_geo_coll(grid_geo_coll,
                                                                                             polygon_threshold)

            # generate GeoDataframe
            gdf_dict = {'offset_longitude': best_offset[0] * len(list_relevant_geo_colls),
                        'offset_latitude': best_offset[1] * len(list_relevant_geo_colls),
                        'grid_edge_length': [given_grid_edge_length] * len(list_relevant_geo_colls),
                        'covered_area': [unary_union(x).area for x in list_relevant_geo_colls],
                        'geometry': list_relevant_geo_colls}
            gdf = gpd.GeoDataFrame(gdf_dict, crs=4326).set_geometry('geometry')  # , crs=4326
            return gdf


def find_grid(selected_area_filename, grid_edge_lengths: list, polygon_threshold: list):
    area_polygon = get_biggest_area_polygon(selected_area_filename)

    # search for biggest tiles first
    gdf_collection = gpd.GeoDataFrame()
    grid_edge_lengths = sorted(grid_edge_lengths, reverse=True)  # from greatest to smallest value
    polygon_threshold = sorted(polygon_threshold, reverse=True)  # from greatest to smallest value

    for idx, grid_length in enumerate(grid_edge_lengths):
        gdf_one_tile_size = find_tile_groups_of_given_edge_length(area_polygon,
                                                                  grid_length,
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
            gdf_collection.to_file('./geodataframes/tiles.geojson', driver='GeoJSON')

        # print(gdf_biggest_tiles.geom_equals(biggest_gdf, align=True))  # seems fine

    fol_map = gdf_collection.explore('covered_area', cmap='jet', scheme='quantiles', legend=True)  # PuBu
    fol_map.save("talsperre_combined.html")
    webbrowser.open("talsperre_combined.html")

    # big_gdf = gpd.read_file('./geodataframes/tiles_new.geojson')
    # big_gdf.set_geometry('geometry')
    # fol_map = big_gdf.explore('covered_area', cmap='PuBu', scheme='quantiles', legend=True)

    return gdf_collection
