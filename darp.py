import numpy as np
import sys
import cv2
from Visualization import darp_area_visualization
import time
from tqdm.auto import tqdm
import imageio
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os
from pyinstrument import Profiler
from numba import njit

np.set_printoptions(threshold=sys.maxsize)
float_overflow = np.finfo(np.float64).max / 10


def check_start_parameter(dict_start_parameter: dict, bool_area: np.ndarray):
    print("Checking DARP start parameters now...")
    start_positions_set = set()  # check for duplicate, even if dict_start_parameter creation could have avoided it
    # sum up tiles_counts per robot / startpoint
    sum_tiles_covered_area = 0
    # check max start points
    for p_id, p_info in dict_start_parameter.items():
        start_positions_set.add((p_info['row'], p_info['col']))
        if not (0 <= p_info['row'] < bool_area.shape[0]) or not (0 <= p_info['col'] < bool_area.shape[1]):
            print(f"Start position [{p_info['row']},{p_info['col']}] is not inside the given area!")
            return False
        elif not bool_area[p_info['row'], p_info['col']]:
            print(f"Start position [{p_info['row']},{p_info['col']}] is on a obstacle tile inside the given area!")
            return False

        if p_info['tiles_count'] < 0:
            print("Can't assign negative value of tiles_count per robot.")
            return False
        elif p_info['tiles_count'] == 0:
            print("Can't assign a zero value to tiles_count per robot.")
            return False

        sum_tiles_covered_area += p_info['tiles_count']

    if len(start_positions_set) != len(dict_start_parameter.items()):
        print("Found a duplicate start point. Can't start DARP under these premise.")
        return False

    # check max tiles_count
    non_obstacle_positions = np.argwhere(bool_area)
    effective_tile_number = non_obstacle_positions.shape[0] - len(dict_start_parameter.items())
    print("Effective number of tiles: ", effective_tile_number)

    diff_tiles = effective_tile_number - sum_tiles_covered_area
    if diff_tiles < 0:
        print("Amount of area tiles to cover (" + str(effective_tile_number) +
              ") is smaller than sum of tiles covered by all robots (" + str(sum_tiles_covered_area) + ").")
    elif diff_tiles > 0:
        print("A number of", str(diff_tiles), "tiles hasn't been assigned to any robot.\n",
              "Aborting Calculation! Please start DARP with at least one more start point")
        return False
    else:
        print("The number of area tiles to cover match the sum of all covered tiles by robots. Perfect!")

    # if everything checks out
    return True


def check_array_continuity(area: np.ndarray):
    """
    Check at the beginning if given area array has no seperated areas / tiles.

    :param area: Must be bool array. True entries are considered assignable tiles, the rest are obstacles.

    :return: True if everything works out, False if seperated areas detected
    """
    connectivity_img = np.zeros(area.shape, dtype=np.uint8)

    mask = np.where(area)
    connectivity_img[mask[0], mask[1]] = 255
    num_labels, labels_im = cv2.connectedComponents(image=connectivity_img, connectivity=4)
    if num_labels > 2:
        print("The given area MUST not have unreachable and/or closed shape regions!")
        return False
    else:
        return True


@njit
def seed(a):
    np.random.seed(a)


@njit(fastmath=True)
def assign(non_obs_pos: np.ndarray,
           Assignment_Matrix: np.ndarray,
           Metric_Matrix: np.ndarray,
           ArrayOfElements: np.ndarray):
    """
    (Re)Assign every tile to a robot.

    :param non_obs_pos: manipulate only non obstacle positions and leave the rest of the MetricMatrix untouched

    :param Assignment_Matrix: The real self.A Assignment Matrix

    :param Metric_Matrix:
    :param ArrayOfElements:
    :return:
    """
    for cell in non_obs_pos:
        # argmin index is same as index of robot in initial_positions array
        if not Assignment_Matrix[cell[0], cell[1]] == np.argmin(Metric_Matrix[:, cell[0], cell[1]]):
            Assignment_Matrix[cell[0], cell[1]] = np.argmin(Metric_Matrix[:, cell[0], cell[1]])

    for i in range(Metric_Matrix.shape[0]):
        ArrayOfElements[i] = np.count_nonzero(Assignment_Matrix == i) - 1  # -1 for the start position of robot i


@njit(fastmath=True)
def FinalUpdateOnMetricMatrix(non_obs_pos: np.ndarray,
                              criterionMatrix: np.ndarray,
                              MetricMatrix: np.ndarray,
                              ConnectedMultiplierMatrix: np.ndarray,
                              random_level: float):
    """
    Calculates the Final MetricMatrix with given criterionMatrix, Random input, MetricMatrix, ConnectedMultiplier
    """
    # manipulate only non obstacle positions and leave the rest of the MetricMatrix untouched
    for cell in non_obs_pos:
        MetricMatrix[cell[0], cell[1]] *= criterionMatrix[cell[0], cell[1]]
        MetricMatrix[cell[0], cell[1]] *= 2 * random_level * np.random.uniform(0, 1) + (1 - random_level)
        MetricMatrix[cell[0], cell[1]] *= ConnectedMultiplierMatrix[cell[0], cell[1]]


@njit(fastmath=True)
def calc_connected_multiplier(non_obs_pos: np.ndarray, cc_variation: float, dist1: np.ndarray, dist2: np.ndarray):
    """
    Calculates the connected multiplier between the binary robot tiles (connected area) and the binary non-robot tiles

    :param non_obs_pos: manipulate only non obstacle positions and leave the rest of the MetricMatrix untouched

    :param cc_variation:

    :param dist1: Must contain the euclidean distances of all tiles around the binary robot tiles

    :param dist2: Must contain the euclidean distances of all tiles around the binary non-robot tiles

    :return: connected multiplier array in the shape of the whole area (dist1.shape & dist2.shape)
    """
    returnM = np.subtract(dist1, dist2)
    MaxV = np.max(returnM)
    MinV = np.min(returnM)

    for cell in non_obs_pos:
        returnM[cell[0], cell[1]] -= MinV
        returnM[cell[0], cell[1]] *= ((2 * cc_variation) / (MaxV - MinV))
        returnM[cell[0], cell[1]] += (1 - cc_variation)

    return returnM


@njit(fastmath=True)
def calculateCriterionMatrix(importance_trigger,
                             TilesImportanceMatrix,
                             MinimumImportance,
                             MaximumImportance,
                             correctionMult,
                             below_zero):
    """
    Generates a new correction multiplier matrix.
    If importance_trigger is True: ImportanceMatrix influence is considered.
    """
    returnCrit = np.zeros(TilesImportanceMatrix.shape)
    if importance_trigger:
        if below_zero:
            returnCrit = (TilesImportanceMatrix - MinimumImportance) * (
                    (correctionMult - 1) / (MaximumImportance - MinimumImportance)) + 1
        else:
            returnCrit = (TilesImportanceMatrix - MinimumImportance) * (
                    (1 - correctionMult) / (MaximumImportance - MinimumImportance)) + correctionMult
    else:
        returnCrit[:, :] = correctionMult
    return returnCrit


@njit(fastmath=True)  # parallel=True, fastmath=True
def construct_binary_images(non_obs_pos: np.ndarray, area_tiles: np.ndarray, robot_start_point):
    """
    Returns 2 maps in the given area_tiles.shape

    - robot_tiles_binary: where all tiles around + robot_start_point are ones, the rest is zero

    - nonrobot_tiles_binary: where tiles which aren't background and not around the robot_start_point are ones, rest is zero

    :param non_obs_pos: cell coordinates inside the lake area as numpy.array

    :param area_tiles: map of tiles with minimum 3 different labels, 0 should always be the background value

    :param robot_start_point: is needed to determine which area of connected tiles should be BinaryRobot area

    :return: robot_tiles_binary, nonrobot_tiles_binary
    """
    robot_tiles_binary = np.zeros(area_tiles.shape, dtype=np.uint8)
    nonrobot_tiles_binary = np.zeros(area_tiles.shape, dtype=np.uint8)

    for cell in non_obs_pos:
        if area_tiles[cell[0], cell[1]] == area_tiles[robot_start_point]:
            robot_tiles_binary[cell[0], cell[1]] = 1
        elif area_tiles[cell[0], cell[1]] > 0 and (area_tiles[cell[0], cell[1]] != area_tiles[robot_start_point]):
            nonrobot_tiles_binary[cell[0], cell[1]] = 1
    return robot_tiles_binary, nonrobot_tiles_binary


@njit(fastmath=True)
def update_connectivity(connectivity_matrix: np.ndarray, assignment_matrix: np.ndarray, non_obs_pos: np.ndarray):
    """
    Updates the self.connectivity maps after the last calculation.
    """
    # manipulate only non obstacle positions and leave the rest of the connectivity_matrix untouched
    for cell in non_obs_pos:
        # manipulate all layers of connectivity_matrix
        for connectid in range(connectivity_matrix.shape[0]):
            # check if assignment_matrix at position cell has the index of layer from connectivity_matrix
            if assignment_matrix[cell[0], cell[1]] == connectid:
                # check if cell needs changes and write access or not
                if not connectivity_matrix[connectid, cell[0], cell[1]] == 255:
                    connectivity_matrix[connectid, cell[0], cell[1]] = 255
            else:
                # check if cell needs changes and write access or not
                if not connectivity_matrix[connectid, cell[0], cell[1]] == 0:
                    connectivity_matrix[connectid, cell[0], cell[1]] = 0


@njit(fastmath=True)
def inverse_binary_map_as_uint8(BinaryMap: np.ndarray):
    return np.logical_not(BinaryMap).astype(np.uint8)


@njit(fastmath=True)
def normalize_euclidian_distance(RobotR, distances_map):
    MaxV = np.amax(distances_map)
    MinV = np.amin(distances_map)

    # Normalization
    if RobotR:
        # why range 1 to 2 and not 0 to 1?
        distances_map -= MinV
        distances_map /= (MaxV - MinV)
        distances_map += 1
    else:
        # range 0 to 1
        distances_map -= MinV
        distances_map /= (MaxV - MinV)


def NormalizedEuclideanDistanceBinary(RobotR: bool, BinaryMap: np.ndarray):
    """
    Calculates the euclidean distances of the tiles around a given binary(non-)robot map and normalizes it.

    :param RobotR: True: given BinaryMap is area of tiles around the robot start point (BinaryRobot); False: if BinaryNonRobot tiles area and not background
    :param BinaryMap: area of tiles as binary map
    :return: Normalized distances map of the given binary (non-/)robot map in BinaryMap.shape
    """
    distances_map = cv2.distanceTransform(inverse_binary_map_as_uint8(BinaryMap), distanceType=2, maskSize=0, dstType=5)
    normalize_euclidian_distance(RobotR, distances_map)

    return distances_map


@njit(fastmath=True)
def euclidian_distance_points2d(array1: np.array, array2: np.array) -> np.float_:
    return (
                   ((array1[0] - array2[0]) ** 2) +
                   ((array1[1] - array2[1]) ** 2)
           ) ** 0.5  # faster function with faster sqrt


# TODO make numba compatible! output has inappropriate values?
#  at some point ArrayOfElements has at least one entry -1 (from/in assign func)
#  maybe doesn't work with numba? Don't know... implemented mask now to see if it solves this problem
def normalize_metric_matrix(non_obs_pos: np.ndarray, area_bool: np.ndarray, metric_matrix: np.ndarray):
    mask = np.where(area_bool)
    metric_matrix_mask = metric_matrix[:, mask[0], mask[1]]
    maxV = np.amax(metric_matrix_mask)
    minV = np.amin(metric_matrix_mask)
    new_metric_matrix = np.empty(metric_matrix.shape, dtype=np.float_)

    for cell in non_obs_pos:
        new_metric_matrix[:, cell[0], cell[1]] = metric_matrix[:, cell[0], cell[1]] - minV
        new_metric_matrix[:, cell[0], cell[1]] /= (maxV - minV)
        new_metric_matrix[:, cell[0], cell[1]] *= 10 ** 6
    return new_metric_matrix


@njit(fastmath=True)
def check_for_near_float64_overflow(metric_matrix: np.ndarray):
    if np.amax(metric_matrix) > (float_overflow):
        return True
    else:
        return False


@njit(cache=True, fastmath=True)
def construct_assignment_matrix(area_bool: np.ndarray,
                                initial_positions: np.ndarray,
                                desireable_tile_assignment: np.ndarray):
    rows, cols = area_bool.shape
    notiles = rows * cols

    non_obstacle_positions = np.argwhere(area_bool)
    num_init_pos = initial_positions.shape[0]
    effective_size = non_obstacle_positions.shape[0] - num_init_pos  # all assignable tiles

    if effective_size % num_init_pos != 0:
        term_thr = 1
    else:
        term_thr = 0

    diff_tiles = effective_size - np.sum(desireable_tile_assignment)

    # ATTENTION! must exit DARP before this point, if diff_tiles > 0 (see check_start_parameter func)
    if diff_tiles < 0:
        while int(desireable_tile_assignment[-1] + diff_tiles) < 0:
            # tiles to cover by last startpoint unnecessary -> reduce startpoint count to maximize efficiency
            # remove tiles_count of last desireable_tile_assignment entry from diff_tiles
            diff_tiles += desireable_tile_assignment[-1]
            # then remove last entries in
            initial_positions = initial_positions[:-1]
            desireable_tile_assignment = desireable_tile_assignment[:-1]
            # do so many times until int(desireable_tile_assignment[-1] + diff_tiles) is greater 0

        # last robot won't get its full desireable_tile_assignment tiles_count
        desireable_tile_assignment[-1] += diff_tiles

    metrics_array = np.zeros((num_init_pos, rows, cols), dtype=np.float_)
    importance_array = np.zeros((num_init_pos, rows, cols), dtype=np.float_)
    max_importance = np.zeros(num_init_pos, dtype=np.float_)
    min_importance = np.full(num_init_pos, np.finfo(np.float64).max)

    for cell in non_obstacle_positions:
        tempSum = 0
        for idx in range(num_init_pos):
            metrics_array[idx, cell[0], cell[1]] = euclidian_distance_points2d(initial_positions[idx], cell)
            tempSum += metrics_array[idx, cell[0], cell[1]]

        for idx in range(num_init_pos):
            if tempSum - metrics_array[idx, cell[0], cell[1]] != 0:
                importance_array[idx, cell[0], cell[1]] = 1 / (tempSum - metrics_array[idx, cell[0], cell[1]])
            else:
                importance_array[idx, cell[0], cell[1]] = 1

            if importance_array[idx, cell[0], cell[1]] > max_importance[idx]:
                max_importance[idx] = importance_array[idx, cell[0], cell[1]]

            if importance_array[idx, cell[0], cell[1]] < min_importance[idx]:
                min_importance[idx] = importance_array[idx, cell[0], cell[1]]

    return metrics_array, non_obstacle_positions, term_thr, notiles, initial_positions, desireable_tile_assignment, importance_array, min_importance, max_importance, effective_size


@njit(cache=True, fastmath=True)
def getBinaryRobotRegions(binary_robot_regions: np.ndarray,
                          non_obs_pos: np.ndarray,
                          final_assignment_matrix: np.ndarray):
    """
    Generate a Bool Matrix for every robot's tile area.
    :return: Manipulates BinaryRobotRegions
    """
    for cell in non_obs_pos:
        for i in range(binary_robot_regions.shape[0]):
            if i == final_assignment_matrix[cell[0], cell[1]]:
                binary_robot_regions[i, cell[0], cell[1]] = True


@njit(fastmath=True)
def check_assignment_state(thresh: int,
                           connected_robot_regions: np.ndarray,
                           desirable_tile_assignment: np.ndarray,
                           current_tile_assignment: np.ndarray):
    """
    Determines if the finishing criterion of the DARP algorithm is met.
    :param current_tile_assignment:
    :param desirable_tile_assignment:
    :param thresh: Sets the possible difference between the number of tiles per robot and their desired assignment
    :param connected_robot_regions: needs array of 'is the tile area of robot x fully connected' or not
    :return: True, if criteria fits; False, if criteria aren't met
    """
    for idx, r in enumerate(connected_robot_regions):
        if np.absolute(desirable_tile_assignment[idx] - current_tile_assignment[idx]) > thresh or not \
                connected_robot_regions[idx]:
            return False
    return True


class DARP:
    def __init__(self, area_bool: np.ndarray, max_iter: np.uint32, cc_variation: float, random_level: float,
                 dynamic_cells: np.uint32, dict_darp_startparameter: dict, seed_value, importance: bool,
                 visualization: bool, video_export: bool, import_file_name: str):

        print("Following dam file will be processed: " + import_file_name)
        print("Grid Dimensions: ", str(area_bool.shape))
        print("DARP Start Parameter: ", dict_darp_startparameter)
        print("Random Seed:", seed_value)
        print("Maximum Iterations: " + str(max_iter))
        print("Dynamic Cells Count: " + str(dynamic_cells))
        print("Importance: " + str(importance))
        print("ConnectedMultiplierMatrix Variation: " + str(cc_variation))
        print("Random Influence Number: " + str(random_level))

        # start performance analyse
        # profiler = Profiler()
        # profiler.start()
        ###########################

        # check the robot start positions and tile_counts per startpoint
        if check_start_parameter(dict_darp_startparameter, area_bool):
            print("The start parameters look alright. Continue...")
            self.init_robot_pos = []
            self.DesirableAssign = np.zeros(len(dict_darp_startparameter.items()))
            for p_id, p_info in dict_darp_startparameter.items():
                self.init_robot_pos.append((p_info['row'], p_info['col']))
                self.DesirableAssign[p_id] = p_info['tiles_count']

        else:
            print("Aborting DARP;  start parameter check failed!")
            sys.exit(1)

        if not check_array_continuity(area_bool):
            print("Given area is divided into several not connected segments. Abort!")
            sys.exit(2)

        self.rows, self.cols = area_bool.shape
        self.visualization = visualization  # should the results get presented in pygame
        self.video_export = video_export  # should steps of changes in the assignment matrix get written down
        self.MaxIter = max_iter
        self.ConnectedMultiplier_variation = cc_variation
        self.randomLevel = random_level
        self.Dynamic_Cells = dynamic_cells
        self.Importance = importance
        self.import_file_name = import_file_name

        self.A = np.full((self.rows, self.cols), len(self.init_robot_pos))
        self.GridEnv_bool = area_bool
        measure_start = time.time()
        self.MetricMatrix, self.non_obstacle_positions, self.termThr, self.Notiles, self.init_robot_pos, self.DesirableAssign, self.TilesImportance, self.MinimumImportance, self.MaximumImportance, self.effectiveTileNumber = construct_assignment_matrix(
            self.GridEnv_bool, np.asarray(self.init_robot_pos), self.DesirableAssign)
        measure_end = time.time()
        print("Measured time construct_assignment_matrix(): ", (measure_end - measure_start), " sec")

        self.connectivity = np.zeros((len(self.init_robot_pos), self.rows, self.cols), dtype=np.uint8)
        self.BinaryRobotRegions = np.full((len(self.init_robot_pos), self.rows, self.cols), False, dtype=bool)
        self.ArrayOfElements = np.zeros(len(self.init_robot_pos))
        self.ConnectedRobotRegions = np.full(len(self.init_robot_pos), False, dtype=bool)

        self.color = []
        for _ in self.init_robot_pos:
            self.color.append(list(np.random.choice(range(256), size=3)))

        # End performance analyses
        # profiler.stop()
        # profiler.print(color=True)
        # profiler.open_in_browser()
        ##########################

        if self.visualization:
            self.assignment_matrix_visualization = darp_area_visualization(self.A, len(self.init_robot_pos),
                                                                           self.color, self.init_robot_pos)

        if self.video_export:
            movie_file_path = Path("result_export", self.import_file_name + ".gif")
            if not movie_file_path.parent.exists():
                os.makedirs(movie_file_path.parent)
            self.gif_writer = imageio.get_writer(movie_file_path, mode='I', duration=0.15)

        if None:
            self.seed_value = None
        else:
            if seed_value > 0:
                self.seed_value = seed_value
                seed(self.seed_value)  # correct numba seeding

        measure_start = time.time()
        self.success, self.absolute_iterations = self.update()
        measure_end = time.time()
        print("Elapsed time update(): ", (measure_end - measure_start), "sec")

    def update(self):
        success = False
        criterionMatrix = np.zeros((self.rows, self.cols))
        absolut_iterations = 0  # absolute iterations number which were needed to find optimal result

        assign(self.non_obstacle_positions, self.A, self.MetricMatrix, self.ArrayOfElements)

        # to reduce overall tile reassignment value over time in self.DesirableAssign:
        # after assigning tiles as voronoi diagram that the lowest value of self.ArrayOfElements should match
        # to the lowest entry in self.DesirableAssign... small optimization from the start but not necessary
        if self.DesirableAssign.max() > self.DesirableAssign.min():
            arrayofelements_lowest_val_idx = self.ArrayOfElements.argmin()
            desirableassign_lowest_val_idx = self.DesirableAssign.argmin()
            print("Rearranging lowest value in DesirableAssign to match lowest value in ArrayOfElements!")
            if arrayofelements_lowest_val_idx != desirableassign_lowest_val_idx:
                temp = self.DesirableAssign[desirableassign_lowest_val_idx]
                self.DesirableAssign[desirableassign_lowest_val_idx] = self.DesirableAssign[
                    arrayofelements_lowest_val_idx]
                self.DesirableAssign[arrayofelements_lowest_val_idx] = temp

        if self.video_export:
            self.video_export_add_frame(absolut_iterations, self.ConnectedRobotRegions)

        if self.visualization:
            self.assignment_matrix_visualization.placeCells()

        print("update() Start:\nDesirable Assignments:", self.DesirableAssign,
              ", Tiles per Robot:", self.ArrayOfElements, "\nTermination threshold: max", self.termThr,
              "tiles difference per robot to desirable value.")

        time_start = time.time()
        while self.termThr <= self.Dynamic_Cells and not success:
            downThres = (self.Notiles - self.termThr * (len(self.init_robot_pos) - 1)) / (
                    self.Notiles * len(self.init_robot_pos))
            upperThres = (self.Notiles + self.termThr) / (self.Notiles * len(self.init_robot_pos))

            # main optimization loop
            for _ in tqdm(range(self.MaxIter)):

                # start performance analyse
                # profiler = Profiler()
                # profiler.start()
                ###########################

                ConnectedMultiplierArrays = np.ones((len(self.init_robot_pos), self.rows, self.cols))
                plainErrors = np.zeros((len(self.init_robot_pos)))
                divFairError = np.zeros((len(self.init_robot_pos)))

                update_connectivity(self.connectivity, self.A, self.non_obstacle_positions)
                for idx, robot in enumerate(self.init_robot_pos):
                    ConnectedMultiplier = np.ones((self.rows, self.cols))
                    self.ConnectedRobotRegions[idx] = True
                    num_labels, labels_im = cv2.connectedComponents(self.connectivity[idx, :, :], connectivity=4)
                    if num_labels > 2:
                        self.ConnectedRobotRegions[idx] = False
                        BinaryRobot, BinaryNonRobot = construct_binary_images(self.non_obstacle_positions, labels_im,
                                                                              robot)
                        ConnectedMultiplier = calc_connected_multiplier(self.non_obstacle_positions,
                                                                        self.ConnectedMultiplier_variation,
                                                                        NormalizedEuclideanDistanceBinary(True,
                                                                                                          BinaryRobot),
                                                                        NormalizedEuclideanDistanceBinary(False,
                                                                                                          BinaryNonRobot))
                    ConnectedMultiplierArrays[idx, :, :] = ConnectedMultiplier
                    plainErrors[idx] = self.ArrayOfElements[idx] / (
                            self.DesirableAssign[idx] * len(self.init_robot_pos))
                    if plainErrors[idx] < downThres:
                        divFairError[idx] = downThres - plainErrors[idx]
                    elif plainErrors[idx] > upperThres:
                        divFairError[idx] = upperThres - plainErrors[idx]

                TotalNegPerc = 0
                totalNegPlainErrors = 0
                correctionMult = np.zeros(len(self.init_robot_pos))

                for idx, robot in enumerate(self.init_robot_pos):
                    if divFairError[idx] < 0:
                        TotalNegPerc += np.absolute(divFairError[idx])
                        totalNegPlainErrors += plainErrors[idx]
                    correctionMult[idx] = 1

                # Restore Fairness among the different partitions
                for idx, robot in enumerate(self.init_robot_pos):
                    if totalNegPlainErrors != 0:
                        if divFairError[idx] < 0:
                            correctionMult[idx] = 1 + (plainErrors[idx] / totalNegPlainErrors) * (TotalNegPerc / 2)
                        else:
                            correctionMult[idx] = 1 - (plainErrors[idx] / totalNegPlainErrors) * (TotalNegPerc / 2)

                        criterionMatrix = calculateCriterionMatrix(self.Importance,
                                                                   self.TilesImportance[idx],
                                                                   self.MinimumImportance[idx],
                                                                   self.MaximumImportance[idx],
                                                                   correctionMult[idx],
                                                                   divFairError[idx] < 0)

                    FinalUpdateOnMetricMatrix(
                        self.non_obstacle_positions,
                        criterionMatrix,
                        self.MetricMatrix[idx],
                        ConnectedMultiplierArrays[idx, :, :],
                        self.randomLevel)

                assign(self.non_obstacle_positions, self.A, self.MetricMatrix, self.ArrayOfElements)

                absolut_iterations += 1
                if self.video_export:
                    self.video_export_add_frame(absolut_iterations, self.ConnectedRobotRegions)

                if self.visualization:
                    self.assignment_matrix_visualization.placeCells(iteration_number=absolut_iterations)
                    # time.sleep(0.1)

                if check_for_near_float64_overflow(self.MetricMatrix):
                    self.MetricMatrix = normalize_metric_matrix(self.non_obstacle_positions, self.GridEnv_bool,
                                                                self.MetricMatrix)
                    print("\nMetricMatrix normalized")

                if check_assignment_state(self.termThr, self.ConnectedRobotRegions,
                                          self.DesirableAssign, self.ArrayOfElements):
                    time_stop = time.time()
                    success = True
                    if self.video_export:
                        self.gif_writer.close()
                    print("Found Final Assignment Matrix:",
                          absolut_iterations, "Iterations in", (time_stop - time_start),
                          "sec;", absolut_iterations / (time_stop - time_start), "iter/sec",
                          "\nDesirable Assignments:", self.DesirableAssign, "\nTiles per Robot:", self.ArrayOfElements)
                    break

                # End performance analyses
                # profiler.stop()
                # profiler.print(color=True)
                # profiler.open_in_browser()
                ##########################

            # next iteration of DARP with increased flexibility
            if not success:
                if self.MaxIter > 10000:
                    self.MaxIter = int(self.MaxIter / 2)
                self.termThr += 10
                print("\nIncreasing termination threshold to", self.termThr, "\n")

        getBinaryRobotRegions(self.BinaryRobotRegions, self.non_obstacle_positions, self.A)
        return success, absolut_iterations

    def video_export_add_frame(self, iteration: int, connected_regions: np.ndarray, draw_meta_infos=False):
        framerate = 5  # every 5th iteration

        if (iteration % framerate) == 0 or iteration == 0:
            uint8_array = np.uint8(np.interp(self.A, (self.A.min(), self.A.max()), (0, 255)))  # TODO interpolate or scale?
            temp_img = Image.fromarray(uint8_array)  # mode="RGB"
            if draw_meta_infos:  # if drawn pictures are big enough: set True to view darp metadata in gif
                font = ImageFont.truetype("arial.ttf", 9)
                txt = f'{time.strftime("%H:%M:%S %d.%m.%Y")}\nInitial positions:\n{str(self.init_robot_pos)}\nSeed: {str(self.seed_value)}\nRandom Influence: {self.randomLevel}\nCriterion Matrix Variation: {self.ConnectedMultiplier_variation}\nImportance: {self.Importance}\nDesired Assignment:\n{str(self.DesirableAssign)}\nAssignment per Robot:\n{str(self.ArrayOfElements)}\nTiles Connected:\n{str(connected_regions)}\nIteration: {iteration}'
                ImageDraw.Draw(temp_img).multiline_text((3, 3), txt, spacing=2, font=font)
            self.gif_writer.append_data(np.asarray(temp_img))
