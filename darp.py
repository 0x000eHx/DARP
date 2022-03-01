import numpy as np
import copy
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
from numba.typed import List

np.set_printoptions(threshold=sys.maxsize)


@njit
def seed(a=42):
    np.random.seed(a)


def check_start_positions(start_positions, area: np.ndarray):
    for position in start_positions:
        if not (0 <= position[0] < area.shape[0]) or not (0 <= position[1] < area.shape[1]):
            print("Start position: " + str(position) + " is not inside the given area!")
            return False
        elif not area[position]:
            print("Start position: " + str(position) + " is on a obstacle tile inside the given area!")
            return False
    return True


def check_portions(start_positions, portions):

    if len(start_positions) != len(portions):
        print("Number of portions and robot start positions don't match. One portion should be defined for each drone.")
        return False
    elif abs(sum(portions) - 1) >= 0.0001:
        print("Sum of portions has to be 1!")
        return False
    else:
        for one in portions:
            if one > 0.5:
                print("One portion is bigger than 50%. This could increase the amount of tiles to rearrange and calculation time!")
            if one < 0:
                print("No negative portion value allowed!")
                return False
        return True


def check_array_continuity(area: np.ndarray):
    connectivity_img = np.zeros(area.shape, dtype=np.uint8)

    mask = np.where(area)
    connectivity_img[mask[0], mask[1]] = 255
    num_labels, labels_im = cv2.connectedComponents(image=connectivity_img, connectivity=4)
    if num_labels > 2:
        print("The given area MUST not have unreachable and/or closed shape regions!")
        return False
    else:
        return True


@njit(cache=True, fastmath=True)  # cache=True
def assign(non_obs_pos: np.ndarray, Assignment_Matrix: np.ndarray, Metric_Matrix: np.ndarray, ArrayOfElements: np.ndarray):
    for cell in non_obs_pos:
        # argmin index is same as index of robot in initial_positions array
        Assignment_Matrix[cell[0], cell[1]] = np.argmin(Metric_Matrix[:, cell[0], cell[1]])

    for i in range(Metric_Matrix.shape[0]):
        ArrayOfElements[i] = np.count_nonzero(Assignment_Matrix == i) - 1  # -1 for the start position of robot i


@njit(cache=True, fastmath=True)
def FinalUpdateOnMetricMatrix(non_obs_pos: np.ndarray, criterionMatrix, MetricMatrix: np.ndarray, ConnectedMultiplierMatrix, random_level: float):
    """
    Calculates the Final MetricMatrix with given criterionMatrix, Random input, MetricMatrix, ConnectedMultiplier
    """
    # manipulate only non obstacle positions and leave the rest of the MetricMatrix untouched
    for cell in non_obs_pos:
        MetricMatrix[cell[0], cell[1]] *= criterionMatrix[cell[0], cell[1]]
        MetricMatrix[cell[0], cell[1]] *= 2 * random_level * np.random.uniform(0, 1) + (1 - random_level)
        MetricMatrix[cell[0], cell[1]] *= ConnectedMultiplierMatrix[cell[0], cell[1]]


@njit(cache=True, fastmath=True)
def CalcConnectedMultiplier(non_obs_pos: np.ndarray, cc_variation, dist1, dist2):
    """
    Calculates the ConnectedMultiplier between the binary robot tiles (connected area) and the binary non-robot tiles

    :param non_obs_pos:
    :param cc_variation:
    :param dist1: Must contain the euclidean distances of all tiles around the binary robot tiles
    :param dist2: Must contain the euclidean distances of all tiles around the binary non-robot tiles
    :return: ConnectedMultiplier array in the shape of the whole area (dist1.shape & dist2.shape)
    """
    returnM = np.subtract(dist1, dist2)
    MaxV = np.max(returnM)
    MinV = np.min(returnM)

    for cell in non_obs_pos:
        returnM[cell[0], cell[1]] -= MinV
        returnM[cell[0], cell[1]] *= ((2 * cc_variation) / (MaxV - MinV))
        returnM[cell[0], cell[1]] += (1 - cc_variation)

    return returnM


@njit(cache=True, fastmath=True)
def calculateCriterionMatrix(importance_trigger, TilesImportanceMatrix, MinimumImportance,
                             MaximumImportance, correctionMult, smallerthan_zero):
    """
    Generates a new correction multiplier matrix.
    If self.Importance is True: TilesImportanceMatrix influence is calculated.
    """
    returnCrit = np.zeros(TilesImportanceMatrix.shape)
    if importance_trigger:
        if smallerthan_zero:
            returnCrit = (TilesImportanceMatrix - MinimumImportance) * (
                    (correctionMult - 1) / (MaximumImportance - MinimumImportance)) + 1
        else:
            returnCrit = (TilesImportanceMatrix - MinimumImportance) * (
                    (1 - correctionMult) / (MaximumImportance - MinimumImportance)) + correctionMult
    else:
        returnCrit[:, :] = correctionMult
    return returnCrit


@njit(cache=True, fastmath=True)  # parallel=True, fastmath=True
def constructBinaryImages(non_obs_pos: np.ndarray, area_tiles: np.ndarray, robot_start_point):
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


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
def inverse_binary_map_as_uint8(BinaryMap: np.ndarray):
    return np.logical_not(BinaryMap).astype(np.uint8)


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
def euclidian_distance_points2d(array1: np.array, array2: np.array) -> np.float_:
    return (
                   ((array1[0] - array2[0]) ** 2) +
                   ((array1[1] - array2[1]) ** 2)
           ) ** 0.5  # faster function with faster sqrt


@njit(cache=True, fastmath=True)
def normalize_metric_matrix(non_obs_pos: np.ndarray, metric_matrix: np.ndarray):
    maxV = np.amax(metric_matrix)
    minv = np.amin(metric_matrix)

    for cell in non_obs_pos:
        metric_matrix[:, cell[0], cell[1]] = 10 * (metric_matrix[:, cell[0], cell[1]] - minv) / (maxV - minv)


@njit(cache=True, fastmath=True)
def check_for_near_float64_overflow(non_obs_pos: np.ndarray, metric_matrix: np.ndarray):
    if np.amax(metric_matrix) > (np.finfo(np.float64).max / 10):
        normalize_metric_matrix(non_obs_pos, metric_matrix)
        return True
    else:
        return False


@njit(cache=True, fastmath=True)  # (parallel=True)
def construct_Assignment_Matrix(area_bool: np.ndarray, initial_positions: List, portions: List):
    rows, cols = area_bool.shape
    Notiles = rows * cols

    obstacle_positions = np.argwhere(~area_bool)
    non_obstacle_positions = np.argwhere(area_bool)
    effectiveSize = Notiles - len(initial_positions) - obstacle_positions.shape[0]
    termThr = 0

    if effectiveSize % len(initial_positions) != 0:
        termThr = 1

    DesireableAssign = np.zeros(len(initial_positions))
    MaximumImportance = np.zeros(len(initial_positions))
    MinimumImportance = np.full(len(initial_positions), np.finfo(np.float64).max)
    AllDistances = np.zeros((len(initial_positions), rows, cols))
    TilesImportance = np.zeros((len(initial_positions), rows, cols))

    for idx in range(len(initial_positions)):
        DesireableAssign[idx] = effectiveSize * portions[idx]
        if DesireableAssign[idx] != int(DesireableAssign[idx]) and termThr != 1:
            termThr = 1  # threshold value of tiles which can be freely moved between assigned robot areas

    for x in range(rows):
        for y in range(cols):
            if area_bool[x, y]:
                tempSum = 0
                for idx, robot in enumerate(initial_positions):
                    AllDistances[idx, x, y] = euclidian_distance_points2d(np.array(robot), np.array((x, y)))
                    tempSum += AllDistances[idx, x, y]

                for idx, robot in enumerate(initial_positions):
                    if tempSum - AllDistances[idx, x, y] != 0:
                        TilesImportance[idx, x, y] = 1 / (tempSum - AllDistances[idx, x, y])
                    else:
                        TilesImportance[idx, x, y] = 1

    for idx in range(len(initial_positions)):
        for x in range(rows):
            for y in range(cols):
                if area_bool[x, y]:
                    if TilesImportance[idx, x, y] > MaximumImportance[idx]:
                        MaximumImportance[idx] = TilesImportance[idx, x, y]
                    elif TilesImportance[idx, x, y] < MinimumImportance[idx]:
                        MinimumImportance[idx] = TilesImportance[idx, x, y]

    return AllDistances, non_obstacle_positions, termThr, Notiles, DesireableAssign, TilesImportance, MinimumImportance, MaximumImportance, effectiveSize


class DARP:
    def __init__(self, area_bool: np.ndarray, max_iter: np.uint32, cc_variation: float, random_level: float,
                 dynamic_cells: np.uint32, importance: bool, start_positions: list[tuple], portions: list,
                 visualization: bool, video_export: bool, import_file_name: str):

        print("Following dam file will be processed: " + import_file_name)
        print("Grid Dimensions: ", str(area_bool.shape))
        print("Robot Number: ", len(start_positions))
        print("Initial Robot positions: ", start_positions)
        print("Portions for each Robot:", portions)
        print("Maximum Iterations: " + str(max_iter))
        print("Dynamic Cells Count: " + str(dynamic_cells))
        print("Importance: " + str(importance))
        print("ConnectedMultiplierMatrix Variation: " + str(cc_variation))
        print("Random Influence Number: " + str(random_level))

        # start performance analyse
        # profiler = Profiler()
        # profiler.start()
        ###########################

        # check the robot start positions, is any of them situated on obstacle tile?
        if not check_start_positions(List(start_positions), area_bool):
            print("Aborting after start positions check")
            sys.exit(1)
        else:
            print("Robot start positions are inside given area, continuing...")
            self.init_robot_pos = start_positions

        # check the portions, are there enough so every start position has one?
        if not check_portions(List(start_positions), List(portions)):
            print("Aborting after portion check")
            sys.exit(2)
        else:
            print("Portions and number compared to robot start positions work out, continuing...")
            self.Rportions = portions

        if not check_array_continuity(area_bool):
            print("Given area is divided into several not connected segments. Abort!")
            sys.exit(3)

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
        self.AllDistances, self.non_obstacle_positions, self.termThr, self.Notiles, self.DesireableAssign, self.TilesImportance, self.MinimumImportance, self.MaximumImportance, self.effectiveTileNumber = construct_Assignment_Matrix(
            self.GridEnv_bool, List(start_positions), List(portions))
        measure_end = time.time()
        print("Measured time construct_Assignment_Matrix(): ", (measure_end - measure_start), " sec")
        print("Effective number of tiles: ", self.effectiveTileNumber)

        self.connectivity = np.zeros((len(self.init_robot_pos), self.rows, self.cols), dtype=np.uint8)
        self.BinaryRobotRegions = np.full((len(self.init_robot_pos), self.rows, self.cols), False, dtype=bool)
        self.MetricMatrix = copy.deepcopy(self.AllDistances)
        # self.BWlist = np.zeros((len(self.init_robot_pos), self.rows, self.cols))
        self.ArrayOfElements = np.zeros(len(self.init_robot_pos))

        self.color = []
        for robot in self.init_robot_pos:
            self.color.append(list(np.random.choice(range(256), size=3)))

        # End performance analyses
        # profiler.stop()
        # profiler.print(color=True)
        # profiler.open_in_browser()
        ##########################

        if self.visualization:
            self.assignment_matrix_visualization = darp_area_visualization(self.A, len(self.init_robot_pos), self.color,
                                                                           self.init_robot_pos)

        if self.video_export:
            movie_file_path = Path("result_export", self.import_file_name + ".gif")
            if not movie_file_path.parent.exists():
                os.makedirs(movie_file_path.parent)
            self.gif_writer = imageio.get_writer(movie_file_path, mode='i', duration=0.15)

        seed()  # correct numba seeding
        measure_start = time.time()
        self.success, self.absolute_iterations = self.update()
        measure_end = time.time()
        print("Elapsed time update(): ", (measure_end - measure_start), "sec")

    def video_export_add_frame(self, iteration=0):
        framerate = 5

        if (iteration % framerate) == 0 or iteration == 0:
            uint8_array = np.uint8(np.interp(self.A, (self.A.min(), self.A.max()), (0, 255)))  # TODO interpolate or scale?
            temp_img = Image.fromarray(uint8_array)  # mode="RGB"
            font = ImageFont.truetype("arial.ttf", 9)
            txt = f'{time.strftime("%H:%M:%S %d.%m.%Y")}\nInitial positions: {str(self.init_robot_pos)}\nPortions: {str(self.Rportions)}\nRandom Influence: {self.randomLevel}\nCriterion Matrix Variation: {self.ConnectedMultiplier_variation}\nImportance: {self.Importance}\nIterations: {iteration}'
            ImageDraw.Draw(temp_img).multiline_text((3, 3), txt, spacing=2, font=font, fill="red")  # fill="red"
            self.gif_writer.append_data(np.asarray(temp_img))

    def update(self):
        success = False
        criterionMatrix = np.zeros((self.rows, self.cols))
        absolut_iterations = 0  # absolute iterations number which were needed to find optimal result

        assign(self.non_obstacle_positions, self.A, self.MetricMatrix, self.ArrayOfElements)

        if self.video_export:
            self.video_export_add_frame()

        if self.visualization:
            self.assignment_matrix_visualization.placeCells()

        while self.termThr <= self.Dynamic_Cells and not success:
            downThres = (self.Notiles - self.termThr * (len(self.init_robot_pos) - 1)) / (
                    self.Notiles * len(self.init_robot_pos))
            upperThres = (self.Notiles + self.termThr) / (self.Notiles * len(self.init_robot_pos))

            # main optimization loop
            time_start = time.time()
            for iteration in tqdm(range(self.MaxIter)):

                # start performance analyse
                # profiler = Profiler()
                # profiler.start()
                ###########################

                ConnectedMultiplierArrays = np.ones((len(self.init_robot_pos), self.rows, self.cols))
                ConnectedRobotRegions = np.full(len(self.init_robot_pos), False, dtype=bool)
                plainErrors = np.zeros((len(self.init_robot_pos)))
                divFairError = np.zeros((len(self.init_robot_pos)))

                for idx, robot in enumerate(self.init_robot_pos):
                    ConnectedMultiplier = np.ones((self.rows, self.cols))
                    ConnectedRobotRegions[idx] = True
                    update_connectivity(self.connectivity, self.A, self.non_obstacle_positions)
                    num_labels, labels_im = cv2.connectedComponents(self.connectivity[idx, :, :], connectivity=4)
                    if num_labels > 2:
                        ConnectedRobotRegions[idx] = False
                        BinaryRobot, BinaryNonRobot = constructBinaryImages(self.non_obstacle_positions, labels_im, robot)
                        ConnectedMultiplier = CalcConnectedMultiplier(self.non_obstacle_positions,
                                                                      self.ConnectedMultiplier_variation,
                                                                      NormalizedEuclideanDistanceBinary(True,
                                                                                                        BinaryRobot),
                                                                      NormalizedEuclideanDistanceBinary(False,
                                                                                                        BinaryNonRobot))
                    ConnectedMultiplierArrays[idx, :, :] = ConnectedMultiplier
                    plainErrors[idx] = self.ArrayOfElements[idx] / (
                            self.DesireableAssign[idx] * len(self.init_robot_pos))
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

                iteration += 1

                if self.video_export:
                    self.video_export_add_frame(iteration)

                if self.visualization:
                    self.assignment_matrix_visualization.placeCells(iteration_number=iteration)
                    # time.sleep(0.1)

                if check_for_near_float64_overflow(self.non_obstacle_positions, self.MetricMatrix):
                    print("MetricMatrix normalized")

                # End performance analyses
                # profiler.stop()
                # profiler.print(color=True)
                # profiler.open_in_browser()
                ##########################

                if self.IsThisAGoalState(self.termThr, ConnectedRobotRegions):
                    time_stop = time.time()
                    success = True
                    absolut_iterations += iteration
                    if self.video_export:
                        self.gif_writer.close()
                    print("Final Assignment Matrix:\n(", absolut_iterations, "Iterations in", (time_stop-time_start), "sec (", absolut_iterations/(time_stop-time_start), "iter/sec)")
                    print("Desireable Assignment:", self.DesireableAssign, ", Tiles per Robot:", self.ArrayOfElements)
                    break

            # next iteration of DARP with increased flexibility
            if not success:
                absolut_iterations += self.MaxIter
                self.MaxIter = int(self.MaxIter / 2)
                self.termThr += 1

        self.getBinaryRobotRegions()
        return success, absolut_iterations

    def getBinaryRobotRegions(self):
        """
        Generate a Bool Matrix for every robot's tile area.
        :return: Manipulates BinaryRobotRegions
        """
        ind = np.where(self.A < len(self.init_robot_pos))
        temp = (self.A[ind].astype(int),) + ind
        self.BinaryRobotRegions[temp] = True

    def IsThisAGoalState(self, thresh, connected_robot_regions):
        """
        Determines if the finishing criterion of the DARP algorithm is met.
        :param thresh: Sets the possible difference between the number of tiles per robot and their desired assignment
        :param connected_robot_regions: needs array of 'is the tile area of robot x fully connected' or not
        :return: True, if criteria fits; False, if criteria aren't met
        """
        for idx, r in enumerate(self.init_robot_pos):
            if np.absolute(self.DesireableAssign[idx] - self.ArrayOfElements[idx]) > thresh or not connected_robot_regions[idx]:
                return False
        return True
