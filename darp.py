# import matplotlib.pyplot
import numpy as np
import copy
import sys
import cv2
from scipy import ndimage
from Visualization import darp_area_visualization
import time
from tqdm.auto import tqdm
import imageio
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
import os
from pyinstrument import Profiler
from numba import njit
from numba.typed import List

np.set_printoptions(threshold=sys.maxsize)
random.seed(1)
os.environ['PYTHONHASHSEED'] = str(1)
np.random.seed(1)


def check_start_positions(start_positions, area: np.ndarray):
    for position in start_positions:
        # TODO check if there are obstacle cells (False) around start position cell as area boundary check
        if not area[position]:
            print("Start position: " + str(position) + " is not inside the given area!")
            return False
    return True


def check_portions(start_positions, portions):
    if len(start_positions) != len(portions):
        print("Number of portions and robot start positions don't match. One portion should be defined for each drone.")
        return False
    s = sum(portions)
    if abs(s - 1) >= 0.0001:
        print("Sum of portions has to be 1!")
        return False
    return True


@njit(cache=True, fastmath=True)  # (fastmath=True)  # cache=True, fastmath=True
def assign(Assignment_Matrix, area_bool, Metric_Matrix, ArrayOfElements):

    for r in range(Assignment_Matrix.shape[0]):
        for c in range(Assignment_Matrix.shape[1]):
            if area_bool[r, c]:
                Assignment_Matrix[r, c] = np.argmin(Metric_Matrix[:, r, c])

    for i in range(Metric_Matrix.shape[0]):
        ArrayOfElements[i] = np.count_nonzero(Assignment_Matrix == i) - 1  # -1 for the start position of robot i


@njit(cache=True, fastmath=True)
def generateRandomMatrix(random_level: float, area_shape: tuple):
    """
    Generates a matrix in area_shape with a random value for every tiles (around 1)
    :return: RandomMatrix
    """
    return 2 * random_level * np.random.uniform(0, 1, size=area_shape) + (1 - random_level)


@njit(cache=True, fastmath=True)
def FinalUpdateOnMetricMatrix(criterionMatrix, MetricMatrix: np.ndarray, ConnectedMultiplierMatrix,
                              random_level: float):
    """
    Calculates the Final Metric Matrix with criterionMatrix, RandomMatrix, MetricMatrix, ConnectedMultiplierList
    :param random_level:
    :param criterionMatrix: criterionMatrix
    :param MetricMatrix: current MetricMatrix of chosen robot which needs to get modified
    :param ConnectedMultiplierMatrix: ConnectedMultiplierMatrix of chosen robot
    :return: new MetricMatrix
    """
    MetricMatrix *= criterionMatrix * ConnectedMultiplierMatrix * generateRandomMatrix(random_level, MetricMatrix.shape)


@njit(cache=True, fastmath=True)
def CalcConnectedMultiplier(cc_variation, dist1, dist2):
    """
    Calculates the ConnectedMultiplier between the binary robot tiles (connected area) and the binary non-robot tiles
    :param cc_variation:
    :param dist1: Must contain the euclidean distances of all tiles around the binary robot tiles
    :param dist2: Must contain the euclidean distances of all tiles around the binary non-robot tiles
    :return: ConnectedMultiplier array in the shape of the whole area (dist1.shape & dist2.shape)
    """
    returnM = np.subtract(dist1, dist2)
    MaxV = np.max(returnM)
    MinV = np.min(returnM)
    return (returnM - MinV) * ((2 * cc_variation) / (MaxV - MinV)) + (1 - cc_variation)


@njit(cache=True, fastmath=True)
def calculateCriterionMatrix(importance_trigger, TilesImportance, MinimumImportance,
                             MaximumImportance, correctionMult, smallerthan_zero):
    """
    Generates a new correction multiplier matrix.
    If self.Importance is True: TilesImportance influence is calculated.
    """
    returnCrit = np.zeros(TilesImportance.shape)
    if importance_trigger:
        if smallerthan_zero:
            returnCrit = (TilesImportance - MinimumImportance) * (
                        (correctionMult - 1) / (MaximumImportance - MinimumImportance)) + 1
        else:
            returnCrit = (TilesImportance - MinimumImportance) * (
                        (1 - correctionMult) / (MaximumImportance - MinimumImportance)) + correctionMult
    else:
        returnCrit[:, :] = correctionMult
    return returnCrit


@njit(cache=True, fastmath=True)  # parallel=True, fastmath=True
def constructBinaryImages(area_tiles, robot_start_point):
    """
    Returns 2 maps in the given area_tiles.shape

    - robot_tiles_binary: where all tiles around + robot_start_point are ones, the rest is zero

    - nonrobot_tiles_binary: where tiles which aren't background and not around the robot_start_point are ones, rest is zero

    :param area_tiles: map of tiles with at least 3 different labels, 0 should always be the background value
    :param robot_start_point: is needed to determine which area of connected tiles should be BinaryRobot area
    :return: robot_tiles_binary, nonrobot_tiles_binary
    """
    # area_map where all tiles with the value of the robot_start_point are 1s end the rest is 0
    robot_tiles_binary = np.where(area_tiles == area_tiles[robot_start_point], 1, 0)

    # background in area_tiles always has the value 0
    nonrobot_tiles_binary = np.where((area_tiles > 0) & (area_tiles != area_tiles[robot_start_point]), 1, 0)
    return robot_tiles_binary, nonrobot_tiles_binary


@njit(cache=True, fastmath=True)
def update_connectivity(connectivity_matrix: np.ndarray, assignment_matrix: np.ndarray):
    """
    Updates the self.connectivity maps after the last calculation.
    :return: Nothing
    """
    for idx in range(connectivity_matrix.shape[0]):
        connectivity_matrix[idx] = np.where(assignment_matrix == idx, 255, 0)


@njit(cache=True, fastmath=True)
def inverse_binary_map_as_uint8(BinaryMap):
    return np.logical_not(BinaryMap).astype(np.uint8)


def NormalizedEuclideanDistanceBinary(RobotR, BinaryMap):
    """
    Calculates the euclidean distances of the tiles around a given binary(non-)robot map and normalizes it.

    :param RobotR: True: given BinaryMap is area of tiles around the robot start point (BinaryRobot); False: if BinaryNonRobot tiles area and not background
    :param BinaryMap: area of tiles as binary map
    :return: Normalized distances map of the given binary (non-/)robot map in BinaryMap.shape
    """
    distances_map = cv2.distanceTransform(inverse_binary_map_as_uint8(BinaryMap), distanceType=2, maskSize=0, dstType=5)

    MaxV = np.amax(distances_map)
    MinV = np.amin(distances_map)

    # Normalization
    if RobotR:
        distances_map = ((distances_map - MinV) / (MaxV - MinV)) + 1  # why range 1 to 2 and not 0 to 1?
    else:
        distances_map = ((distances_map - MinV) / (MaxV - MinV))  # range 0 to 1
    return distances_map


@njit(cache=True, fastmath=True)
def euclidian_distance_points2d(array1: np.array, array2: np.array) -> np.float_:
    return (
                   ((array1[0] - array2[0]) ** 2) +
                   ((array1[1] - array2[1]) ** 2)
           ) ** 0.5  # faster function with faster sqrt


@njit(cache=True, fastmath=True)  # (parallel=True) fastmath=True
def construct_Assignment_Matrix(area_bool: np.ndarray, initial_positions: List, portions: List):
    rows, cols = area_bool.shape
    Notiles = rows * cols

    obstacle_positions = np.where(~area_bool)
    non_obstacle_positions = np.where(area_bool)
    effectiveSize = Notiles - len(initial_positions) - len(obstacle_positions[0])
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
        print("Initial Conditions Defined:")
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

        self.rows, self.cols = area_bool.shape

        self.visualization = visualization  # should the results get presented in pygame
        self.video_export = video_export  # should steps of changes in the assignment matrix get written down
        self.MaxIter = max_iter
        self.ConnectedMultiplierMatrix_variation = cc_variation
        self.randomLevel = random_level
        self.Dynamic_Cells = dynamic_cells
        self.Importance = importance
        self.import_file_name = import_file_name

        self.A = np.full((self.rows, self.cols), len(self.init_robot_pos))

        self.GridEnv_bool = area_bool
        measure_start = time.time()
        self.AllDistances, self.no_obstacle_position, self.termThr, self.Notiles, self.DesireableAssign, self.TilesImportance, self.MinimumImportance, self.MaximumImportance, self.effectiveTileNumber = construct_Assignment_Matrix(
            self.GridEnv_bool, List(start_positions), List(portions))
        measure_end = time.time()
        print("Elapsed time construct_Assignment_Matrix(): ", (measure_end - measure_start), " sec")
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

        measure_start = time.time()
        self.success, self.absolute_iterations = self.update()
        measure_end = time.time()
        print("Elapsed time update(): ", (measure_end - measure_start), "sec")

    def video_export_add_frame(self, iteration=0):
        framerate = 5
        write_frame = (iteration % framerate) == 0

        if write_frame or iteration == 0:
            uint8_array = np.uint8(np.interp(self.A, (self.A.min(), self.A.max()), (0, 255)))
            temp_img = Image.fromarray(uint8_array)  # mode="RGB"
            font = ImageFont.truetype("arial.ttf", 9)
            txt = f'{time.strftime("%H:%M:%S %d.%m.%Y")}\nInitial positions: {str(self.init_robot_pos)}\nPortions: {str(self.Rportions)}\nRandom Influence: {self.randomLevel}\nCriterion Matrix Variation: {self.ConnectedMultiplierMatrix_variation}\nImportance: {self.Importance}\nIterations: {iteration}'
            ImageDraw.Draw(temp_img).multiline_text((3, 3), txt, spacing=2, font=font, fill="red")  # fill="red"
            self.gif_writer.append_data(np.asarray(temp_img))

    def update(self):
        success = False
        criterionMatrix = np.zeros((self.rows, self.cols))
        absolut_iterations = 0  # absolute iterations number which were needed to find optimal result

        assign(self.A, self.GridEnv_bool, self.MetricMatrix, self.ArrayOfElements)

        if self.video_export:
            self.video_export_add_frame()

        if self.visualization:
            self.assignment_matrix_visualization.placeCells()
            # time.sleep(0.1)

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

                ConnectedMultiplierList = np.ones((len(self.init_robot_pos), self.rows, self.cols))
                ConnectedRobotRegions = np.full(len(self.init_robot_pos), False, dtype=bool)
                plainErrors = np.zeros((len(self.init_robot_pos)))
                divFairError = np.zeros((len(self.init_robot_pos)))

                for idx, robot in enumerate(self.init_robot_pos):
                    ConnectedMultiplier = np.ones((self.rows, self.cols))
                    ConnectedRobotRegions[idx] = True
                    update_connectivity(self.connectivity, self.A)
                    num_labels, labels_im = cv2.connectedComponents(self.connectivity[idx, :, :], connectivity=4)
                    if num_labels > 2:
                        ConnectedRobotRegions[idx] = False
                        BinaryRobot, BinaryNonRobot = constructBinaryImages(labels_im, robot)
                        ConnectedMultiplier = CalcConnectedMultiplier(self.ConnectedMultiplierMatrix_variation,
                                                                      NormalizedEuclideanDistanceBinary(True,
                                                                                                        BinaryRobot),
                                                                      NormalizedEuclideanDistanceBinary(False,
                                                                                                        BinaryNonRobot))
                    ConnectedMultiplierList[idx, :, :] = ConnectedMultiplier
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
                        criterionMatrix,
                        self.MetricMatrix[idx],
                        ConnectedMultiplierList[idx, :, :],
                        self.randomLevel)

                assign(self.A, self.GridEnv_bool, self.MetricMatrix, self.ArrayOfElements)

                iteration += 1

                if self.video_export:
                    self.video_export_add_frame(iteration)

                if self.visualization:
                    self.assignment_matrix_visualization.placeCells(iteration_number=iteration)
                    # time.sleep(0.1)

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
