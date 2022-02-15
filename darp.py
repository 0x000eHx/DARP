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
from PIL import Image
from pathlib import Path
import os
from pyinstrument import Profiler
from numba import njit, prange
from numba.typed import List

np.set_printoptions(threshold=sys.maxsize)


# @njit
def check_start_positions(start_positions, area: np.ndarray):
    for position in start_positions:
        # TODO check if there are obstacle cells (False) around start position cell as area boundary check
        if not area[position]:
            # print("Start position: " + str(position) + " is not inside the given area!")
            return False
    return True


# @njit
def check_portions(start_positions, portions):
    if len(start_positions) != len(portions):
        # print("Number of portions and robot start positions don't match. One portion should be defined for each drone.")
        return False
    s = sum(portions)
    if abs(s - 1) >= 0.0001:
        # print("Sum of portions has to be 1!")
        return False
    return True


@njit  # parallel=True cache=True, fastmath=True
def assign(start_positions, area_shape, Assignment_Matrix, Grid_Enviroment, Metric_Matrix):
    rows, cols = area_shape

    ArrayOfElements = np.zeros(len(start_positions))

    for i in prange(rows):
        for j in prange(cols):

            # if non obstacle tile
            if Grid_Enviroment[i, j] == -1:
                minV = Metric_Matrix[0, i, j]  # finding minimal value from here on (argmin)
                indMin = 0  # number of assigned robot of tile (i,j)
                for idx, robot in enumerate(start_positions):
                    if Metric_Matrix[idx, i, j] < minV:
                        # the actual decision making if distance of tile is lower for one robo startpoint than to another
                        minV = Metric_Matrix[idx, i, j]
                        indMin = idx
                Assignment_Matrix[i][j] = indMin
                ArrayOfElements[indMin] += 1

            # if obstacle tile
            elif Grid_Enviroment[i, j] == -2:
                Assignment_Matrix[i, j] = len(start_positions)

    return Assignment_Matrix, ArrayOfElements


@njit
def generateRandomMatrix(random_level: float, area_shape: tuple):
    """
    Generates a matrix in area_shape with a random value for every tiles (around 1)
    :return: RandomMatrix
    """
    RandomMatrix = 2 * random_level * np.random.uniform(0, 1, size=area_shape) + (1 - random_level)
    return RandomMatrix


@njit  # (parallel=True)
def FinalUpdateOnMetricMatrix(criterionMatrix, currentMetricMatrix: np.ndarray, ConnectedMultiplierMatrix,
                              random_level: float):
    """
    Calculates the Final Metric Matrix with criterionMatrix, RandomMatrix, MetricMatrix, ConnectedMultiplierList
    :param random_level:
    :param criterionMatrix: criterionMatrix
    :param currentMetricMatrix: current MetricMatrix of chosen robot which needs to get modified
    :param ConnectedMultiplierMatrix: ConnectedMultiplierMatrix of chosen robot
    :return: new MetricMatrix
    """
    metric_matrix_new = currentMetricMatrix * criterionMatrix * generateRandomMatrix(random_level, currentMetricMatrix.shape) * ConnectedMultiplierMatrix
    return metric_matrix_new


@njit  # (parallel=True)
def CalcConnectedMultiplier(cc_variation, dist1, dist2):
    """
    Calculates the ConnectedMultiplier between the binary robot tiles (connected area) and the binary non-robot tiles
    :param cc_variation:
    :param dist1: Must contain the euclidean distances of all tiles around the binary robot tiles
    :param dist2: Must contain the euclidean distances of all tiles around the binary non-robot tiles
    :return: ConnectedMultiplier array in the shape of the whole area (dist1.shape & dist2.shape)
    """
    returnM = dist1 - dist2  # 2 numpy ndarray subtracted, shortform of np.subtract, returns a freshly allocated array
    MaxV = np.max(returnM)
    MinV = np.min(returnM)

    returnM = (returnM - MinV) * ((2 * cc_variation) / (MaxV - MinV)) + (1 - cc_variation)

    return returnM


@njit  # (parallel=True)  # , fastmath=True
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


# TODO consider numba object mode
def NormalizedEuclideanDistanceBinary(RobotR, BinaryMap):
    """
    Calculates the euclidean distances of the tiles around a given binary(non-)robot map and normalizes it.
    :param RobotR: True: given BinaryMap is area of tiles around the robot start point (BinaryRobot); False: if BinaryNonRobot tiles area and not background
    :param BinaryMap: area of tiles as binary map
    :return: Normalized distances map of the given binary (non-/)robot map in BinaryMap.shape
    """
    distances_map = ndimage.distance_transform_edt(np.logical_not(BinaryMap))
    MaxV = np.max(distances_map)
    MinV = np.min(distances_map)

    # Normalization
    if RobotR:
        distances_map = ((distances_map - MinV) / (MaxV - MinV)) + 1  # why range 1 to 2 and not 0 to 1?
    else:
        distances_map = ((distances_map - MinV) / (MaxV - MinV))  # range 0 to 1

    return distances_map


@njit(parallel=True)
def gen_cust_dist_func(kernel_inner, kernel_outer, parallel=True):
    kernel_inner_nb = njit(kernel_inner, fastmath=True, inline='always')
    kernel_outer_nb = njit(kernel_outer, fastmath=True, inline='always')

    def cust_dot_T(A, B):
        assert B.shape[1] == A.shape[1]

        out = np.empty((A.shape[0], B.shape[0]), dtype=A.dtype)

        for i in prange(A.shape[0]):
            for j in prange(B.shape[0]):
                acc = 0
                for k in prange(A.shape[1]):
                    acc += kernel_inner_nb(A[i, k], B[j, k])
                out[i, j] = kernel_outer_nb(acc)
        return out

    if parallel:
        return njit(cust_dot_T, fastmath=True, parallel=True)
    else:
        return njit(cust_dot_T, fastmath=True, parallel=False)


@njit
def euclidian_distance(first, second):
    inner = lambda first, second: (first - second) ** 2
    outer = lambda acc: np.sqrt(acc)
    return gen_cust_dist_func(inner, outer, parallel=True)


@njit  # (parallel=True)  # fastmath=True, cache=True
def construct_Assignment_Matrix(area_shape, initial_positions, obstacle_number, portions):
    rows, cols = area_shape
    Notiles = rows * cols
    effectiveSize = Notiles - len(initial_positions) - obstacle_number
    termThr = 0

    if effectiveSize % len(initial_positions) != 0:
        termThr = 1

    DesireableAssign = np.zeros(len(initial_positions))
    MaximunDist = np.zeros(len(initial_positions))
    MaximumImportance = np.zeros(len(initial_positions))
    MinimumImportance = np.zeros(len(initial_positions))
    AllDistances = np.zeros((len(initial_positions), rows, cols))
    TilesImportance = np.zeros((len(initial_positions), rows, cols))

    for idx in prange(len(initial_positions)):
        DesireableAssign[idx] = effectiveSize * portions[idx]
        if DesireableAssign[idx] != int(DesireableAssign[idx]) and termThr != 1:
            termThr = 1  # threshold value of tiles which can be freely moved between assigned robot areas

    for x in prange(rows):
        for y in prange(cols):
            tempSum = 0
            for idx, robot in enumerate(initial_positions):
                temp = np.subtract(np.array(robot), np.array((x, y))).astype(np.float64)
                AllDistances[idx, x, y] = np.linalg.norm(temp)   # euclidian_distance(np.array(robot), np.array((x, y)))  # E!
                tempSum += AllDistances[idx, x, y]

            for idx, robot in enumerate(initial_positions):
                if tempSum - AllDistances[idx, x, y] != 0:
                    TilesImportance[idx, x, y] = 1 / (tempSum - AllDistances[idx, x, y])
                else:
                    TilesImportance[idx, x, y] = 1

    for idx in prange(len(initial_positions)):
        MaximunDist[idx] = AllDistances[idx, :, :].max()
        MaximumImportance[idx] = TilesImportance[idx, :, :].max()
        MinimumImportance[idx] = TilesImportance[idx, :, :].min()

    return AllDistances, termThr, Notiles, DesireableAssign, TilesImportance, MinimumImportance, MaximumImportance, effectiveSize


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
        self.import_geometry_file_name = import_file_name
        self.MaxIter = max_iter
        self.ConnectedMultiplierMatrix_variation = cc_variation
        self.randomLevel = random_level
        self.Dynamic_Cells = dynamic_cells
        self.Importance = importance

        self.A = np.zeros((self.rows, self.cols))

        empty_space = []  # TODO if extra restricted area is necessary later
        self.GridEnv, self.Obstacle_Number = self.defineGridEnv(area_bool, empty_space)
        self.AllDistances, self.termThr, self.Notiles, self.DesireableAssign, self.TilesImportance, self.MinimumImportance, self.MaximumImportance, effectiveTileNumber = construct_Assignment_Matrix(
            area_bool.shape, List(start_positions), self.Obstacle_Number, List(portions))
        print("Effective number of tiles: ", effectiveTileNumber)

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
        # profiler.print(color=True, show_all=True)
        # profiler.open_in_browser()
        ##########################

        if self.visualization:
            self.assignment_matrix_visualization = darp_area_visualization(self.A, len(self.init_robot_pos), self.color,
                                                                           self.init_robot_pos)

        if self.video_export:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            gif_file_name = timestamp + "_" + str(self.import_geometry_file_name) + "_start" + str(self.init_robot_pos)\
                            + "_portions" + str(portions) + "_rand" + str(self.randomLevel)\
                            + "_ccvar" + str(self.ConnectedMultiplierMatrix_variation)\
                            + "_imp" + str(self.Importance) + ".gif"

            b = {' ': '', '.geojson': ''}
            for x, y in b.items():
                gif_file_name = gif_file_name.replace(x, y)

            movie_file_path = Path("result_export", gif_file_name)
            if not movie_file_path.parent.exists():
                os.makedirs(movie_file_path.parent)
            self.gif_writer = imageio.get_writer(movie_file_path, mode='i', duration=0.3)

        self.success = self.update()

    def defineGridEnv(self, area: np.ndarray, empty_space):

        local_grid_env = np.full(shape=(self.rows, self.cols), fill_value=-1)  # create non obstacle map with value -1

        # initial robot tiles will have their array.index as value
        for idx, position in enumerate(self.init_robot_pos):
            local_grid_env[position] = idx
            self.A[position] = idx

        # obstacle tiles value is -2
        obstacle_positions = np.where(area == False)
        obstacle_num = len(obstacle_positions[0])
        local_grid_env[obstacle_positions] = -2

        for idx, es_pos in enumerate(empty_space):
            local_grid_env[es_pos] = -2

        return local_grid_env, obstacle_num

    def video_export_add_frame(self, iteration=0):

        framerate = 2
        write_frame = (iteration % framerate) == 0

        if write_frame or iteration == 0:
            uint8_array = np.uint8(np.interp(self.A, (self.A.min(), self.A.max()), (0, 255)))
            temp_img = Image.fromarray(uint8_array)
            self.gif_writer.append_data(np.asarray(temp_img))

    def update(self):
        success = False
        criterionMatrix = np.zeros((self.rows, self.cols))
        absolut_iterations = 0  # absolute iterations number which were needed to find optimal result

        self.A, self.ArrayOfElements = assign(List(self.init_robot_pos),
                                              (self.rows, self.cols),
                                              self.A,
                                              self.GridEnv,
                                              self.MetricMatrix)

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
                    self.update_connectivity()
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

                        criterionMatrix = self.calculateCriterionMatrix(
                            self.TilesImportance[idx],
                            self.MinimumImportance[idx],
                            self.MaximumImportance[idx],
                            correctionMult[idx],
                            divFairError[idx] < 0)

                    self.MetricMatrix[idx] = FinalUpdateOnMetricMatrix(
                        criterionMatrix,
                        self.MetricMatrix[idx],
                        ConnectedMultiplierList[idx, :, :],
                        self.randomLevel)

                self.A, self.ArrayOfElements = assign(List(self.init_robot_pos),
                                                      (self.rows, self.cols),
                                                      self.A,
                                                      self.GridEnv,
                                                      self.MetricMatrix)
                iteration += 1

                if self.video_export:
                    self.video_export_add_frame(iteration)

                if self.visualization:
                    self.assignment_matrix_visualization.placeCells(iteration_number=iteration)
                    # time.sleep(0.1)

                # End performance analyses
                # profiler.stop()
                # profiler.open_in_browser()
                ##########################

                if self.IsThisAGoalState(self.termThr, ConnectedRobotRegions):
                    print("\nFinal Assignment Matrix (" + str(iteration) + " Iterations, Tiles per Robot " + str(
                        self.ArrayOfElements) + ")")
                    success = True
                    absolut_iterations += iteration
                    break

            # next iteration of DARP with increased flexibility
            if not success:
                absolut_iterations += self.MaxIter
                self.MaxIter = int(self.MaxIter / 2)
                self.termThr += 1

        self.getBinaryRobotRegions()
        return success

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

    def update_connectivity(self):
        """
        Updates the self.connectivity maps after the last calculation.
        :return: Nothing
        """
        for idx, r in enumerate(self.init_robot_pos):
            self.connectivity[idx] = np.where(self.A == idx, 255, 0)

    def calculateCriterionMatrix(self, TilesImportance, MinimumImportance, MaximumImportance, correctionMult,
                                 smallerthan0):
        """
        Generates a new correction multiplier matrix.
        If self.Importance is True: TilesImportance influence is calculated.
        :param TilesImportance:
        :param MinimumImportance:
        :param MaximumImportance:
        :param correctionMult:
        :param smallerthan0:
        :return: returnCrit
        """
        if self.Importance:
            if smallerthan0:
                returnCrit = (TilesImportance - MinimumImportance) * (
                        (correctionMult - 1) / (MaximumImportance - MinimumImportance)) + 1
            else:
                returnCrit = (TilesImportance - MinimumImportance) * (
                        (1 - correctionMult) / (MaximumImportance - MinimumImportance)) + correctionMult
        else:
            returnCrit = np.full(TilesImportance.shape, correctionMult)

        return returnCrit
