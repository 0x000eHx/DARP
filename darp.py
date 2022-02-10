# import matplotlib.pyplot
import numpy as np
import copy
import sys
import cv2
from scipy import ndimage
from Visualization import darp_area_visualization
# import time
from tqdm.auto import tqdm
import imageio
from PIL import Image
from pathlib import Path
import os
# from pyinstrument import Profiler

np.set_printoptions(threshold=sys.maxsize)


def check_start_positions(start_positions: list[tuple], area: np.array):

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


class DARP:
    def __init__(self, area_bool: np.ndarray, max_iter: np.uint32, cc_variation: float, random_level: float, dynamic_cells: np.uint32, importance: bool, start_positions: list[tuple], portions: list, visualization: bool, video_export: bool, import_file_name: str):

        # start performance analyse
        # profiler = Profiler()
        # profiler.start()
        ###########################

        # check the robot start positions, is any of them situated on obstacle tile?
        if not check_start_positions(start_positions, area_bool):
            print("Aborting after start positions check")
            sys.exit(1)
        else:
            print("Robot start positions are inside given area, continuing...")
            self.init_robot_pos = start_positions

        # check the portions, are there enough so every start position has one?
        if not check_portions(start_positions, portions):
            print("Aborting after portion check")
            sys.exit(2)
        else:
            print("Portions and number compared to robot start positions work out, continuing...")
            self.Rportions = portions

        self.rows, self.cols = area_bool.shape
        self.effectiveSize = 0

        self.visualization = visualization  # should the results get presented in pygame
        self.video_export = video_export  # should steps of changes in the assignment matrix get written down
        self.import_geometry_file_name = import_file_name

        self.A = np.zeros((self.rows, self.cols))

        empty_space = []  # TODO if extra restricted area is necessary later
        self.GridEnv, self.Obstacle_Number = self.defineGridEnv(area_bool, empty_space)

        self.MaxIter = max_iter
        self.CC_variation = cc_variation
        self.randomLevel = random_level
        self.Dynamic_Cells = dynamic_cells
        self.Importance = importance
        self.connectivity = np.zeros((len(self.init_robot_pos), self.rows, self.cols), dtype=np.uint8)
        self.BinaryRobotRegions = np.full((len(self.init_robot_pos), self.rows, self.cols), False, dtype=bool)

        self.AllDistances, self.termThr, self.Notiles, self.DesireableAssign, self.TilesImportance, self.MinimumImportance, self.MaximumImportance = self.construct_Assignment_Matrix()

        self.MetricMatrix = copy.deepcopy(self.AllDistances)
        self.BWlist = np.zeros((len(self.init_robot_pos), self.rows, self.cols))
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
            self.assignment_matrix_visualization = darp_area_visualization(self.A, len(self.init_robot_pos), self.color, self.init_robot_pos)

        if self.video_export:
            # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            movie_file_path = Path("result_export", str(self.import_geometry_file_name).replace(" " and ".geojson", "") + "-DARP_animation.gif")
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
        # listOfCoordinates = list(zip(obstacle_positions[0], obstacle_positions[1]))
        local_grid_env[obstacle_positions] = -2

        for idx, es_pos in enumerate(empty_space):
            local_grid_env[es_pos] = -2

        return local_grid_env, obstacle_num

    def video_export_add_frame(self, iteration):

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

                self.assign()

                if self.video_export:
                    self.video_export_add_frame(iteration)

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
                        BinaryRobot, BinaryNonRobot = self.constructBinaryImages(labels_im, robot)
                        ConnectedMultiplier = self.CalcConnectedMultiplier(
                            self.NormalizedEuclideanDistanceBinary(True, BinaryRobot),
                            self.NormalizedEuclideanDistanceBinary(False, BinaryNonRobot))
                    ConnectedMultiplierList[idx, :, :] = ConnectedMultiplier
                    plainErrors[idx] = self.ArrayOfElements[idx] / (
                            self.DesireableAssign[idx] * len(self.init_robot_pos))
                    if plainErrors[idx] < downThres:
                        divFairError[idx] = downThres - plainErrors[idx]
                    elif plainErrors[idx] > upperThres:
                        divFairError[idx] = upperThres - plainErrors[idx]

                if self.IsThisAGoalState(self.termThr, ConnectedRobotRegions):
                    print("\nFinal Assignment Matrix (" + str(iteration) + " Iterations, Tiles per Robot " + str(self.ArrayOfElements) + ")")
                    success = True
                    absolut_iterations += iteration
                    break

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

                    self.MetricMatrix[idx] = self.FinalUpdateOnMetricMatrix(
                        criterionMatrix,
                        self.MetricMatrix[idx],
                        ConnectedMultiplierList[idx, :, :])

                # End performance analyses
                # profiler.stop()
                # profiler.open_in_browser()
                ##########################

                iteration += 1
                if self.visualization:
                    self.assignment_matrix_visualization.placeCells(iteration_number=iteration)
                    # time.sleep(0.1)

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

    def generateRandomMatrix(self):
        """
        Generates a matrix in map.shape with a random value for every tiles (around 1)
        :return: RandomMatrix
        """
        RandomMatrix = 2 * self.randomLevel * np.random.uniform(0, 1, size=(self.rows, self.cols)) + (
                1 - self.randomLevel)
        return RandomMatrix

    def FinalUpdateOnMetricMatrix(self, CM, currentMetricMatrix, CC):
        """
        Calculates the Final Metric Matrix with criterionMatrix, RandomMatrix, MetricMatrix, ConnectedMultiplierList
        :param CM: criterionMatrix
        :param currentMetricMatrix: current MetricMatrix of chosen robot which needs to get modified
        :param CC: ConnectedMultiplierMatrix of chosen robot
        :return: new MetricMatrix
        """
        MMnew = currentMetricMatrix * CM * self.generateRandomMatrix() * CC
        return MMnew

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

    def constructBinaryImages(self, area_tiles, robot_start_point):
        """
        Returns 2 maps in the given area_tiles.shape
        - robot_tiles_binary: where all tiles around + robot_start_point are ones, the rest is zero
        - nonrobot_tiles_binary: where tiles which aren't background and not around the robot_start_point are ones, rest is zero
        :param area_tiles: map of tiles with at least 3 different labels, 0 must always be the background
        :param robot_start_point: is needed to determine which area of connected tiles should be BinaryRobot area
        :return: robot_tiles_binary, nonrobot_tiles_binary
        """
        # area_map where all tiles with the value of the robot_start_point are 1s end the rest is 0
        robot_tiles_binary = np.where(area_tiles == area_tiles[robot_start_point], 1, 0)

        # background in area_tiles always has the value 0
        nonrobot_tiles_binary = np.where((area_tiles > 0) & (area_tiles != area_tiles[robot_start_point]), 1, 0)
        return robot_tiles_binary, nonrobot_tiles_binary

    def assign(self):
        self.BWlist = np.zeros((len(self.init_robot_pos), self.rows, self.cols))
        for idx, robot in enumerate(self.init_robot_pos):
            self.BWlist[idx, robot[0], robot[1]] = 1

        self.ArrayOfElements = np.zeros(len(self.init_robot_pos))

        for i in range(self.rows):
            for j in range(self.cols):
                # if non obstacle tile
                if self.GridEnv[i, j] == -1:
                    minV = self.MetricMatrix[0, i, j]  # finding minimal value from here on (argmin)
                    indMin = 0  # number of assigned robot of tile (i,j)
                    for idx, robot in enumerate(self.init_robot_pos):
                        if self.MetricMatrix[idx, i, j] < minV:
                            # the actual decision making if distance of tile is lower for one robo startpoint than to another
                            minV = self.MetricMatrix[idx, i, j]
                            indMin = idx

                    self.A[i][j] = indMin
                    self.BWlist[indMin, i, j] = 1
                    self.ArrayOfElements[indMin] += 1

                # if obstacle tile
                elif self.GridEnv[i, j] == -2:
                    self.A[i, j] = len(self.init_robot_pos)

    def construct_Assignment_Matrix(self):

        Notiles = self.rows * self.cols
        self.effectiveSize = Notiles - len(self.init_robot_pos) - self.Obstacle_Number
        print("Effective number of tiles: " + str(self.effectiveSize))
        termThr = 0

        if self.effectiveSize % len(self.init_robot_pos) != 0:
            termThr = 1

        DesireableAssign = np.zeros(len(self.init_robot_pos))
        MaximunDist = np.zeros(len(self.init_robot_pos))
        MaximumImportance = np.zeros(len(self.init_robot_pos))
        MinimumImportance = np.zeros(len(self.init_robot_pos))

        for idx, robot in enumerate(self.init_robot_pos):
            DesireableAssign[idx] = self.effectiveSize * self.Rportions[idx]
            MinimumImportance[idx] = sys.float_info.max
            if DesireableAssign[idx] != int(DesireableAssign[idx]) and termThr != 1:
                termThr = 1  # threshold value of tiles which can be freely moved between assigned robot areas

        AllDistances = np.zeros((len(self.init_robot_pos), self.rows, self.cols))
        TilesImportance = np.zeros((len(self.init_robot_pos), self.rows, self.cols))

        # TODO OPTIMIZE THIS MONSTER
        for x in range(self.rows):
            for y in range(self.cols):
                tempSum = 0
                for idx, robot in enumerate(self.init_robot_pos):
                    AllDistances[idx, x, y] = np.linalg.norm(np.array(robot) - np.array((x, y)))  # E!
                    if AllDistances[idx, x, y] > MaximunDist[idx]:
                        MaximunDist[idx] = AllDistances[idx, x, y]
                    tempSum += AllDistances[idx, x, y]

                for idx, robot in enumerate(self.init_robot_pos):
                    if tempSum - AllDistances[idx, x, y] != 0:
                        TilesImportance[idx, x, y] = 1 / (tempSum - AllDistances[idx, x, y])
                    else:
                        TilesImportance[idx, x, y] = 1
                    # Todo FixMe!
                    if TilesImportance[idx, x, y] > MaximumImportance[idx]:
                        MaximumImportance[idx] = TilesImportance[idx, x, y]

                    if TilesImportance[idx, x, y] < MinimumImportance[idx]:
                        MinimumImportance[idx] = TilesImportance[idx, x, y]

        return AllDistances, termThr, Notiles, DesireableAssign, TilesImportance, MinimumImportance, MaximumImportance

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

    def CalcConnectedMultiplier(self, dist1, dist2):
        """
        Calculates the ConnectedMultiplier between the binary robot tiles (connected area) and the binary non-robot tiles
        :param dist1: Must contain the euclidean distances of all tiles around the binary robot tiles
        :param dist2: Must contain the euclidean distances of all tiles around the binary non-robot tiles
        :return: ConnectedMultiplier array in the shape of the whole area (dist1.shape & dist2.shape)
        """
        returnM = dist1 - dist2  # 2 numpy ndarray subtracted, shortform of np.subtract, returns a freshly allocated array
        MaxV = np.max(returnM)
        MinV = np.min(returnM)

        returnM = (returnM - MinV) * ((2 * self.CC_variation) / (MaxV - MinV)) + (1 - self.CC_variation)

        return returnM

    def NormalizedEuclideanDistanceBinary(self, RobotR, BinaryMap):
        """
        Calculates the euclidean distances of the tiles around a given binary(non-)robot map and normalizes it.
        :param RobotR: True: given BinaryMap is area of tiles around the robot start point (BinaryRobot); False: if BinaryNonRobot tiles area and not background
        :param BinaryMap: area of tiles as binary map
        :return: Normalized distances map of the given binary (non-/)robot map in BinaryMap.shape
        """
        distRobot = ndimage.morphology.distance_transform_edt(np.logical_not(BinaryMap))
        MaxV = np.max(distRobot)
        MinV = np.min(distRobot)

        # Normalization
        if RobotR:
            distRobot = ((distRobot - MinV) / (MaxV - MinV)) + 1  # why range 1 to 2 and not 0 to 1?
        else:
            distRobot = ((distRobot - MinV) / (MaxV - MinV))  # range 0 to 1

        return distRobot

