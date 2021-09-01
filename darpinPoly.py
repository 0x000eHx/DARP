from darp import DARP
import numpy as np
from kruskal import Kruskal
from CalculateTrajectories import CalculateTrajectories
from Visualization import visualize_paths
import sys
import random
import argparse


class DARPinPoly(DARP):
    def __init__(self, nx, ny, MaxIter, CCvariation, randomLevel, dcells, importance, notEqualPortions, initial_positions, portions, obstacles_positions):
        DARP.__init__(self, nx, ny, MaxIter, CCvariation, randomLevel, dcells, importance, notEqualPortions, initial_positions, portions, obstacles_positions)

        if not self.success:
            print("DARP did not manage to find a solution for the given configuration!")
            sys.exit(0)

        for mode in range(4):
            MSTs = self.calculateMSTs(self.BinaryRobotRegions, self.droneNo, self.rows, self.cols, mode)
            AllRealPaths = []
            for r in range(self.droneNo):
                ct = CalculateTrajectories(self.rows, self.cols, MSTs[r])
                ct.initializeGraph(self.CalcRealBinaryReg(self.BinaryRobotRegions[r], self.rows, self.cols), True)
                ct.RemoveTheAppropriateEdges()
                ct.CalculatePathsSequence(4 * self.init_robot_pos[r][0] * self.cols + 2 * self.init_robot_pos[r][1])
                AllRealPaths.append(ct.PathSequence)

            TypesOfLines = np.zeros((self.rows*2, self.cols*2, 2))
            for r in range(self.droneNo):
                flag = False
                for connection in AllRealPaths[r]:
                    if flag:
                        if TypesOfLines[connection[0]][connection[1]][0] == 0:
                            indxadd1 = 0
                        else:
                            indxadd1 = 1

                        if TypesOfLines[connection[2]][connection[3]][0] == 0 and flag:
                            indxadd2 = 0
                        else:
                            indxadd2 = 1
                    else:
                        if not (TypesOfLines[connection[0]][connection[1]][0] == 0):
                            indxadd1 = 0
                        else:
                            indxadd1 = 1
                        if not (TypesOfLines[connection[2]][connection[3]][0] == 0 and flag):
                            indxadd2 = 0
                        else:
                            indxadd2 = 1

                    flag = True
                    if connection[0] == connection[2]:
                        if connection[1] > connection[3]:
                            TypesOfLines[connection[0]][connection[1]][indxadd1] = 2
                            TypesOfLines[connection[2]][connection[3]][indxadd2] = 3
                        else:
                            TypesOfLines[connection[0]][connection[1]][indxadd1] = 3
                            TypesOfLines[connection[2]][connection[3]][indxadd2] = 2

                    else:
                        if (connection[0] > connection[2]):
                            TypesOfLines[connection[0]][connection[1]][indxadd1] = 1
                            TypesOfLines[connection[2]][connection[3]][indxadd2] = 4
                        else:
                            TypesOfLines[connection[0]][connection[1]][indxadd1] = 4
                            TypesOfLines[connection[2]][connection[3]][indxadd2] = 1

            subCellsAssignment = np.zeros((2*self.rows, 2*self.cols))
            for i in range(self.rows):
                for j in range(self.cols):
                    subCellsAssignment[2 * i][2 * j] = self.A[i][j]
                    subCellsAssignment[2 * i + 1][2 * j] = self.A[i][j]
                    subCellsAssignment[2 * i][2 * j + 1] = self.A[i][j]
                    subCellsAssignment[2 * i + 1][2 * j + 1] = self.A[i][j]

            image = visualize_paths(AllRealPaths, subCellsAssignment, self.droneNo, self.color)
            # image.visualize_darp_area()
            image.visualize_paths(mode)

    def CalcRealBinaryReg(self, BinaryRobotRegion, rows, cols):
        temp = np.zeros((2*rows, 2*cols))
        RealBinaryRobotRegion = np.zeros((2 * rows, 2 * cols), dtype=bool)
        for i in range(2*rows):
            for j in range(2*cols):
                temp[i, j] = BinaryRobotRegion[(int(i / 2))][(int(j / 2))]
                if temp[i, j] == 0:
                    RealBinaryRobotRegion[i, j] = False
                else:
                    RealBinaryRobotRegion[i, j] = True

        return RealBinaryRobotRegion

    def calculateMSTs(self, BinaryRobotRegions, droneNo, rows, cols, mode):
        MSTs = []
        for r in range(self.droneNo):
            k = Kruskal(rows, cols)
            k.initializeGraph(self.BinaryRobotRegions[r, :, :], True, mode)
            k.performKruskal()
            MSTs.append(k.mst)
        return MSTs


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-Grid',
        default=(10, 10),
        type=int,
        nargs=2,
        help='Dimensions of the Grid (default: (10, 10))')
    argparser.add_argument(
        '-ObsPos',
        default=[5, 6, 7],
        nargs='*',
        type=int,
        help='Dimensions of the Grid (default: (10, 10))')
    argparser.add_argument(
        '-InPos',
        default=[1, 3, 9],
        nargs='*',
        type=int,
        help='Initial Positions of the robots (default: (1, 3, 9))')
    argparser.add_argument(
        '-NEP',
        action='store_true',
        help='Not Equal Portions shared between the Robots in the Grid (default: False)')
    argparser.add_argument(
        '-Portions',
        default=[0.6, 0.7, 0.1],
        nargs='*',
        type=float,
        help='Portion for each Robot in the Grid (default: (0.2, 0.7, 0.1))')
    args = argparser.parse_args()

    rows, cols = args.Grid

    cell = 0
    obstacles_positions = []
    initial_positions = []
    for row in range(rows):
        for col in range(cols):
            cell += 1
            for obstacle in args.ObsPos:
                if cell == obstacle:
                    obstacles_positions.append((row, col))
            for position in args.InPos:
                if cell == position:
                    initial_positions.append((row, col))

    portions = []
    print(bool(args.NEP))
    if args.NEP:
        input("a")
        for portion in args.Portions:
            portions.append(portion)
    else:
        for drone in range(len(initial_positions)):
            portions.append(1/len(initial_positions))

    if len(initial_positions) != len(args.Portions):
        print("Portions should be defined for each drone")
        sys.exit(4)

    s = sum(portions)
    if abs(s-1) >= 0.0001:
        print("Sum of portions should be equal to 1.")
        sys.exit(1)

    for position in initial_positions:
        if (position[0] < 0 or position[0] >= rows) or (position[1] < 0 or position[1] >= cols):
            print("Initial positions should be inside the Grid.")
            sys.exit(2)
        for obstacle in obstacles_positions:
            if position[0] == obstacle[0] and position[1] == obstacle[1]:
                print("Initial positions should not be on obstacles")
                sys.exit(3)

    for obstacle in obstacles_positions:
        if (obstacle[0] < 0 or obstacle[0] >= rows) or (obstacle[1] < 0 or obstacle[1] >= cols):
            print("Obstacles should be inside the Grid.")
            sys.exit(3)

    MaxIter = 80000
    CCvariation = 0.01
    randomLevel = 0.0001
    dcells = 2
    importance = False

    print("")
    print("\nInitial Conditions Defined:")
    print("Grid Dimensions:", rows, cols)
    print("Robot Number:", len(initial_positions))
    print("Initial Robots' positions", initial_positions)
    print("Portions for each Robot:", portions, "\n")
    print("")

    poly = DARPinPoly(rows, cols, MaxIter, CCvariation, randomLevel, dcells, importance, args.NEP, initial_positions, portions, obstacles_positions)
