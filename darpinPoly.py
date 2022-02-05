from pathlib import Path
from gridding_spacial_geometries import multi_processing_geometry_boundary_check, serial_processing_geometry_boundary_check
import geopandas as gpd
from darp import DARP
import numpy as np
from kruskal import Kruskal
from CalculateTrajectories import CalculateTrajectories
from Visualization import visualize_paths
import sys
from turns import turns


class multiRobotPathPlanner(DARP):
    def __init__(self, nx, ny, MaxIter, CCvariation, randomLevel, dcells, importance, notEqualPortions, initial_positions, portions, obstacles_positions, visualization, image_export, map_name):
        DARP.__init__(self, nx, ny, MaxIter, CCvariation, randomLevel, dcells, importance, notEqualPortions, initial_positions, portions, obstacles_positions, visualization, image_export, map_name)

        if not self.success:
            print("DARP did not manage to find a solution for the given configuration!")
            sys.exit(4)

        mode_to_drone_turns = dict()

        for mode in range(4):
            MSTs = self.calculateMSTs(self.BinaryRobotRegions, len(self.init_robot_pos), self.rows, self.cols, mode)
            AllRealPaths = []
            for idx, robot in enumerate(self.init_robot_pos):
                ct = CalculateTrajectories(self.rows, self.cols, MSTs[idx])
                ct.initializeGraph(self.CalcRealBinaryReg(self.BinaryRobotRegions[idx], self.rows, self.cols), True)
                ct.RemoveTheAppropriateEdges()
                ct.CalculatePathsSequence(4 * robot[0] * self.cols + 2 * robot[1])
                AllRealPaths.append(ct.PathSequence)

            TypesOfLines = np.zeros((self.rows*2, self.cols*2, 2))
            for idx, robot in enumerate(self.init_robot_pos):
                flag = False
                for connection in AllRealPaths[idx]:
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

            drone_turns = turns(AllRealPaths)
            drone_turns.count_turns()
            mode_to_drone_turns[mode] = drone_turns

            if self.visualization:
                image = visualize_paths(AllRealPaths, subCellsAssignment, len(self.init_robot_pos), self.color)
                image.visualize_paths(mode)

        print("\nResults:\n")
        for mode, val in mode_to_drone_turns.items():
            print(mode, val)

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
        for idx, robot in enumerate(self.init_robot_pos):
            k = Kruskal(rows, cols)
            k.initializeGraph(self.BinaryRobotRegions[idx, :, :], True, mode)
            k.performKruskal()
            MSTs.append(k.mst)
        return MSTs


def get_area_indices(area, value, inv=False):
    if inv:
        return np.concatenate([np.where((area != value))]).T
    return np.concatenate([np.where((area == value))]).T


def get_random_start_points(number_of_start_points: int, area_array: np.ndarray, obstacle=False):
    start_coordinates = []

    for i in range(number_of_start_points):
        rows, cols = area_array.shape
        random_row = np.random.randint(0, rows)

        for ix, column in enumerate(area_array[random_row]):
            if area_array[random_row][ix]:
                start_coordinates.append((random_row, ix))
                break

    return start_coordinates


if __name__ == '__main__':

    talsperre_geojsonfile = Path("Talsperren_einzeln_geojson_files", "Talsperre Malter.geojson")
    gdf_talsperre = gpd.read_file(talsperre_geojsonfile)

    talsperre_gdf_exploded = gdf_talsperre.geometry.explode().tolist()
    max_area_talsperre = max(talsperre_gdf_exploded, key=lambda a: a.area)

    multiprocessing = True

    if multiprocessing:
        grid_cells = multi_processing_geometry_boundary_check(max_area_talsperre, gdf_talsperre)

    else:
        grid_cells = serial_processing_geometry_boundary_check(max_area_talsperre, gdf_talsperre)

    obstacles_positions = get_area_indices(grid_cells, value=False)

    rows, cols = grid_cells.shape
    start_points = get_random_start_points(3, grid_cells)

    not_equal_portions = True  # this trigger should be True, if the portions are not equal

    if not_equal_portions:
        portions = [0.4, 0.3, 0.3]
    else:
        portions = []
        for idx, drone in enumerate(start_points):
            portions.append(1 / len(start_points))

    if len(start_points) != len(portions):
        print("Portions should be defined for each drone")
        sys.exit(1)

    s = sum(portions)
    if abs(s-1) >= 0.0001:
        print("Sum of portions should be equal to 1.")
        sys.exit(2)

    # TODO fix startpoint sanity check and move to DARP
    # for start_point in start_points:
    #    transformed = np.array(start_point).T
    #    if transformed in obstacles_positions:
    #        print("Initial robot start position should not be on obstacle.")
    #        print("Problems at following init position: " + str(start_point))
    #        sys.exit(3)

    MaxIter = 10000
    CCvariation = 0.01
    randomLevel = 0.0001
    dcells = 30
    importance = False
    visualize = False
    image_export = True

    print("\nInitial Conditions Defined:")
    print("Grid Dimensions:", rows, cols)
    print("Robot Number:", len(start_points))
    print("Initial Robots' positions", start_points)
    print("Portions for each Robot:", portions, "\n")

    poly = multiRobotPathPlanner(rows, cols, MaxIter, CCvariation, randomLevel, dcells, importance, not_equal_portions, start_points, portions, obstacles_positions, visualize, image_export, talsperre_geojsonfile.stem)
