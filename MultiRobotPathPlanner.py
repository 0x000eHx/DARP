from pathlib import Path
from darp import DARP
import numpy as np
from kruskal import Kruskal
from CalculateTrajectories import CalculateTrajectories
from Visualization import visualize_paths
import sys
from turns import turns
import matplotlib.pyplot as plt
import os
import time
import imageio
from tqdm.auto import tqdm


def generate_file_name(filename: str, initial_positions, max_tiles, seed, random, cc_var, importance):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    export_file_name = timestamp + "_" + str(filename) + "_start" + str(initial_positions) \
                       + "_maxtiles" + str(max_tiles) + "_seed" + str(seed) + "_rand" + str(random) \
                       + "_ccvar" + str(cc_var) \
                       + "_imp" + str(importance)
    # Replace all characters in dict
    b = {' ': '', '.geojson': ''}
    for x, y in b.items():
        export_file_name = export_file_name.replace(x, y)
    return export_file_name


class MultiRobotPathPlanner(DARP):
    def __init__(self, area, max_iter, cc_variation, random_level, dynamic_cells, max_tiles_pr, seed, importance,
                 start_positions, visualization, image_export, import_file_name, video_export):

        self.export_file_name = generate_file_name(import_file_name, start_positions, max_tiles_pr, seed, random_level,
                                                   cc_variation, importance)

        DARP.__init__(self, area, max_iter, cc_variation, random_level, dynamic_cells, max_tiles_pr, seed, importance,
                      start_positions, visualization, video_export, self.export_file_name)

        if not self.success:
            print("DARP did not manage to find a solution for the given configuration!")
            sys.exit(3)

        if image_export:
            self.to_image()

        if video_export:
            self.to_video()

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

            TypesOfLines = np.zeros((self.rows * 2, self.cols * 2, 2))
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

            subCellsAssignment = np.zeros((2 * self.rows, 2 * self.cols))
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
        temp = np.zeros((2 * rows, 2 * cols))
        RealBinaryRobotRegion = np.zeros((2 * rows, 2 * cols), dtype=bool)
        for i in range(2 * rows):
            for j in range(2 * cols):
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

    def to_image(self):
        file_path = Path('result_export', self.export_file_name + ".jpg")
        if not file_path.parent.exists():
            os.makedirs(file_path.parent)
        plt.imsave(file_path, self.A, dpi=100)
        print("Exported image of final assignment matrix")

    def to_video(self):
        # existing gif in results_export folder?
        input_path = Path("result_export", self.export_file_name + ".gif")
        reader = imageio.get_reader(input_path)
        output_path = Path("result_export", self.export_file_name + ".mp4")
        writer = imageio.get_writer(output_path)
        for i, im in tqdm(enumerate(reader)):
            writer.append_data(im)
        writer.close()
        print("Created .mp4 video from assignment matrix animation .gif file")
