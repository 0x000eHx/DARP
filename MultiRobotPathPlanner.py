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
import moviepy.editor as mp


class MultiRobotPathPlanner(DARP):
    def __init__(self, np_bool_area: np.ndarray, max_iter: np.uint32, cc_variation: float, random_level: float,
                 dynamic_cells: np.uint32, dict_darp_start: dict, seed, importance: bool, visualization,
                 image_export, video_export, export_file_name):

        self.export_file_name = export_file_name

        DARP.__init__(self, np_bool_area, max_iter, cc_variation, random_level, dynamic_cells, dict_darp_start, seed,
                      importance, visualization, video_export, self.export_file_name)

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
        num_of_processes = os.cpu_count() - 1
        clip = mp.VideoFileClip("./result_export/" + self.export_file_name + ".gif")
        clip.write_videofile("./result_export/" + self.export_file_name + ".mp4", audio=False, threads=num_of_processes)
        clip.close()
        print("Created MP4 video from assignment matrix animation GIF file")
