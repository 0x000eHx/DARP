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
import time


class MultiRobotPathPlanner(DARP):
    def __init__(self, np_bool_area: np.ndarray, max_iter: np.uint32, cc_variation: float, random_level: float,
                 dynamic_cells: np.uint32, dict_darp_start: dict, seed, importance: bool, visualization,
                 image_export, video_export, export_file_name):

        start_time = time.time()

        self.darp_instance = DARP(np_bool_area, max_iter, cc_variation, random_level, dynamic_cells, dict_darp_start,
                                  seed, importance, visualization, video_export, export_file_name)
        self.export_file_name = export_file_name

        # start dividing regions
        measure_start = time.time()
        self.darp_success, self.iterations = self.darp_instance.divideRegions()
        measure_end = time.time()
        print("Elapsed time divideRegions(): ", (measure_end - measure_start), "sec")

        if not self.darp_success:
            print("DARP did not manage to find a solution for the given configuration!")

        else:
            # don't let every darp execution write an image, only if darp actually does anything the image gets written
            if image_export and len(self.darp_instance.init_robot_pos) > 1:
                self.to_image()

            if video_export and len(self.darp_instance.init_robot_pos) > 1:
                # otherwise there will be no saved .gif file to convert
                self.to_video()

            # Iterate for 4 different ways to join edges in MST
            self.mode_to_drone_turns = []
            AllRealPaths_dict = {}
            subCellsAssignment_dict = {}
            for mode in range(4):
                MSTs = self.calculateMSTs(self.darp_instance.BinaryRobotRegions, len(self.darp_instance.init_robot_pos),
                                          self.darp_instance.rows, self.darp_instance.cols, mode)
                AllRealPaths = []
                for r in range(len(self.darp_instance.init_robot_pos)):
                    ct = CalculateTrajectories(self.darp_instance.rows, self.darp_instance.cols, MSTs[r])
                    ct.initializeGraph(
                        self.CalcRealBinaryReg(self.darp_instance.BinaryRobotRegions[r], self.darp_instance.rows,
                                               self.darp_instance.cols), True)
                    ct.RemoveTheAppropriateEdges()
                    ct.CalculatePathsSequence(
                        4 * self.darp_instance.init_robot_pos[r][0] * self.darp_instance.cols + 2 *
                        self.darp_instance.init_robot_pos[r][1])
                    AllRealPaths.append(ct.PathSequence)

                self.TypesOfLines = np.zeros((self.darp_instance.rows * 2, self.darp_instance.cols * 2, 2))
                for r in range(len(self.darp_instance.init_robot_pos)):
                    flag = False
                    for connection in AllRealPaths[r]:
                        if flag:
                            if self.TypesOfLines[connection[0]][connection[1]][0] == 0:
                                indxadd1 = 0
                            else:
                                indxadd1 = 1

                            if self.TypesOfLines[connection[2]][connection[3]][0] == 0 and flag:
                                indxadd2 = 0
                            else:
                                indxadd2 = 1
                        else:
                            if not (self.TypesOfLines[connection[0]][connection[1]][0] == 0):
                                indxadd1 = 0
                            else:
                                indxadd1 = 1
                            if not (self.TypesOfLines[connection[2]][connection[3]][0] == 0 and flag):
                                indxadd2 = 0
                            else:
                                indxadd2 = 1

                        flag = True
                        if connection[0] == connection[2]:
                            if connection[1] > connection[3]:
                                self.TypesOfLines[connection[0]][connection[1]][indxadd1] = 2
                                self.TypesOfLines[connection[2]][connection[3]][indxadd2] = 3
                            else:
                                self.TypesOfLines[connection[0]][connection[1]][indxadd1] = 3
                                self.TypesOfLines[connection[2]][connection[3]][indxadd2] = 2

                        else:
                            if connection[0] > connection[2]:
                                self.TypesOfLines[connection[0]][connection[1]][indxadd1] = 1
                                self.TypesOfLines[connection[2]][connection[3]][indxadd2] = 4
                            else:
                                self.TypesOfLines[connection[0]][connection[1]][indxadd1] = 4
                                self.TypesOfLines[connection[2]][connection[3]][indxadd2] = 1

                subCellsAssignment = np.zeros((2 * self.darp_instance.rows, 2 * self.darp_instance.cols))
                for i in range(self.darp_instance.rows):
                    for j in range(self.darp_instance.cols):
                        subCellsAssignment[2 * i][2 * j] = self.darp_instance.A[i][j]
                        subCellsAssignment[2 * i + 1][2 * j] = self.darp_instance.A[i][j]
                        subCellsAssignment[2 * i][2 * j + 1] = self.darp_instance.A[i][j]
                        subCellsAssignment[2 * i + 1][2 * j + 1] = self.darp_instance.A[i][j]

                drone_turns = turns(AllRealPaths)
                drone_turns.count_turns()
                drone_turns.find_avg_and_std()
                self.mode_to_drone_turns.append(drone_turns)

                AllRealPaths_dict[mode] = AllRealPaths
                subCellsAssignment_dict[mode] = subCellsAssignment

            # Find mode with the smaller number of turns
            averge_turns = [x.avg for x in self.mode_to_drone_turns]
            self.min_mode = averge_turns.index(min(averge_turns))

            # Retrieve number of cells per robot for the configuration with the smaller number of turns
            min_mode_num_paths = [len(x) for x in AllRealPaths_dict[self.min_mode]]
            min_mode_returnPaths = AllRealPaths_dict[self.min_mode]

            # Uncomment if you want to visualize all available modes

            # if self.darp_instance.visualization:
            #     for mode in range(4):
            #         image = visualize_paths(AllRealPaths_dict[mode], subCellsAssignment_dict[mode],
            #                                 self.darp_instance.droneNo, self.darp_instance.color)
            #         image.visualize_paths(mode)
            #     print("Best Mode:", self.min_mode)

            # Combine all modes to get one mode with the least available turns for each drone
            combined_modes_paths = []
            combined_modes_turns = []

            for r in range(len(self.darp_instance.init_robot_pos)):
                min_turns = sys.maxsize
                temp_path = []
                for mode in range(4):
                    if self.mode_to_drone_turns[mode].turns[r] < min_turns:
                        temp_path = self.mode_to_drone_turns[mode].paths[r]
                        min_turns = self.mode_to_drone_turns[mode].turns[r]
                combined_modes_paths.append(temp_path)
                combined_modes_turns.append(min_turns)

            self.best_case = turns(combined_modes_paths)
            self.best_case.turns = combined_modes_turns
            self.best_case.find_avg_and_std()

            # Retrieve number of cells per robot for the best case configuration
            best_case_num_paths = [len(x) for x in self.best_case.paths]
            best_case_returnPaths = self.best_case.paths

            # visualize best case
            if self.darp_instance.visualization:
                image = visualize_paths(self.best_case.paths, subCellsAssignment_dict[self.min_mode],
                                        len(self.darp_instance.init_robot_pos), self.darp_instance.color)
                image.visualize_paths("Combined Modes")

            self.execution_time = time.time() - start_time

            if len(best_case_returnPaths) > 0:
                print(f'\nResults for "{export_file_name}" tiles group:')
                print(f'Number of cells per robot: {best_case_num_paths}')
                print(f'Minimum number of cells in robots paths: {min(best_case_num_paths)}')
                print(f'Maximum number of cells in robots paths: {max(best_case_num_paths)}')
                print(f'Average number of cells in robots paths: {np.mean(np.array(best_case_num_paths))}')
                print(f'\nTurns Analysis for: {self.best_case}')
            else:
                print(f'ATTENTION:\nSomething went wrong in path construction or finding '
                      f'the best path for "{export_file_name}" tiles group!')
                print("self.best_case.paths doesn't hold paths tuples!")

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
        for r in range(droneNo):
            k = Kruskal(rows, cols)
            k.initializeGraph(BinaryRobotRegions[r, :, :], True, mode)
            k.performKruskal()
            MSTs.append(k.mst)
        return MSTs

    def to_image(self):
        file_path = Path('result_export', self.export_file_name + ".jpg")
        if not file_path.parent.exists():
            os.makedirs(file_path.parent)
        plt.imsave(file_path, self.darp_instance.A, dpi=100)
        print("Exported image of final assignment matrix")

    def to_video(self):
        # existing gif in results_export folder?
        num_of_processes = os.cpu_count() - 1
        clip = mp.VideoFileClip("./result_export/" + self.export_file_name + ".gif")
        clip.write_videofile("./result_export/" + self.export_file_name + ".mp4", audio=False, threads=num_of_processes)
        clip.close()
        print("Created MP4 video from assignment matrix animation GIF file")
