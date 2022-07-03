import os
import sys
from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString


def search_newest_file_in_folder(path_to_folder, unique_search_string):
    search_pattern = [unique_search_string]
    files_in_folder = os.listdir(Path(path_to_folder).resolve())
    files = [Path(path_to_folder, nm).resolve() for ps in search_pattern for nm in files_in_folder if ps in nm]
    last_file_with_suffix = max(files, key=os.path.getctime)
    return Path(last_file_with_suffix).name.removesuffix('.geojson')


if __name__ == '__main__':
    last_file_no_suffix = search_newest_file_in_folder(Path('../geodataframes'), 'path_per_tilegroup')
    path_per_tilegroup_gdf = gpd.read_file(filename=f'./geodataframes/{last_file_no_suffix}.geojson')

    for i, series in path_per_tilegroup_gdf.iterrows():

        coords_lines = []
        if isinstance(series.geometry, LineString):
            coords_lines.extend(list(series.geometry.coords))
            with open(f'./temp/{series.hash}.txt', 'w') as f:
                for line in coords_lines:
                    f.write(str(line))
                    f.write('\n')

        if isinstance(series.geometry, MultiLineString):
            for line in list(series.geometry.geoms):
                coords_lines.extend(list(line.coords))
            with open(f'./temp/{series.hash}.txt', 'w') as f:
                for line in coords_lines:
                    f.write(str(line))
                    f.write('\n')

    sys.exit(0)
