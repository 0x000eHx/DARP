import geopandas as gpd
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import shapely
import numpy as np
from fiona.crs import from_epsg

if __name__ == '__main__':

    talsperre_geojsonfile = Path("Talsperren_einzeln_geojson_files", "Talsperre Malter.geojson")
    gdf_talsperre = gpd.read_file(talsperre_geojsonfile)

    # plot talsperre
    fig, ax = plt.subplots(figsize=(16, 16))

    gdf_talsperre.plot(ax=ax, markersize=.1, figsize=(16, 16), cmap='jet')

    xmin, ymin, xmax, ymax = gdf_talsperre.total_bounds

    # real cell dimensions
    cell_width = 0.0001  # 0.0817750
    cell_height = 0.0001  # 0.0321233

    grid_cells = []
    talsperre_bounds = gdf_talsperre.geometry

    talsperre_gdf_exploded = gdf_talsperre.geometry.explode().tolist()
    max_area_talsperre = max(talsperre_gdf_exploded, key=lambda a: a.area)

    for x0 in tqdm(np.arange(xmin, xmax + cell_width, cell_width)):
        for y0 in np.arange(ymin, ymax + cell_height, cell_height):
            x1 = x0 - cell_width
            y1 = y0 + cell_height
            new_cell = shapely.geometry.box(x0, y0, x1, y1)
            # if new_cell.intersects(max_area_talsperre):
            if new_cell.within(max_area_talsperre):
                grid_cells.append(new_cell)
            else:
                pass

    cell_df = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=from_epsg(4326))
    cell_df.plot(ax=ax, facecolor="none", edgecolor='grey')

    plt.show()

    pass