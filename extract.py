import geopandas as gpd
# import geoplot
from area import area
from shapely.geometry import Point
import pyproj
from shapely.geometry import shape
from shapely.ops import transform
from geographiclib.geodesic import Geodesic
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path


if __name__ == '__main__':

    saxony_geojson_filepath = Path("Overall_geojson_files", "Saxony.geojson")
    df_boundary = gpd.read_file(saxony_geojson_filepath)

    water_saxony_geojson_filepath = Path("Overall_geojson_files", "Water_filtered.geojson")
    df = gpd.read_file(water_saxony_geojson_filepath)
    df = df[~df['name'].isnull()]
    df['name'] = df['name'].astype('str')

    df["dam"] = df.name.apply(lambda x: "Talsperre" in x)
    dams = df[df.dam]  # == True
    dams = dams.reset_index()

    print("Datagram contains {} dams.".format(dams.shape[0]))

    # dams.to_file("TalSperren.geojson", encoding='utf-8', driver='GeoJSON')  # export dams only

    # export dams, one file per dam
    # for idx, row in dams.iterrows():
    #    json_filename = row['name'] + ".geojson"
    #    json_filepath = Path("Talsperren_einzeln_geojson_files", json_filename)
    #    bla = dams.iloc[[idx]]
    #    bla.to_file(json_filepath, encoding='utf-8', driver='GeoJSON')

    print(dams[['landuse', 'natural', 'water', 'name', 'geometry']].head(5))

    print(dams.crs())

    # fig, ax = plt.subplots(figsize=(20, 20))
    # df_boundary[df_boundary.wikipedia == "de:Sachsen"].plot(color='white', edgecolor='green', ax=ax)
    # dams.plot(ax=ax)

    pass
