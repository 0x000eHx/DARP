import geopandas as gpd
# from shapely.geometry import Point
# import pyproj
# from shapely.geometry import shape
# from shapely.ops import transform
# import matplotlib
# import matplotlib.pyplot as plt
# from functools import partial
from pathlib import Path


if __name__ == '__main__':

    saxony_geojson_filepath = Path("../overall_geojsons", "saxony.geojson")
    df_boundary = gpd.read_file(saxony_geojson_filepath)

    water_saxony_geojson_filepath = Path("../overall_geojsons", "water_only.geojson")
    df = gpd.read_file(water_saxony_geojson_filepath)
    df = df[~df['name'].isnull()]
    df['name'] = df['name'].astype('str')

    df["dam"] = df.name.apply(lambda x: "Talsperre" in x)
    dams = df[df.dam]  # == True
    dams = dams.reset_index()

    print("Datagram contains {} dams.".format(dams.shape[0]))

    # export filtered dams only
    # dams.to_file("dams.geojson", encoding='utf-8', driver='GeoJSON')

    # export every single dam into its own file
    # for idx, row in dams.iterrows():
    #    json_filename = row['name'] + ".geojson"
    #    json_filepath = Path("dams_single_geojsons", json_filename)
    #    bla = dams.iloc[[idx]]
    #    bla.to_file(json_filepath, encoding='utf-8', driver='GeoJSON')

    print(dams[['landuse', 'natural', 'water', 'name', 'geometry']].head(5))

    print(dams.crs)

    # plot every damn dam inside county boundary
    # fig, ax = plt.subplots(figsize=(20, 20))
    # df_boundary[df_boundary.wikipedia == "de:Sachsen"].plot(color='white', edgecolor='green', ax=ax)
    # dams.plot(ax=ax)

    pass
