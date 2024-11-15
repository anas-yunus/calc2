
import geopandas as gpd

geo_data = gpd.read_file('district_map.geojson')
print(geo_data.columns)