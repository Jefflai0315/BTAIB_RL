from shapely.geometry import Polygon
from pyproj import Proj, transform, Transformer
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def polygon_to_Poly3DCollection(polygon, height, config={"facecolors":"cyan", "linewidths":1, "edgecolors":"r", "alpha":0.5}): 
    """
    Returns a Poly3DCollection object for a given polygon and height
    input: 
        polygon: shapely.geometry.Polygon
        height: int 
        config: dict (Optional) 
    output:
        A 3D Poly3DCollection object representing the input polygon extruded to the specified height: mpl_toolkits.mplot3d.art3d.Poly3DCollection
    """
    polygon_coords_2d = polygon.exterior.xy
    polygon_coords_2d = list(zip(polygon_coords_2d[0], polygon_coords_2d[1]))            
    polygon_coords_3d = [(x, y, 0) for x, y in polygon_coords_2d]  # Bottom face
    polygon_coords_3d_top = [(x, y, height) for x, y in polygon_coords_2d]  # Top face
    verts = [polygon_coords_3d, polygon_coords_3d_top]
    poly3d = [[verts[0][i], verts[0][i + 1], verts[1][i + 1], verts[1][i]] for i in range(len(verts[0]) - 1)]
    return Poly3DCollection(poly3d, **config)