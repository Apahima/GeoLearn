
import pyvista as pv

class Mesh():
    def __init__(self, v,f):

        self.v = v
        self.f = f
        self.n_vertices, _ = v.shape
        self.n_faces, _ = f.shape
        self.vf_adj = None
        self.vv_adj = None
        self.v_degree = None


    def render_pointcloud(self, func, colormap=None, screenshot='defualt'):
        """
        #Rendering only the vertices of mesh as sphees and asign color to rach sphere
        :param func: Color map weighted function
        :param colormap: Color pallet
        :return: Ploting the graph
        """

        pointcloud = pv.PolyData(self.v)
        pointcloud['color'] = func
        pointcloud.plot(render_points_as_spheres=True, cmap=colormap, screenshot=screenshot)

