import numpy as np
import pyvista as pv
from scipy import sparse
from scipy.sparse import bsr_matrix

def read_off(files):
    """

    :param files: OFF raw file
    :return: verts and faces as np.array
    """
    with open(files) as file:
        if 'OFF' != file.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        # faces = [(int(x), int(y), int(z)) for x, y, z in faces] #Convert faces 'xxx' str to int

        # faces = []
        # for i_face in range(n_faces):
        #     for s in file.readline().strip().split(' '):
        #         faces.append(s)
        verts = np.array([np.array(xi) for xi in verts])
        faces = np.array([np.array(xi) for xi in faces])

        return verts, faces


def write_off(filename, v, f):
    """

    :param filename: Name of files want to be saved
    :param v: the vertices to be saved
    :param f: the faces ti be saved
    :return: OFF file per standard
    """
    offwfile = open(filename, 'w')
    offwfile.write('OFF\n')
    offwfile.write('{} {} 0\n'.format(len(v), len(f)))
    n_vertices, _ = v.shape
    vertices = '\n'.join(['{} {} {}'.format(v[i][0], v[i][1], v[i][2]) for i in range(n_vertices)])
    offwfile.writelines(vertices)
    n_faces, n_vertices_per_face = f.shape
    faces_prefix = str(n_vertices_per_face) + ' '
    faces = '\n'.join([faces_prefix + ' '.join(str(s) for s in f[i]) for i in range(n_faces)])
    offwfile.writelines(faces)

# #An option to conver Numpy array to Pyvista format
# def numpy_to_pyvista(v, f=None):
#     if f is None:
#         return pv.PolyData(v)
#     else:
#         return pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))

class Mesh():
    def __init__(self, v,f):

        self.v = v
        self.f = f
        self.n_vertices, _ = v.shape
        self.n_faces, _ = f.shape
        self.vf_adj = None
        self.vv_adj = None
        self.v_degree = None

        # Option to represent OFF file
        # point_cloud = numpy_to_pyvista(self.v)
        # point_cloud
        # point_cloud.plot(eye_dome_lighting=True)


    def vertex_face_adjacency(self):
        """

        :return: Return the sparse Boolean vertex-face adjacency
        """
        Ver = []
        Fac = []
        for face in range(self.n_faces):
            Ver += list(self.f[face])
            Fac += [face for i in range(len(self.f[face]))]

        A =([True for i in range(len(Ver))], (Ver, Fac))
        self.vf_adj = sparse.coo_matrix(A, shape=(self.n_vertices, self.n_faces), dtype=bool)
        return self.vf_adj


    def vertex_vertex_adjacency(self):
        """

        :return: Return the sparse Boolean vertex-vertex
        """
        self.vv_adj = (self.vf_adj * self.vf_adj.T).multiply(sparse.coo_matrix(1 - np.identity(self.n_vertices), dtype=bool))
        return self.vv_adj

    def vertex_degree(self):
        """
        # Get Vertex - Vertex matrix and return the angle of each vertex
        :return:
        """

        self.v_degree = np.array(np.concatenate(self.vv_adj.sum(axis=1), axis=1))[0]
        return self.v_degree

    def render_wireframe(self):
        """
        #Render the mesh in wireframe view
        :return:
        """

        Smothsurface = pv.PolyData(self.v, np.concatenate((np.full((self.f.shape[0], 1), 3), self.f), 1))
        Smothsurface.plot()

    def render_pointcloud(self, func, colormap=None):
        """
        #Rendering only the vertices of mesh as sphees and asign color to rach sphere
        :param func: Color map weighted function
        :param colormap: Color pallet
        :return: Ploting the graph
        """

        pointcloud = pv.PolyData(self.v)
        pointcloud['color'] = func
        pointcloud.plot(render_points_as_spheres=True, cmap=colormap)

    def render_surface(self, scalar_func=None, colormap=None):
        """

        :param scalar_func: Scalar funcition
        :param colormap: color map pallet
        :return: Plotting loaded OFF file with define scalar function
        """
        surface = pv.PolyData(self.v, np.concatenate((np.full((self.f.shape[0], 1), 3), self.f), 1))
        surface['color'] = scalar_func
        surface.plot(cmap=colormap)

    def face_normal(self,normalized=True, plotting=True, scalar_func=None, colormap=None):
        """
        # Calculating face normals
        :param normalized: Normaliztion
        :param plotting: Plot - True \ False
        :param scalar_func: Weighted scalar function
        :param colormap: Color map pallet
        :return: Return the face normals
        """
        vectors = self.v[self.f]
        Face_Normals = np.cross(vectors[:,1,:]-vectors[:,0,:], vectors[:,2,:]-vectors[:,0,:], axis =1)

        if normalized == True:
            Norm = np.expand_dims(np.linalg.norm(Face_Normals, axis=1), axis=1)
            Face_Normals /= Norm

        if plotting == True:
            # plot the normal function
            surface = pv.PolyData(self.v, np.concatenate((np.full((self.f.shape[0], 1), 3), self.f), 1))
            if scalar_func is not None:
                surface['color'] = scalar_func

            # Plot
            plotter = pv.Plotter()
            plotter.add_mesh(surface, cmap=colormap)
            plotter.add_arrows(cent=self.face_barycenters(), color='yellow',direction=Face_Normals, mag=0.3)
            plotter.show_grid()
            plotter.show()

        return Face_Normals

    def face_barycenters(self):
        vectors = self.v[self.f]
        face_barycenters = np.mean(vectors, axis=1)

        return face_barycenters

    def Face_Area(self):
        Normal = self.face_normal(normalized=False, plotting=False)
        Face_Area = np.linalg.norm(Normal, axis=1) / 2
        return Face_Area

    def barycentric_vertex_areas(self):
        """
        Computing the Barycentric Vertex Areas
        :return: Faces area
        """

        Face_Area = np.expand_dims(self.Face_Area(), axis=1)
        v_areas = (1/3) * np.matmul(self.vf_adj.toarray(), Face_Area)
        return v_areas

    def vertex_normals(self, normalized=True, plotting=True, scalar_func=None, colormap=None):
        """
         #Compute the vertex normals
        :param normalized:
        :param plotting:
        :param scalar_func:
        :param colormap:
        :return: Vertex normals
        """


        Face_Area = np.expand_dims(self.Face_Area(), axis=1)
        Face_Normals = self.face_normal(normalized=False, plotting=False)
        Vertex_Normal = np.matmul(self.vf_adj.toarray(), np.multiply(Face_Area, Face_Normals))

        if normalized == True:
            norms = np.expand_dims(np.linalg.norm(Vertex_Normal, axis=1), axis=1)
            Vertex_Normal /= norms

        if plotting == True:
            # Create the surface of the mesh
            surface = pv.PolyData(self.v, np.concatenate((np.full((self.f.shape[0], 1), 3), self.f), 1))
            if scalar_func is not None:
                surface['color'] = scalar_func

            # Plot the surface and the normals
            plotter = pv.Plotter()
            plotter.add_mesh(surface, cmap=colormap)
            plotter.add_arrows(cent=self.v, direction=Vertex_Normal, mag=0.1, color='yellow')
            plotter.show_grid()
            plotter.show()

        return Vertex_Normal

    def gaussian_curvature(self):
        """
        Calculating the gaussian curvature
        :return: Kappa parameter for surface
        """
        vectors = self.v[self.f]

        norm_0_1 = np.linalg.norm(vectors[:, 1, :] - vectors[:, 0, :], axis=1)
        norm_1_2 = np.linalg.norm(vectors[:, 2, :] - vectors[:, 1, :], axis=1)
        norm_2_0 = np.linalg.norm(vectors[:, 0, :] - vectors[:, 2, :], axis=1)

        vectors0 = np.sum(np.multiply(vectors[:, 1, :] - vectors[:, 0, :], vectors[:, 2, :] - vectors[:, 0, :]), axis=1)
        vectors1 = np.sum(np.multiply(vectors[:, 2, :] - vectors[:, 1, :], vectors[:, 0, :] - vectors[:, 1, :]), axis=1)
        vectors2 = np.sum(np.multiply(vectors[:, 0, :] - vectors[:, 2, :], vectors[:, 1, :] - vectors[:, 2, :]), axis=1)

        vectors0 /= (np.multiply(norm_0_1, norm_2_0))
        vectors1 /= (np.multiply(norm_0_1, norm_1_2))
        vectors2 /= (np.multiply(norm_1_2, norm_2_0))

        vectors0 = np.arccos(vectors0)
        vectors1 = np.arccos(vectors1)
        vectors2 = np.arccos(vectors2)

        k = 2 * np.pi * np.ones((self.n_vertices, 1))
        for i in range(self.n_vertices):
            p0 = np.where(self.f[:, 0] == i)
            p1 = np.where(self.f[:, 1] == i)
            p2 = np.where(self.f[:, 2] == i)
            k[i] -= np.sum(np.concatenate((vectors0[p0], vectors1[p1], vectors2[p2]), axis=0))

        k /= self.barycentric_vertex_areas()
        return k

    def vertex_centroid(self, colormap=None):
        """
        Computing Vertex centroid per method we ask for
        :param colormap: Color pallete
        :return: Vertex centroid across the mesh
        """
        v_centroid = np.mean(self.v, axis=0)

        # Compute the distance of each vertex from the centroid
        dist_centeroid = np.linalg.norm(self.v - np.expand_dims(v_centroid, axis=0), axis=1)
        pointcloud = pv.PolyData(self.v)
        pointcloud['color'] = dist_centeroid

        surface = pv.PolyData(self.v, np.concatenate((np.full((self.f.shape[0], 1), 3), self.f), 1))

        # Plot the surface and the normals
        plotter = pv.Plotter()
        plotter.add_mesh(pointcloud, render_points_as_spheres=True, cmap=colormap)
        plotter.add_mesh(surface, color='yellow', opacity=0.9)
        plotter.add_points(points=v_centroid)
        plotter.show_grid()
        plotter.show()

        return v_centroid

# if __name__ == '__main__':
#     off_path = r'example_off_files\phands.off'
#     verts, faces = read_off(off_path)
