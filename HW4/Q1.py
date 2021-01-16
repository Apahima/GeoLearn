import numpy as np

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

def render_pointcloud(v, func, colormap='hot', screenshot='defualt'):
    """
    #Rendering only the vertices of mesh as sphees and asign color to rach sphere
    :param func: Color map weighted function
    :param colormap: Color pallet
    :return: Ploting the graph
    """
    pointcloud = pv.PolyData(v)
    pointcloud['color'] = func
    pointcloud.plot(render_points_as_spheres=True, cmap=colormap, screenshot=screenshot)