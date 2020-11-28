import MeshImport
from matplotlib import cm


if __name__ == '__main__':
    off_path = r'example_off_files\teddy171.off'
    verts, faces = MeshImport.read_off(off_path)
    MyMesh = MeshImport.Mesh(verts, faces)

    #Computing the certex face adjacency
    vf_adj = MyMesh.vertex_face_adjacency()
    #Computing the vertex vertex adjacency
    vv_adj = MyMesh.vertex_vertex_adjacency()
    #Compute degrees
    v_degree = MyMesh.vertex_degree()

    #Define Cmap
    cmap = cm.get_cmap('magma')


    #####-----#####
    # To active each section in question one
    # need to remove the note and running the code
    #####-----#####

    ## plot the mesh as a wireframe
    # MyMesh.render_wireframe()

    ## plot the vertices as colored point cloud
    # MyMesh.render_pointcloud(MyMesh.v[:,0], colormap=cmap)

    ## plot the vertices and the faces with scalar function coloring
    ## We can use arbitrary color function map such as faces values for example
    # MyMesh.render_surface(MyMesh.v[:,0],colormap=cmap) # color vertices and faces using vertices values MyMesh.v[:,0]

    ## Plot the face normal point toward "outside" of the surface
    # MyMesh.face_normal(scalar_func=MyMesh.f[:,0], plotting=True, colormap=cmap)

    ##Computing face area
    # fa = MyMesh.Face_Area()

    ## Plot vertices normals with scalar function
    # MyMesh.vertex_normals(scalar_func=MyMesh.f[:,0], colormap=cmap)

    ## Plot the faces with gauusain curvature scalar color function
    # MyMesh.render_surface(MyMesh.gaussian_curvature(), colormap=cmap)

    ## compute the vertices centroid,
    # vertex_centroid = MyMesh.vertex_centroid()

    print('Finish to simulate or compute Q1')