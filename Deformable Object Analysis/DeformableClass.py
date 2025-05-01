import open3d as o3d
import numpy as np
import copy
import point_cloud_utils as pcu
import pyvista as pv
import cmocean
from matplotlib.colors import ListedColormap
import cmasher as cmr
import colorcet as cc
import matplotlib as mpl

from scipy.spatial import KDTree
import matplotlib.colors as mcolors
#taken from: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
#taken from: http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html?highlight=rough%20registration


class DeformableClass:

    def __init__(self,source_fname = "", target_fname = "", draw_intermediate = False, draw_final = True, grasper = "soft", control_type = "open"):
        self.draw_intermediate = draw_intermediate #draw intermediate steps before final registration result
        self.draw_final = draw_final #draw final registration result

        self.source_fname = source_fname #name of source
        self.target_fname = target_fname #name of target
        self.transformation_ransac = [] #ransac transformation
        self.transformation_icp = [] #ICP fine registratin
        self.source_mesh = None
        self.target_mesh = None
        self.source_mesh_downsampled = None
        self.target_mesh_downsampled = None
        self.grasper = grasper #either "soft" or "rigid"
        self.control_type = control_type #either "open" or "closed"
        self.n = 100000 #number of points to sample
        self.voxel_size = 0.5 # 0.5 means 0.5 mm for this dataset  # means 5cm for this dataset -> 1 mm?

        self.template_fname = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"

        # For visualizations
        if self.grasper.lower() == "soft":

            if self.control_type.lower() == "open":
                self.ang0 = np.radians(25)
                self.ang_offset = [np.radians(x) + self.ang0 for x in
                                   [0, 127, 228]]  # each plane is 120 degrees from each other
                self.rot_angle = [0, 150, 210]  # how to rotate each slices
                self.center_pos = [[21.927, 33.805, 9.4167], [-22.0046, 37.137, 11.706], [-7, 36.423,
                                                                                          -24.1]]  # coordinates of the center of the contact region, found via hand picking points
                self.zoom = 3.2
            else:
                self.ang0 = np.radians(25)
                self.ang_offset = [np.radians(x) + self.ang0 for x in
                                   [0, 134, 234]]  # each plane is 120 degrees from each other
                self.rot_angle = [0, 150, 210]  # how to rotate each slices
                self.center_pos = [[23.217, 27.86, 12.14], [-23.5, 31.85, 7.91], [-4.84, 31.15,
                                                                                  -26.19]]  # coordinates of the center of the contact region, found via hand picking points
                self.zoom = 3.0


            self.group = [(np.s_[:], 0), (0, 1), (1, 1), (2, 1)]
            self.shape = [3,2]
            self.col_weights = [2 / 3, 1 / 3]
            self.row_weights = [1 / 3, 1 / 3, 1 / 3]


        elif self.grasper.lower()=="rigid":
            self.ang0 = np.radians(-2.5)
            self.ang_offset = [np.radians(x) + self.ang0 for x in [0, 173]]
            self.rot_angle = [0, 180]  # how to rotate each slice


            if self.control_type.lower() == "open":
                self.center_pos = [[20.1552887, 37.84481049, -0.76447487], [-19.65922928, 37.84709167,
                                                                            3.53266525]]  # TODO: Update this.  coordinates of the center of the contact region, found via hand picking points

                self.zoom = 2.3
            else:
                self.center_pos = [[24.33,31.77,0.575], [-22.813, 31.77, 3.394]]  # coordinates of the center of the contact region, found via hand picking points
                self.zoom = 2.2

            self.group = [(np.s_[:], 0), (0, 1), (1, 1)]
            self.shape = [2, 2]
            self.col_weights = [2/3, 1/3]
            self.row_weights = [1/2, 1/2]

        elif self.grasper.lower() == "soft w/ rigid jaws":
            self.ang0 = np.radians(25)
            self.ang_offset = [np.radians(x) + self.ang0 for x in
                               [0, 127, 228]]  # each plane is 120 degrees from each other
            self.rot_angle = [0, 150, 210]  # how to rotate each slices
            self.center_pos = [[22.504, 33.328, 9.34], [-20.082, 36.47, 6.00], [-8.20, 37.2,
                                                                                      -21.82]]  # coordinates of the center of the contact region, found via hand picking points

            self.group = [(np.s_[:], 0), (0, 1), (1, 1), (2, 1)]
            self.shape = [3, 2]
            self.col_weights = [2 / 3, 1 / 3]
            self.row_weights = [1 / 3, 1 / 3, 1 / 3]
            self.zoom = 2.9

        #For distance calcs
        self.avg_length_mm = 5 #average cube length around center to average over

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=30, origin=[0, 0, 0])

        #o3d.visualization.draw_geometries([source_temp, target_temp]) #original viewer
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(source_temp)
        viewer.add_geometry(target_temp)
        viewer.add_geometry(mesh_frame)
        opt = viewer.get_render_option()

        #opt.show_coordinate_frame = True
        #opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()



    def preprocess_point_cloud(self, pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        if self.draw_intermediate == True:
            o3d.visualization.draw_geometries([pcd_down])

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh


    def prepare_dataset(self, source,target,voxel_size):
        print(":: Load two point clouds and disturb initial pose.")

        #demo_icp_pcds = o3d.data.DemoICPPointClouds()
        #source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
        #target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
        # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        # source.transform(trans_init)
        if self.draw_intermediate == True:
            self.draw_registration_result(source, target, np.identity(4))

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999))
        print("Transformation: ")
        print(result.transformation)
        return result


    def refine_registration(self, source, target, source_fpfh, target_fpfh, result_ransac, voxel_size):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return result

    def execute_fast_global_registration(self, source_down, target_down, source_fpfh,
                                         target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        print(result.transformation)
        return result

    def center_rotate_meshes(self,source_fname=None, source_mesh=None, target_fname=None,
                  target_mesh=None, n=None, center_meshes=False, center_source=False, center_target=False,
                  rotate_source=None, rotate_target=None, transform_source = None, transform_target = None):

            self.source_fname = source_fname if source_fname is not None else self.source_fname
            self.target_fname = target_fname if target_fname is not None else self.target_fname
            self.n = n if n is not None else self.n

            ## Source Mesh
            if source_mesh is None:
                mesh = o3d.io.read_triangle_mesh(self.source_fname)
                mesh.compute_vertex_normals()
            else:
                mesh = source_mesh

            if center_meshes == True or center_source == True:
                mesh.translate([0, 0, 0], relative=False)

            if rotate_source is not None:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotate_source)
                mesh.rotate(R)

            if transform_source is not None:
                mesh.transform(transform_source)


            if self.draw_intermediate == True:
                o3d.visualization.draw_geometries([mesh])



            source_original = mesh
            source = mesh.sample_points_uniformly(number_of_points=self.n)

            ## Target Mesh
            if target_mesh is None:
                mesh2 = o3d.io.read_triangle_mesh(self.target_fname)
                mesh2.compute_vertex_normals()
            else:
                mesh2 = target_mesh

            if center_meshes == True or center_target == True:
                mesh2.translate([0, 0, 0], relative=False)

            if rotate_target is not None:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotate_target)
                mesh2.rotate(R)

            if transform_target is not None:
                mesh2.transform(transform_target)

            if self.draw_intermediate == True:
                o3d.visualization.draw_geometries([mesh2])

            target_original = mesh2
            target = mesh2.sample_points_uniformly(number_of_points=self.n)

            return source_original, source, target_original, target
    def alignMesh_method(self,source_fname  = None, source_mesh = None, target_fname = None, target_mesh = None,
                         n = None, center_meshes = False, center_source = False, center_target = False,
                         rotate_source = None, rotate_target = None, voxel_size = None):

        source_original, source, target_original, target = self.center_rotate_meshes(source_fname, source_mesh, target_fname, target_mesh, n, center_meshes, center_source, center_target, rotate_source, rotate_target)

        # #global registration
        voxel_size = self.voxel_size if voxel_size is None else voxel_size
        source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(source, target,
                                                                                             voxel_size)

        result_ransac = self.execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        print(result_ransac)
        result_ransac

        if self.draw_intermediate == True:
            self.draw_registration_result(source_down, target_down, result_ransac.transformation)

        # Fast global registration
        # voxel_size = 20  # means 5cm for this dataset -> 1 mm?
        # source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,
        #     voxel_size)
        # result_fast = execute_fast_global_registration(source_down, target_down,
        #                                                source_fpfh, target_fpfh,
        #                                                voxel_size)
        #
        # draw_registration_result(source_down, target_down, result_fast.transformation)

        # ICP registration
        result_icp = self.refine_registration(source, target, source_fpfh, target_fpfh, result_ransac,
                                         voxel_size)
        print(result_icp)
        self.transformation_icp = result_icp

        if self.draw_intermediate == True:
            self.draw_registration_result(source, target, result_icp.transformation)

        source.transform(result_icp.transformation)

        if self.draw_final == True:
            self.draw_registration_result(source, target, np.identity(4))
            #o3d.visualization.draw_geometries([source, target])



        source_mesh_downsampled = source
        target_mesh_downsampled = target
        source_mesh = source_original.transform(result_icp.transformation)
        target_mesh = target_original

        return source_mesh_downsampled, target_mesh_downsampled, source_mesh, target_mesh, result_ransac, result_icp


    def alignMesh(self, source_fname  = None, target_fname = None, n = None, center_meshes = False, align_to_template = False,
                  save_transform_name = None):
        if align_to_template == True:

            source_fname = source_fname if source_fname is not None else self.source_fname
            target_fname = target_fname if target_fname is not None else self.target_fname


            target_mesh_downsampled, template_mesh_downsampled, \
            target_mesh, template_mesh, target_ransac, target_icp = self.alignMesh_method(source_fname = target_fname, target_fname = self.template_fname,
                                                               center_meshes = True, rotate_target = [0,np.pi,0], n=n )

            self.source_mesh_downsampled, self.target_mesh_downsampled, \
            self.source_mesh, self.target_mesh, source_ransac, source_icp = self.alignMesh_method(source_fname=source_fname,
                                                               target_mesh=target_mesh,
                                                               center_source = True, center_target = False, n = n)

            self.target_transform = target_icp.transformation
            self.source_transform = source_icp.transformation
            print(self.target_transform)
            print(self.source_transform)



            pass


        else:
            self.source_mesh_downsampled, self.target_mesh_downsampled, \
            self.source_mesh, self.template_mesh, source_ransac, source_icp = self.alignMesh_method(source_fname=source_fname,
                                                                         target_fname=target_fname,
                                                                         center_meshes=center_meshes, n = n)

            self.target_transform = np.eye(4) #homogeneous transformation
            self.source_transform = source_icp.transformation

        if save_transform_name is not None:
            with open(save_transform_name, 'wb') as f:
                np.save(f, self.target_transform)
                np.save(f, self.source_transform)



    def compute_chamfer_distance(self, source = None, target = None):
        source = source if source is not None else self.source_mesh_downsampled
        target = target if target is not None else self.target_mesh_downsampled

        # Compute chamfer distance

        cd = pcu.chamfer_distance(np.array(source.points), np.array(target.points))
        print("Chamfer between original scan and the scan after grasp is %f" % cd)
        return cd

    def compute_chamfer_distance_custom(self,source = None, target = None):
        source = source if source is not None else self.source_mesh_downsampled
        target = target if target is not None else self.target_mesh_downsampled

        # Compute chamfer distance
        d_kdtree, idx = self.get_distances(source = source, target = target)
        n1 = len(source.points)
        t_d1 = sum(d_kdtree)

        d_kdtree2, idx2 = self.get_distances(source = target, target = source) #flip position
        n2 = len(target.points)
        t_d2 = sum(d_kdtree2)

        CD_total = t_d1/n1 + t_d2/n2

        return [CD_total]

    def get_distances(self,source = None, target = None):
        tree = KDTree(target.points)  # distance from source to target
        d_kdtree, idx = tree.query(source.points)
        return d_kdtree,idx

    def split_mesh_get_center(self, source_mesh, length = 5, angle = None, center_point = None, normal = None, clip_origin = None, debug = False):

        cld = source_mesh.clip(normal=tuple(normal), origin=list(clip_origin), invert=False)

        cube = pv.Cube(center=tuple(center_point),
                       x_length=50,
                       y_length=length,
                       z_length=length)
        angle = angle if angle is not None else np.arctan2(normal[2], normal[0]) * 180 / np.pi
        cube = cube.rotate_vector([0, 1, 0], -angle, center_point) #get the cube

        intersection = cld.clip_box(cube, invert=False)
        mean_distance = np.mean(intersection["distance"])


        #Cube for clipping for visualization
        cube_clip = pv.Cube(center=tuple(center_point),
                            x_length=50,
                            y_length=39,
                            z_length=30)

        cube_clip = cube_clip.rotate_vector([0, 1, 0], -angle, center_point)  # get the cube

        intersection2 = cld.clip_box(cube_clip, invert=False)


        if debug == True:
            p = pv.Plotter()

            p.background_color = 'white'
            p.enable_point_picking(callback=self.display_picked_point)
            p.add_mesh(cld, scalars="distance", cmap="CET_CBD2", color=True, opacity=0.5)
            p.add_mesh(cube, opacity=0.15)
            p.add_mesh(intersection, opacity=1)
            p.show()

        return cld,mean_distance, intersection2

    def display_picked_point(self,p):
        print(p)
    def PyVista_show(self, source_mesh = None, target_mesh = None, grasper = None, dmin_mm = 0, dmax_mm = 7.5,plot_style = "project", show_plot = False, debug = False, fname = None, show_axes = True, show_colorbar = True, show_planes = False, camera_pos = None, fname_movie = None):

        source_mesh = copy.deepcopy(source_mesh) if source_mesh is not None else copy.deepcopy(
            self.source_mesh_downsampled)
        target_mesh = copy.deepcopy(target_mesh) if target_mesh is not None else copy.deepcopy(
            self.target_mesh_downsampled)
        grasper = grasper if grasper is not None else self.grasper

        pv.global_theme.font.color = 'black'
        pv.global_theme.font.size = 10
        pv.global_theme.font.label_size = 10
        pv.global_theme.allow_empty_mesh = True

        if plot_style.lower() == "project":
            p = pv.Plotter()
            p.background_color = 'white'

            # p.enable_point_picking(callback = self.display_picked_point)

        else:
            group = self.group

            p = pv.Plotter(shape=self.shape, window_size=(1000, 1000), col_weights=self.col_weights,
                           row_weights=self.row_weights,
                           groups=group, lighting = "three lights")
            p.disable_shadows()
            for k in np.arange(0, self.shape[1]):
                p.subplot(k, 1)
                p.background_color = 'white'

        p.subplot(0, 0)
        p.set_viewup([0, 1, 0])

        [d, i] = self.get_distances(source=source_mesh,
                                    target=target_mesh)  # get distance of source mesh (deformed mesh) from the target (undeformed mesh), referenced to the points in the source mesh.
        source_mesh["distance"] = d
        # source_mesh["distance"][
        #     0] = dmin_mm  # set one of the values to what is hopefully the maximum distance for the colorbar scaling to work properly
        # source_mesh["distance"][
        #     1] = dmax_mm  # set one of the values to what is hopefully the maximum distance for the colorbar scaling to work properly

        ang0 = self.ang0
        ang_offset = self.ang_offset
        avg_length = self.avg_length_mm

        normal = [np.array([np.cos(x), 0, np.sin(x)]) for x in ang_offset]

        normalpos = [np.array([np.cos(x), 20, np.sin(x)]) for x in ang_offset]
        light_pos = [-20 * x for x in normalpos]

        clip_offset_mag = 15  # offset from origin of the clip plane
        projection_mag = 60  # offset from origin of the projection plane
        clip_origin = [clip_offset_mag * x for x in normal]
        proj_origin = [projection_mag * x for x in normal]
        clipped_data = []  # hold the clipped data
        # opacity = [0, 0.2, 1, 1, 1, 1, 1, 1, 1, 1]
        source_mesh["opacity"] = np.ones(len(source_mesh.points))
        source_mesh["opacity"][source_mesh["distance"] <= 0.4] = 1

        my_colormap = cmr.get_sub_cmap('cmr.neon_r', 0, 1)

        project_plane = False
        angle = self.rot_angle
        mean_distance = []

        if plot_style != "project":
            p.subplot(0, 0)  # activate the large view

        p.set_viewup([0, 1, 0])

        p.add_mesh(source_mesh, scalars="distance", cmap=my_colormap, clim=[dmin_mm, dmax_mm], color=True)
        if show_axes == True:
            p.show_axes()

        cam_position_default = [(121.8811292356143, 112.98073524919948, -97.00768870196411),
                             (0.8108825716953554, 17.391831208985828, -1.449314787971784),
                             (-0.47772489624343795, 0.8450701284445499, 0.24007374183760427)]

        p.camera_position = cam_position_default if camera_pos is None else camera_pos

        # cpos = [(223.98191723264253, 232.29015597418473, -87.18915118775462),
        #         (0.8108825716953554, 17.391831208985828, -1.449314787971784),
        #         (-0.6707736342526432, 0.7352928425645483, 0.09699055245150295)]

        for ix, n in enumerate(normal):

            # Add light
            light = pv.Light(position=tuple(light_pos[ix]), color='white', cone_angle=25, exponent=20, intensity=0.25)
            light.positional = True

            # Split mesh
            cld, md, cld_clipped = self.split_mesh_get_center(source_mesh, center_point=self.center_pos[ix],
                                                 length=avg_length, angle=np.rad2deg(ang_offset[ix]),
                                                 normal=normal[ix], clip_origin=clip_origin[ix], debug=debug)

            clipped_data.append(cld)
            mean_distance.append(md)

            # Add a plane for visualization
            x = ang_offset[ix]

            planepos = [np.array([40 * np.cos(x), 25, 40 * np.sin(x)])]
            plane = pv.Plane(center=tuple(planepos), direction=n, i_resolution=5, j_resolution=5, i_size=25, j_size=25)

            if plot_style == "project":
                # Project as Plane or as 3D slice
                if project_plane == True:
                    projected = cld.project_points_to_plane(origin=list(proj_origin[ix]), normal=tuple(n))
                else:
                    projected = cld.translate(list(proj_origin[ix]))
                    if ix != 0:
                        projected = projected.rotate_vector(vector=(0, 1, 0), angle=angle[ix], point=proj_origin[ix])
                        projected = projected.translate(list(proj_origin[ix] * 0.4))


            else:
                p.subplot(ix, 1)  # to activate that subplot
                projected = cld_clipped
                p.view_vector(-1 * normal[ix])
                p.set_viewup([0, 1, 0])
                camera_offset = [200, 1, 200]
                camera_pos = [normal[ix][0], 0, normal[ix][2]]
                p.camera.position = tuple([x * camera_offset[ix] for (ix, x) in enumerate(camera_pos)])
                # p.camera.focal_point = (0.2, 0.3, 0.3)
                p.camera.up = (0.0, 1.0, 0.0)
                p.camera.zoom(self.zoom)
                p.enable_parallel_projection()

                projected = projected.translate([0, -32, 0])

            p.add_mesh(projected, scalars="distance", cmap=my_colormap, color=True, opacity="opacity", clim=[dmin_mm, dmax_mm])

            p.subplot(0, 0) #reactive the large 3d plot
            # p.add_light(light)
            if show_planes == True:
                p.add_mesh(plane, opacity=0.1, show_edges=True, color="white")

        # Plot

        p.subplot(0,0)
        if show_colorbar == False:
            p.remove_scalar_bar() #remove the color bar


        if fname is not None:
            p.save_graphic(fname)

        if show_plot == True:
            cpos = p.show( return_cpos=True)

        if fname_movie is not None:
            filename = fname_movie  # https://docs.pyvista.org/examples/02-plot/movie.html
            ang_rot = np.linspace(0, 360, 240,
                                  dtype=int)  # 6 seconds to do the full rotation, 30 fps, so 60 frames total
            p = pv.Plotter(window_size=(1000, 1000), lighting="three lights")
            p.background_color = 'white'
            p.open_movie(filename, quality=10, framerate=30)

            for rot in ang_rot:
                rot_source = source_mesh.rotate_y(rot, point=(0, 0, 0), inplace=False)
                actor = p.add_mesh(rot_source, scalars="distance", cmap=my_colormap, clim=[dmin_mm, dmax_mm],
                                   color=True)
                rot_target = target_mesh.rotate_y(rot, point=(0, 0, 0), inplace=False)
                actor2 = p.add_mesh(rot_target, opacity = 0.175,
                                   color=True)
                p.camera_position = cam_position_default
                p.remove_scalar_bar()
                p.write_frame()
                p.remove_actor(actor)
                p.remove_actor(actor2)

            p.close()





        return mean_distance






    def py_vista_show_debug(self,grasper = "soft"):

        source_mesh = self.source_mesh_downsampled
        target_mesh = self.target_mesh_downsampled

        pv.global_theme.font.color = 'black'
        pv.global_theme.font.size = 10
        pv.global_theme.font.label_size = 10
        p = pv.Plotter()
        p.background_color = 'white'

        # axes = pv.Axes()
        # axes.axes_actor.shaft_type = axes.axes_actor.ShaftType.CYLINDER
        # axes.axes_actor.total_length = (50, 50, 50)
        # axes.axes_actor.cylinder_radius = 0.01
        # axes.axes_actor.cone_radius = 0.015
        #
        # p.add_actor(axes.axes_actor)

        # Get nearest neighbor distances
        [d, i] = self.get_distances(source=source_mesh, target=target_mesh) #get distance of source mesh (deformed mesh) from the target (undeformed mesh), referenced to the points in the source mesh.
        source_mesh["distance"] = d
        source_mesh["distance"][0] = 7.5

        # setup clipped planes

        if grasper == "soft":
            ang0 = np.radians(30)
            ang_offset = [np.radians(x) + ang0 for x in [0, 120, 240]]  #each plane is 120 degrees from each other
        else:
            ang0 = np.radians(0)
            ang_offset = [np.radians(x) + ang0 for x in [0, 180]]


        normal = [np.array([np.cos(x),0,np.sin(x)]) for x in ang_offset]

        normalpos = [np.array([np.cos(x), 20, np.sin(x)]) for x in ang_offset]
        light_pos = proj_origin = [-20 * x for x in normalpos]


        clip_offset_mag = 15 #offset from origin of the clip plane
        projection_mag = 60 #offset from origin of the projection plane
        clip_origin = [clip_offset_mag*x for x in normal]
        proj_origin = [projection_mag*x for x in normal]
        clipped_data = [] #hold the clipped data

        opacity = [0, 0.2, 1, 1, 1, 1, 1, 1, 1, 1]
        #opacity = [0, 0.4, 0.8, 0.9, 1, 1]

        # Define the colors we want to use
        white = np.array([1, 1, 1, 1.0])
        green1 = np.array([216 / 256, 220 / 256, 230 / 256, 1])
        green2 = np.array([45 / 256, 227 / 256, 166 / 256, 1])
        blue1 = np.array([121 / 256, 196 / 256, 219 / 256, 1])
        blue2 = np.array([90 / 256, 127 / 256, 219 / 256, 1])
        yellow = np.array([255 / 256, 247 / 256, 0 / 256, 1.0])
        red = np.array([1.0, 0.0, 0.0, 1.0])

        mapping = np.linspace(0, 10, 256)
        newcolors = np.empty((256, 4))
        newcolors[mapping >= 6] = red
        newcolors[mapping < 6] = yellow
        newcolors[mapping < 5] = blue2
        newcolors[mapping < 4] = blue1
        newcolors[mapping < 3] = green2
        newcolors[mapping < 2] = green1
        newcolors[mapping < 1] = white

        # Make the colormap from the listed colors
        my_colormap = ListedColormap(newcolors)
        my_colormap ="CET_CBD2"

        project_plane = False
        if grasper == "soft":
            angle = [0, 150, 210]
        else:
            angle = [0, 180]
        for ix, n in enumerate(normal):
            light = pv.Light(position=tuple(light_pos[ix]), color='white', cone_angle=25, exponent=20, intensity=0.25)
            light.positional = True
            p.add_light(light)
            cld = source_mesh.clip(normal=tuple(n), origin=list(clip_origin[ix]), invert=False)
            if project_plane == True:
                projected = cld.project_points_to_plane(origin=list(proj_origin[ix]), normal=tuple(n))
            else:
                projected = cld.translate(list(proj_origin[ix]))
                if ix != 0:
                    projected = projected.rotate_vector(vector=(0, 1, 0), angle=angle[ix], point=proj_origin[ix])
                    projected = projected.translate(list(proj_origin[ix] * 0.4))
            p.add_mesh(projected, scalars="distance", cmap=my_colormap, color=True, opacity=opacity)
            x = ang_offset[ix]
            planepos = [np.array([30 * np.cos(x), 25, 30 * np.sin(x)])]
            plane = pv.Plane(center=tuple(planepos), direction=n, i_resolution=5, j_resolution=5, i_size=25, j_size=25)
            p.add_mesh(plane, opacity=0.25, show_edges=True, color="white")
            clipped_data.append(cld)

        # Plot

        p.add_mesh(source_mesh, scalars="distance", cmap=my_colormap, color=True)

        # projected = self.source_mesh.project_points_to_plane(origin=[40,0,0],normal = [1, 0, 0])
        # p.add_mesh(projected, cmap='rain', color=True)

        # p.add_mesh(target_mesh, opacity=0.45, color=True)
        # p.add_mesh_clip_plane(source_mesh)

        p.show()

        pass





    def convert_Open3d_to_PyVista(self,meshes=[], reconstruct = False):
        converted_mesh = [None for x in meshes]

        for ix,mesh in enumerate(meshes):

            if reconstruct[ix] == False:
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])  # Add a column of 3s to indicate triangle faces
                converted_mesh[ix] = pv.PolyData(vertices, faces)
            else:
                mesh.estimate_normals()
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(mesh)
                mesh.compute_vertex_normals()
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                faces = np.hstack(
                    [np.full((faces.shape[0], 1), 3), faces])  # Add a column of 3s to indicate triangle faces
                converted_mesh[ix] = pv.PolyData(vertices, faces)

        return converted_mesh

    def convert_to_PyVista(self,meshes=None):
        meshes = meshes if meshes is not None else [self.source_mesh, self.target_mesh, self.source_mesh_downsampled, self.target_mesh_downsampled]
        reconstruct = [False, False, True, True] #the downsampled version are just pointclouds, not meshes
        converted_mesh = self.convert_Open3d_to_PyVista(meshes = meshes, reconstruct= reconstruct)
        self.source_mesh = converted_mesh[0]
        self.target_mesh = converted_mesh[1]
        self.source_mesh_downsampled = converted_mesh[2]
        self.target_mesh_downsampled = converted_mesh[3]





if __name__ == "__main__":
    # target = "Objects\\Deformables\\Soft_ClosedLoop_K4p28_January2nd_PreScan_1_redone.obj"
    # source = "Objects\\Deformables\\Soft_ClosedLoop_K4p28_January2nd_PostScan_1_redone.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "soft"
    # control_type = "closed"
    # fname = "test_K4p28.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.

    # target = "Objects\\Deformables\\Object_Parallel_undeformed_1000x.obj"
    # source = "Objects\\Deformables\\ObjectDeformed_ParallelJaws_Rigid_1000x.obj"
    # template = "Objects\\Object_Parallel_undeformed_1000x.obj"

    # target = "Objects\\Deformables\\SoftGrasper_Dec31_OpenLoop_35mm_PreScan_3.obj"
    # source = "Objects\\Deformables\\SoftGrasper_Dec31_OpenLoop_35mm_PostScan_3.obj"
    # grasper = "soft"
    # control_type = "Open"
    # fname = "test.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.

    # target = "Objects\\Deformables\\SoftGrasper_Dec31_OpenLoop_35mm_PreScan_2.obj"
    # source = "Objects\\Deformables\\SoftGrasper_Dec31_OpenLoop_35mm_PostScan_2.obj"
    # grasper = "soft"
    # control_type = "Open"
    # fname = "test_soft_35mm_2.npy"

    target = "Objects\\Deformables\\Jan18th_2025_Rigid_OpenLoop_35mm_Prescan1.obj"
    source = "Objects\\Deformables\\Jan18th_2025_Rigid_OpenLoop_35mm_Postscan1.obj"
    grasper = "rigid"
    control_type = "Open"
    fname = "test_rigid.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.

    template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"

    supply_transform = True  #supply transform == true uses the target and source homogeneous transforms applied above instead of doing RANSAC + ICP.
    save_transformation = False


    if supply_transform == False:
        deformable = DeformableClass(source_fname = source, target_fname = target, draw_intermediate = False, draw_final = False, grasper = grasper)
        if save_transformation == True:
            deformable.alignMesh(align_to_template = True, save_transform_name = fname)
        else:
            deformable.alignMesh(align_to_template=True)
        deformable.compute_chamfer_distance()
    else:
        with open(fname, 'rb') as f:
            t_transform = np.load(f)
            s_transform = np.load(f)
        deformable = DeformableClass(source_fname=source, target_fname=target, draw_intermediate=False,
                                     draw_final=False, grasper = grasper,control_type = control_type)
        source_original, source_ds, target_original, target_ds = deformable.center_rotate_meshes(
                                                                                                 source_fname=source,  target_fname=target, center_meshes=True, transform_source=s_transform,
                                                                                                 transform_target = t_transform)
        #deformable.draw_registration_result(source_original, target_original, transformation=np.eye(4))

        deformable.source_mesh_downsampled = source_ds
        deformable.target_mesh_downsampled = target_ds
        deformable.source_mesh = source_original
        deformable.target_mesh = target_original

        deformable.compute_chamfer_distance(source = source_ds, target = target_ds)

    deformable.convert_to_PyVista()
    deformable.compute_chamfer_distance_custom()
    md = deformable.PyVista_show(plot_style = "Separate",show_plot = True, debug = True,show_colorbar = True,show_axes = True,fname = "colorbar.svg",show_planes = False,fname_movie="test.mp4")

    pass















