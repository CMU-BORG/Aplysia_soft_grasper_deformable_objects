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
import os

from scipy.spatial import KDTree
import matplotlib.colors as mcolors
#taken from: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
#taken from: http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html?highlight=rough%20registration


class DeformableClass:

    def __init__(self,source_fname = "", target_fname = "", template_fname = None, draw_intermediate = False, draw_final = True, grasper = "soft", control_type = "open",  center_pos = None, zoom = None, projection_clip_box = None, projected_translate = None,camera_pos = None,
                 n = None, clip_base_dim = None):
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
        self.n = 100000 if n is None else n #number of points to sample
        self.voxel_size = 0.9 # 0.5 means 0.5 mm for this dataset  # means 5cm for this dataset -> 1 mm?

        self.bound_box_crop_min = np.array([-np.inf,-15,-np.inf])
        self.bound_box_crop_max = np.array([np.inf, 40, np.inf])
        self.CD_crop_min = np.array([-np.inf,14,-np.inf])
        self.CD_crop_max = np.array([np.inf, np.inf, np.inf])
        self.clip_base_dim = [-np.inf, np.inf, 14, np.inf, -np.inf, np.inf] if clip_base_dim is None else clip_base_dim

        self.template_fname = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY" if template_fname is None else template_fname

        # For visualizations

        self.projection_clip_box = {"x_length": 50, "y_length": 39, "z_length": 30}
        self.projected_translate = [0, -32, 0]
        self.camera_pos = camera_pos

        if self.grasper.lower() == "soft":

            if self.control_type.lower() == "open":
                self.ang0 = np.radians(25)
                self.ang_offset = [np.radians(x) + self.ang0 for x in
                                   [0, 127, 228]]  # each plane is 120 degrees from each other
                self.rot_angle = [0, 150, 210]  # how to rotate each slices
                self.center_pos = [[21.927, 33.805, 9.4167], [-22.0046, 37.137, 11.706], [-7, 36.423,
                                                                                          -24.1]]  # coordinates of the center of the contact region, found via hand picking points
                self.projection_clip_box = {"x_length": 100, "y_length": 100, "z_length": 100}
                self.zoom = 3.0
            else:
                self.ang0 = np.radians(25)
                self.ang_offset = [np.radians(x) + self.ang0 for x in
                                   [0, 110, 250]]  # each plane is 120 degrees from each other
                self.rot_angle = [0, 150, 210]  # how to rotate each slices
                # coordinates of the center of the contact region, found via hand picking points

                self.center_pos = [[17.90659523, 34.59533691, 13.31189537],
                                   [-14.53621006, 34.83499146, 16.64286995],
                                   [5.35486031, 32.43847656, -24.24461937]]

                self.zoom = 2.6
                self.projection_clip_box = {"x_length": 100, "y_length": 100, "z_length": 100}



            self.group = [(np.s_[:], 0), (0, 1), (1, 1), (2, 1)]
            self.shape = [3,2]
            self.col_weights = [2 / 3, 1 / 3]
            self.row_weights = [1 / 3, 1 / 3, 1 / 3]


        elif self.grasper.lower()=="rigid":
            self.ang0 = np.radians(-2.5)
            self.ang_offset = [np.radians(x) + self.ang0 for x in [0, 173]]
            self.rot_angle = [0, 180]  # how to rotate each slice


            if self.control_type.lower() == "open":
                self.center_pos = [[23.45102119, 32.8572464, -1.80361366], [-23.40859795, 34.50458145, -4.15695]]
                self.ang_offset = [np.radians(x) + self.ang0 for x in [0, 193]]
                self.projection_clip_box = {"x_length": 100, "y_length": 100, "z_length": 100}
                # TODO: Update this.  coordinates of the center of the contact region, found via hand picking points

                self.zoom = 2.3
            else:
                self.center_pos =  [[12.67793655 , 31.26482391 , -4.30991745], [-14.51636124,   32.10226822  , -4.03077126]]  # coordinates of the center of the contact region, found via hand picking points
                self.zoom = 1.5
                self.projection_clip_box = {"x_length": 120, "y_length": 90, "z_length": 120}
                self.projected_translate = [0, -32, 3.5]



            self.group = [(np.s_[:], 0), (0, 1), (1, 1)]
            self.shape = [2, 2]
            self.col_weights = [2/3, 1/3]
            self.row_weights = [1/2, 1/2]

        elif self.grasper.lower() == "soft w/ rigid jaws":
            self.ang0 = np.radians(25)
            self.ang_offset = [np.radians(x) + self.ang0 for x in
                               [0, 140, 228]]  # each plane is 120 degrees from each other
            self.rot_angle = [0, 150, 210]  # how to rotate each slices
            self.center_pos = [[22.504, 33.328, 9.34], [-20.082, 36.47, 6.00], [-8.20, 37.2,
                                                                                      -21.82]]  # coordinates of the center of the contact region, found via hand picking points

            self.group = [(np.s_[:], 0), (0, 1), (1, 1), (2, 1)]
            self.shape = [3, 2]
            self.col_weights = [2 / 3, 1 / 3]
            self.row_weights = [1 / 3, 1 / 3, 1 / 3]
            self.zoom = 3.0
            self.projection_clip_box = {"x_length": 100, "y_length": 100, "z_length": 100}

        #For distance calcs
        self.avg_length_mm = 5 #average cube length around center to average over

        self.center_pos = self.center_pos if center_pos is None else center_pos #override center_pos if center_pos is specified
        self.zoom = self.zoom if zoom is None else zoom #override zoom if this is specified above
        self.projection_clip_box = self.projection_clip_box if projection_clip_box is None else projection_clip_box
        self.projected_translate = self.projected_translate if projected_translate is None else projected_translate
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

        radius_normal = voxel_size * 2 #radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5 #radius feature = voxel_size * 5
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
        distance_threshold = voxel_size * 1.5 #was 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999))
        print("Transformation: ")
        print(result.transformation)
        return result


    def refine_registration(self, source, target, source_fpfh, target_fpfh, result_ransac, voxel_size):
        distance_threshold = voxel_size * 0.4 #was 0.1 for fruits, 0.4 for others
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        if type(result_ransac) == type(np.identity(4)):
            rr = result_ransac #get directly from 4x4 homogeneous matrix
        else:
            rr = result_ransac.transformation #get from registration result

        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, rr,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return result

    def execute_fast_global_registration(self, source_down, target_down, source_fpfh,
                                         target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5 #was 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        print(result.transformation)
        return result

    def resample_template(self, mesh):
        pcd = mesh.sample_points_uniformly(number_of_points=600000)
        #min_bound_dense = np.array([-30, 1.8, -5]) #without translate and rotate
        #max_bound_dense = np.array([-26, 10.2, 5]) #without translate and rotate

        min_bound_dense = np.array([20, -8, -15])
        max_bound_dense = np.array([35, 8, 15])

        bbox_dense = o3d.geometry.AxisAlignedBoundingBox(min_bound_dense, max_bound_dense)

        # Crop the point cloud into dense and sparse regions
        pcd_dense = pcd.crop(bbox_dense)
        pcd_sparse = pcd.crop(bbox_dense, invert=True)

        # Downsample the dense region more densely
        pcd_dense_sampled = pcd_dense.voxel_down_sample(voxel_size=0.01)

        # Downsample the sparse region more sparsely
        pcd_sparse_sampled = pcd_sparse.voxel_down_sample(voxel_size=0.5)

        # Combine the sampled point clouds
        non_uniform_pcd = pcd_dense_sampled + pcd_sparse_sampled

        # Visualize the non-uniformly sampled point cloud
        # o3d.visualization.draw_geometries_with_editing([non_uniform_pcd])
        #o3d.visualization.draw_geometries([non_uniform_pcd, bbox_dense])
        return non_uniform_pcd, mesh #return point cloud and original mesh

    def pick_points(self,pcd):
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    def register_via_correspondences(self,source, target, source_points, target_points, voxel_size = None):
        voxel_size = voxel_size if voxel_size is not None else self.voxel_size
        corr = np.zeros((len(source_points), 2))
        corr[:, 0] = source_points
        corr[:, 1] = target_points
        # estimate rough transformation using correspondences
        print("Compute a rough transform using the correspondences given by user")
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target,
                                                o3d.utility.Vector2iVector(corr))
        # point-to-point ICP for refinement
        print("Perform point-to-plane ICP refinement")
        threshold = voxel_size * 0.1 # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        if self.draw_intermediate == True:
            self.draw_registration_result(source, target, reg_p2p.transformation)
        return reg_p2p

    def manual_align(self, source_fname = None, target_fname = None, source_mesh = None,
                     target_mesh = None, n = None, center_meshes = False, center_source = False, center_target = False,
                     rotate_source = None, rotate_target = None, source_points = None, target_points = None, save_fname = None):

        n = n if n is not None else self.n
        source_mesh, source, target_mesh, target = self.center_rotate_meshes(source_fname, source_mesh,
                                                                                     target_fname, target_mesh, n,
                                                                                     center_meshes, center_source,
                                                                                     center_target, rotate_source,
                                                                                     rotate_target)

        source_points = self.pick_points(source) if source_points is None else source_points
        if save_fname is None:
            target_points = self.pick_points(target) if target_points is None else target_points
        else:
            if not os.path.isfile(save_fname): #if file doesn't exist, save the point cloud, allow user to click target points, then save them
                o3d.io.write_point_cloud(save_fname, target)
                target_points = self.pick_points(target)
                np.save('Objects//template_points.npy', target_points)
            else: #if file does exist
                target = o3d.io.read_point_cloud(save_fname) #read the point cloud
                target_points = np.load('Objects//template_points.npy') #load the target points

        assert (len(source_points) >= 3 and len(target_points) >= 3)
        assert (len(source_points) == len(target_points))
        reg_p2p = self.register_via_correspondences(source, target, source_points, target_points)
        print("")
        source_mesh = source_mesh.transform(reg_p2p.transformation) #transform the mesh
        source = source.transform(reg_p2p.transformation)
        if self.draw_final == True:
            self.draw_registration_result(source_mesh, target_mesh, np.identity(4))
        return source_mesh, target_mesh, source, target, source_points, target_points, reg_p2p



    def center_rotate_meshes(self,source_fname=None, source_mesh=None, target_fname=None,
                  target_mesh=None, n=None, center_meshes=False, center_source=False, center_target=False,
                  rotate_source=None, rotate_target=None, transform_source = None, transform_target = None, source_pcd = None, target_pcd = None,
                             template_nonuniform = None):

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
            source = mesh.sample_points_uniformly(number_of_points=self.n) if source_pcd is None else source_pcd

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

            if template_nonuniform == True:
                # non uniformly sample mesh
                template_pcd, template_mesh = self.resample_template(mesh2)

                target_original = template_mesh
                target = template_pcd
                #source = source_original.sample_points_uniformly(number_of_points=len(template_pcd.points)) #ensures it has the same number of samples


            else:
                target_original = mesh2
                target = mesh2.sample_points_uniformly(number_of_points=self.n) if target_pcd is None else target_pcd



            return source_original, source, target_original, target
    def alignMesh_method(self,source_fname  = None, source_mesh = None, source_pcd = None, target_fname = None, target_mesh = None, target_pcd = None,
                         n = None, center_meshes = False, center_source = False, center_target = False,
                         rotate_source = None, rotate_target = None, voxel_size = None, crop_align_base = False, bound_box_crop = None, template_nonuniform = None,
                         skip_ransac = False):

        source_original, source, target_original, target = self.center_rotate_meshes(source_fname, source_mesh, target_fname, target_mesh, n, center_meshes, center_source, center_target, rotate_source, rotate_target, source_pcd = source_pcd, target_pcd = target_pcd, template_nonuniform = template_nonuniform)

        # #global registration
        voxel_size = self.voxel_size if voxel_size is None else voxel_size
        source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(source, target,
                                                                                             voxel_size)
        if skip_ransac == False:
            result_ransac = self.execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
            result_ransac = result_ransac.transformation
        else:
            result_ransac = np.identity(4) #no transformation

        print(result_ransac)
        result_ransac

        if self.draw_intermediate == True:
            self.draw_registration_result(source_down, target_down)

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

        if crop_align_base == True:
            bound_box_crop_min = bound_box_crop[0] if bound_box_crop is not None else self.bound_box_crop_min
            bound_box_crop_max = bound_box_crop[1] if bound_box_crop is not None else self.bound_box_crop_max
            source_mesh_cropped_downsampled = source_mesh_downsampled.crop(o3d.geometry.AxisAlignedBoundingBox(bound_box_crop_min, bound_box_crop_max))
            source_mesh_cropped = source_mesh.crop(o3d.geometry.AxisAlignedBoundingBox(bound_box_crop_min, bound_box_crop_max))
            return source_mesh_downsampled, target_mesh_downsampled, source_mesh, target_mesh, source_mesh_cropped_downsampled, source_mesh_cropped, result_ransac, result_icp

        return source_mesh_downsampled, target_mesh_downsampled, source_mesh, target_mesh, result_ransac, result_icp


    def alignMesh(self, source_fname  = None, target_fname = None, n = None, center_meshes = False, align_to_template = False,
                  save_transform_name = None, align_method = None):
        if align_to_template == True:

            source_fname = source_fname if source_fname is not None else self.source_fname
            target_fname = target_fname if target_fname is not None else self.target_fname

            match align_method:

                case "Crop Align Base":

                    ncrop = n


                    #------------Target Mesh----------------------------
                    #First, do rough align of target to the template then crop
                    target_mesh_downsampled, template_mesh_downsampled, \
                    target_mesh, template_mesh, target_mesh_cropped_downsampled, target_mesh_cropped,\
                    target_ransac, target_icp = self.alignMesh_method(source_fname=target_fname,
                                                                                                  target_fname=self.template_fname, template_nonuniform = True,
                                                                                                  center_meshes=True,
                                                                                                  rotate_target=[0, np.pi,
                                                                                                                 0], n=ncrop,
                                                                                                  crop_align_base = True) #need to repeat this step because now you want to align with the crop

                    #Now that it has been cropped, take the target mesh that has been cropped and do another align to the template, this time should be better than the above
                    target_mesh_downsampled_2, template_mesh_downsampled, \
                    target_mesh_2, template_mesh, target_ransac2, target_icp2 = self.alignMesh_method(source_mesh=target_mesh_cropped,
                                                                                                  target_mesh= template_mesh, target_pcd = template_mesh_downsampled,
                                                                                                  center_meshes= False, n=n)

                    target_icp.transformation = target_icp2.transformation@target_icp.transformation #multiply the homogenous transforms

                    #--------------Now do the same for the source--------------------
                    source_mesh_downsampled, template_mesh_downsampled, \
                    source_mesh, template_mesh, source_mesh_cropped_downsampled, source_mesh_cropped, \
                    source_ransac, source_icp = self.alignMesh_method(source_fname=source_fname,
                                                                      target_fname=self.template_fname,
                                                                      center_meshes=True,
                                                                      rotate_target=[0, np.pi,
                                                                                     0], n=ncrop,
                                                                      crop_align_base=True)  # need to repeat this step because now you want to align with the crop

                    source_mesh_downsampled_2, template_mesh_downsampled, \
                    source_mesh_2, template_mesh, source_ransac2, source_icp2 = self.alignMesh_method(
                        source_mesh=source_mesh_cropped,
                        target_mesh=template_mesh,
                        center_meshes=False, n= n)

                    # #-----------do final align of source to target -----------#
                    # source_mesh_downsampled_3, target_mesh_downsampled_3, \
                    # source_mesh_3, target_mesh_3, source_ransac3, source_icp3 =   self.alignMesh_method(source_mesh=source_mesh_2,
                    #                                                                                     target_mesh=target_mesh_2,
                    #                                                                                     center_source=True, center_target=False, n=n)
                    #
                    # source_icp.transformation = source_icp3.transformation@(source_icp2.transformation @ source_icp.transformation)  # multiply the homogenous transforms

                    self.source_mesh_downsampled = source_mesh_downsampled.transform(source_icp2.transformation)
                    self.target_mesh_downsampled = target_mesh_downsampled.transform(target_icp2.transformation)
                    self.source_mesh = source_mesh.transform(source_icp2.transformation)
                    self.target_mesh = target_mesh.transform(target_icp2.transformation)

                case "Manual Correspondence To Template then to each other":
                    #----------- Align to Template ------------
                    target_mesh, template_mesh, target, template, target_points, template_points, target_icp = self.manual_align(source_fname = target_fname,
                                                                                             target_fname = self.template_fname,
                                                                                             center_meshes = True,
                                                                                             rotate_target = [0,np.pi,0],
                                                                                             n=n)
                    #----------- Align source to target ------------
                    self.source_mesh, self.target_mesh, self.source_mesh_downsampled, \
                    self.target_mesh_downsampled, source_points, target_points, source_icp = self.manual_align(source_fname=source_fname,
                                                                                 target_mesh=target_mesh,
                                                                                 center_source=True,
                                                                                 center_target = False,
                                                                                 n=n)


                    # self.source_mesh_downsampled, self.target_mesh_downsampled, \
                    # self.source_mesh, self.target_mesh, source_ransac, source_icp2 = self.alignMesh_method(source_mesh=self.source_mesh,
                    #                                                    target_mesh=target_mesh,
                    #                                                    center_source = False, center_target = False, n = n)



                case "Crop base then align to each other":
                    ncrop = n

                    # ------------Target Mesh----------------------------
                    # First, do rough align of target to the template then crop
                    target_mesh_downsampled, template_mesh_downsampled, \
                    target_mesh, template_mesh, target_mesh_cropped_downsampled, target_mesh_cropped, \
                    target_ransac, target_icp = self.alignMesh_method(source_fname=target_fname,
                                                                      target_fname=self.template_fname,
                                                                      template_nonuniform=True,
                                                                      center_meshes=True,
                                                                      rotate_target=[0, np.pi,
                                                                                     0], n=ncrop,
                                                                      crop_align_base=True)  # need to repeat this step because now you want to align with the crop

                    # Now that it has been cropped, take the target mesh that has been cropped and do another align to the template, this time should be better than the above
                    target_mesh_downsampled_2, template_mesh_downsampled, \
                    target_mesh_2, template_mesh, target_ransac2, target_icp2 = self.alignMesh_method(
                        source_mesh=target_mesh_cropped,
                        target_mesh=template_mesh, target_pcd=template_mesh_downsampled,
                        center_meshes=False, n=n)

                    target_icp.transformation = target_icp2.transformation @ target_icp.transformation  # multiply the homogenous transforms

                    self.source_mesh_downsampled, self.target_mesh_downsampled, \
                    self.source_mesh, self.target_mesh, source_ransac, source_icp = self.alignMesh_method(
                        source_fname=source_fname,
                        target_mesh=target_mesh,
                        center_source=True, center_target=False, n=n)

                case "Manual Correspondence to template then auto align to each other":


                    # ----------- Align to Template ------------
                    target_mesh, template_mesh, target, template, target_points, template_points, target_icp = self.manual_align(
                        source_fname=target_fname,
                        target_fname=self.template_fname,
                        center_meshes=True,
                        rotate_target=[0, np.pi, 0],
                        n=n, save_fname = "Objects//template_point_cloud.pcd")

                    # ----------- Auto align to each other ----------
                    self.source_mesh_downsampled, self.target_mesh_downsampled, \
                    self.source_mesh, self.target_mesh, source_ransac, source_icp = self.alignMesh_method(
                        source_fname=source_fname,
                        target_mesh=target_mesh,
                        center_source=True, center_target=False, n=n)

                case "Manual Correspondence of source to template then auto align to each other":

                    # ----------- Align to Template ------------
                    source_mesh, template_mesh, source, template, source_points, template_points, source_icp = self.manual_align(
                        source_fname=source_fname,
                        target_fname=self.template_fname,
                        center_meshes=True,
                        rotate_target=[0, np.pi, 0],
                        n=n, save_fname="Objects//template_point_cloud.pcd")

                    # ----------- Auto align to each other ----------
                    self.target_mesh_downsampled, self.source_mesh_downsampled, \
                    self.target_mesh, self.source_mesh, target_ransac, target_icp = self.alignMesh_method(
                        target_mesh=source_mesh,
                        source_fname=target_fname,
                        center_source=True, center_target=False, n=n)



                case _: #default case

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



    def compute_chamfer_distance(self, source = None, target = None, crop = False, resample = False):
        source = source if source is not None else self.source_mesh_downsampled
        target = target if target is not None else self.target_mesh_downsampled

        if crop == True:
            bound_box_crop_min = self.CD_crop_min
            bound_box_crop_max = self.CD_crop_max
            source = source.crop(
                o3d.geometry.AxisAlignedBoundingBox(bound_box_crop_min, bound_box_crop_max))

            target = target.crop(
                o3d.geometry.AxisAlignedBoundingBox(bound_box_crop_min, bound_box_crop_max))

        if resample == True:
            source = source.sample_points_uniformly(number_of_points = self.n)
            target = target.sample_points_uniformly(number_of_points = self.n)

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

    def split_mesh_get_center_illustration(self, source_mesh, target_mesh=None, length=5, angle=None, center_point=None, normal=None,
                              clip_origin=None, debug=False): #call this function instead of split_mesh_get_center to get the illsutration used in the supplementary figures
        # cld = source_mesh.clip(normal=tuple(normal), origin=list(clip_origin), invert=False)

        #compatible with debug_list = ["Soft Closed 2.4 n1"]
        cld = source_mesh
        cube = pv.Cube(center=tuple(center_point),
                       x_length=50,
                       y_length=length,
                       z_length=length)
        angle = angle if angle is not None else np.arctan2(normal[2], normal[0]) * 180 / np.pi
        cube = cube.rotate_vector([0, 1, 0], -angle, center_point)  # get the cube
        intersection = cld.clip_box(cube, invert=False)
        mean_distance = np.mean(intersection["distance"])
        # Cube for clipping for visualization
        intersection2 = cld.clip_box(cube, invert=True)

        centroid = [16.0000, 32.3200, 13.7250]
        xlength = 50
        ylength = 30
        zlength = 23
        cube2 = pv.Cube(center=tuple(centroid),
                        x_length=xlength,
                        y_length=ylength,
                        z_length=zlength)
        cube2 = cube2.rotate_vector([0, 1, 0], -angle, centroid)  # get the cube

        intersection3 = cld.clip_box(cube2, invert=False)
        intersection4 = cld.clip_box(cube2, invert=True)

        clipped = intersection2.clip_scalar(scalars='distance', value=0.4, invert=False)
        clipped_invert = intersection2.clip_scalar(scalars='distance', value=0.4, invert=True)

        my_colormap = cmr.get_sub_cmap('cmr.neon_r', 0, 1)
        p = pv.Plotter(lighting='three lights')
        p.camera_position = [(83.4694551617329, 134.73970279806196, 115.89920527263094),
                             (-13.21325718649091, 15.913841123134265, 14.397938028966129),
                             (-0.4062820686756501, 0.7617039981438724, -0.504719625023908)]
        p.background_color = 'white'
        p.enable_point_picking(callback=self.display_picked_point)
        # p.add_mesh(cld, scalars="distance", cmap=my_colormap, color=True, opacity=0.9)
        p.add_mesh(intersection3, scalars="distance", cmap=my_colormap, opacity=1, clim=[0, 8])
        p.add_mesh(intersection4, scalars="distance", cmap=my_colormap, opacity=0.45, clim=[0, 8])
        # p.add_mesh(cube, opacity=0.5)
        p.add_mesh(intersection, scalars="distance", cmap=my_colormap, opacity=1, clim=[0, 8])
        p.add_mesh(target_mesh, opacity=0.05)
        p.save_graphic("Mean_vs_Max_Displacement_Illustration.svg", raster=False)
        cpos = p.show(return_cpos=True)

    def split_mesh_get_center(self, source_mesh, target_mesh = None, length = 5, angle = None, center_point = None, normal = None, clip_origin = None, debug = False):

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
                            **self.projection_clip_box)



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

        max_distance = np.max(cld["distance"])

        return cld,mean_distance, max_distance, intersection2

    def display_picked_point(self,p):
        print(p)
    def PyVista_show(self, source_mesh = None, target_mesh = None, grasper = None, dmin_mm = 0, dmax_mm = 7.5,plot_style = "project", show_plot = False,
                     debug = False, fname = None, show_axes = True, show_colorbar = True,
                     show_planes = False, camera_pos = None, fname_movie = None,
                     clip_base = False, clip_base_dim = None, return_max_distance = False):

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

        clip_offset_mag = 5  # offset from origin of the clip plane
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
        max_distance = []

        if plot_style != "project":
            p.subplot(0, 0)  # activate the large view

        p.set_viewup([0, 1, 0])

        if clip_base == True:
            clip_base_dim = clip_base_dim if clip_base_dim is not None else self.clip_base_dim
            clipped_cloud = source_mesh.clip_box(clip_base_dim, invert=False)
            clipped_cloud2 = source_mesh.clip_box(clip_base_dim, invert=True)
            p.add_mesh(clipped_cloud, scalars="distance", cmap=my_colormap, clim=[dmin_mm, dmax_mm], color=True)
            p.add_mesh(clipped_cloud2, color='ivory', opacity=1) #try ivory or cyan
            s_mesh = clipped_cloud #for use in split mesh for the projections later

        else:
            p.add_mesh(source_mesh, scalars="distance", cmap=my_colormap, clim=[dmin_mm, dmax_mm], color=True)
            s_mesh = source_mesh
        if show_axes == True:
            p.show_axes()

        cam_position_default = [(121.8811292356143, 112.98073524919948, -97.00768870196411),
                             (0.8108825716953554, 17.391831208985828, -1.449314787971784),
                             (-0.47772489624343795, 0.8450701284445499, 0.24007374183760427)]


        p.camera_position = cam_position_default if self.camera_pos is None else self.camera_pos

        # cpos = [(223.98191723264253, 232.29015597418473, -87.18915118775462),
        #         (0.8108825716953554, 17.391831208985828, -1.449314787971784),
        #         (-0.6707736342526432, 0.7352928425645483, 0.09699055245150295)]

        for ix, n in enumerate(normal):

            # Add light
            light = pv.Light(position=tuple(light_pos[ix]), color='white', cone_angle=25, exponent=20, intensity=0.25)
            light.positional = True

            # Split mesh
            cld, md, max_d, cld_clipped = self.split_mesh_get_center(s_mesh, target_mesh = target_mesh, center_point=self.center_pos[ix],
                                                 length=avg_length, angle=np.rad2deg(ang_offset[ix]),
                                                 normal=normal[ix], clip_origin=clip_origin[ix], debug=debug)

            clipped_data.append(cld)
            mean_distance.append(md)
            max_distance.append(max_d)

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

                projected = projected.translate(self.projected_translate)

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
            p.set_viewup([0, 1, 0])
            #print(self.camera_pos)
            p.camera_position = cam_position_default if self.camera_pos is None else self.camera_pos
            #print(p.camera_position)
            p.open_movie(filename, quality=10, framerate=30)

            for rot in ang_rot:
                rot_source = source_mesh.rotate_y(rot, point=(0, 0, 0), inplace=False)
                actor = p.add_mesh(rot_source, scalars="distance", cmap=my_colormap, clim=[dmin_mm, dmax_mm],
                                   color=True)
                rot_target = target_mesh.rotate_y(rot, point=(0, 0, 0), inplace=False)
                actor2 = p.add_mesh(rot_target, opacity = 0.175,
                                   color=True)
                #p.camera_position = cam_position_default
                p.remove_scalar_bar()
                p.write_frame()
                p.remove_actor(actor)
                p.remove_actor(actor2)

            p.close()




        if return_max_distance == False:
            return mean_distance
        else:
            return mean_distance, max_distance






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

    center_pos = None
    zoom = None
    projection_clip_box = None
    projected_translate = None
    cam_position = None
    clip_base_dim = None
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

    # target = "Objects\\Deformables\\Jan18th_2025_Rigid_OpenLoop_35mm_Prescan1.obj"
    # source = "Objects\\Deformables\\Jan18th_2025_Rigid_OpenLoop_35mm_Postscan1.obj"
    # grasper = "rigid"
    # control_type = "Open"
    # fname = "test_rigid.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.

    # target = "Objects\\Deformables\\Rigid_18thDecember_Gain5_N1_pre.obj"
    # source = "Objects\\Deformables\\Rigid_18thDecember_Gain5_N1_post.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "rigid"
    # control_type = "closed"
    # fname = "test_K4p28_rigid.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.

    # target = "Objects\\Deformables\\October20_2025_S boftClay_Rigid_K5_PreScan1.obj"
    # source = "Objects\\Deformables\\October20_2025_SoftClay_Rigid_K5_PostScan1.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "rigid"
    # control_type = "closed"
    # fname = "test_K5_rigid2.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.

    # target = "Objects\\Deformables\\October17_2025_Avocado_PostSoft_2.obj"
    # source = "Objects\\Deformables\\October17_2025_Avocado_PostRigid_2.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "rigid"
    # control_type = "open"
    # fname = "test_Avo2.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.
    # center_pos = [[23.02794647, 56.3971138, - 0.93143082], [-23.02794647, 56.3971138, - 0.93143082]]
    # zoom = 0.9
    # projection_clip_box = {"x_length": 120, "y_length": 90, "z_length": 120}
    # projected_translate = [0,-45, 0]
    # cam_position = [(213.52721270317406, 110.87607063093205, -105.46126743208039),
    #                         (-4.705856150566401, 34.92994333668984, 2.27341546851139),
    #                         (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)]

    # target = "Objects\\Deformables\\October17_2025_Avocado_PreGrasp_2.obj"
    # source = "Objects\\Deformables\\October17_2025_Avocado_PostSoft_2.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "soft"
    # control_type = "open"
    # fname = "test_Avo2_soft.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.
    # center_pos = [[21.927, 56.397, 9.4167], [-22.0046, 56.397, 11.706], [-7, 56.397,-24.1]]
    # zoom = 1.2
    # projection_clip_box = {"x_length": 120, "y_length": 90, "z_length": 120}
    # projected_translate = [0, -53, 0]
    # cam_position = [(213.52721270317406, 98.87607063093205, -105.46126743208039),
    #                 (-4.705856150566401, 34.92994333668984, 2.27341546851139),
    #                 (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)]

    # target = "Objects\\Deformables\\October18_2025_Tomato_Pre1.obj"
    # source = "Objects\\Deformables\\October18_2025_Tomato_PostSoft1.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "soft"
    # control_type = "open"
    # fname = "test_Tomato2.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.
    # zoom = 1.5
    # projection_clip_box = {"x_length": 120, "y_length": 90, "z_length": 120}
    # projected_translate = [0, -47, 0]
    # cam_position = [(213.52721270317406, 98.87607063093205, -105.46126743208039),
    #                 (-4.705856150566401, 34.92994333668984, 2.27341546851139),
    #                 (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)]

    # target = "Objects\\Deformables\\October18_2025_Tomato_PostSoft3.obj"
    # source = "Objects\\Deformables\\October19_2025_Tomato_PostRigid3.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "rigid"
    # control_type = "open"
    # fname = "test_Tomato3_rigid.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.
    # zoom = 1.5
    # projection_clip_box = {"x_length": 120, "y_length": 90, "z_length": 120}
    # projected_translate = [0, -47, 0]
    # center_pos = [[23.02794647, 46.7, - 0.93143082], [-23.02794647, 46.7, - 0.93143082]]
    # cam_position = [(213.52721270317406, 98.87607063093205, -105.46126743208039),
    #                 (-4.705856150566401, 34.92994333668984, 2.27341546851139),
    #                 (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)]

    # target = "Objects\\Deformables\\October19_2025_Strawberry_PostSoft1.obj"
    # source = "Objects\\Deformables\\October19_2025_Strawberry_PostRigid1.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "rigid"
    # control_type = "open"
    # fname = "test_strawberry.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.
    # center_pos = [[21.88280869, 27.63329697,  0.48669434], [-21.88280869, 27.63329697,  0.48669434]]
    # zoom = 1.6
    # projection_clip_box = {"x_length": 120, "y_length": 90, "z_length": 120}
    # projected_translate = [0, -30, -5]
    # cam_position = [(213.52721270317406, 98.87607063093205, -105.46126743208039),
    #                 (-4.705856150566401, 34.92994333668984, 2.27341546851139),
    #                 (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)]

    # target = "Objects\\Deformables\\October19_2025_Strawberry_Pre1.obj"
    # source = "Objects\\Deformables\\October19_2025_Strawberry_PostSoft1.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "soft"
    # control_type = "open"
    # fname = "test_strawberry_soft.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.
    # center_pos = [[21.927, 27.633, 9.4167], [-22.0046, 27.633, 11.706], [-7, 27.633,-24.1]]
    # zoom = 1.6
    # projection_clip_box = {"x_length": 120, "y_length": 90, "z_length": 120}
    # projected_translate = [0, -30, -5]
    # cam_position = [(213.52721270317406, 98.87607063093205, -105.46126743208039),
    #                 (-4.705856150566401, 34.92994333668984, 2.27341546851139),
    #                 (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)]

    # target = "Objects\\Deformables\\Oct28th_2025_RigidGrasper_K3_5N_PreScan1.obj"
    # source = "Objects\\Deformables\\Oct28th_2025_RigidGrasper_K3_5N_PostScan1.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "rigid"
    # control_type = "open"
    # fname = "oct28th_rigid.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.

    # target = "Objects\\Deformables\\Nov6th_Rigid_Open_45mm_PreScan2.obj"
    # source = "Objects\\Deformables\\Nov6th_Rigid_Open_45mm_PostScan2.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "rigid"
    # control_type = "open"
    # fname = "test_nov6th.npy"  # name of .npy file with the 4x4 homogeneous arrays for the source and target transforms.

    # target = "Objects\\Deformables\\Nov5th_K2p4_Rigid_PreScan4.obj"
    # source = "Objects\\Deformables\\Nov5th_K2p4_Rigid_PostScan4.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "rigid"
    # control_type = "open"
    # fname = "test_rigid_k2p4.npy"

    # target = "Objects\\Deformables\\Nov5th_K2p4_Soft_PreScan6.obj"
    # source = "Objects\\Deformables\\Nov5th_K2p4_Soft_PostScan6.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "soft"
    # control_type = "open"
    # fname = "test_soft_k2p4.npy"

    # target = "Objects\\Deformables\\Nov5th_K4_Soft_PreScan1.obj"
    # source = "Objects\\Deformables\\Nov5th_K4_Soft_PostScan1.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "soft"
    # control_type = "closed"
    # fname = "test_soft_k4.npy"

    # target = "Objects\\Deformables\\Nov19_Avocado_Prescan_4.obj"
    # source = "Objects\\Deformables\\Nov19_Avocado_PostSoft_4.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "soft"
    # control_type = "open"
    # fname = "test_avo_post_soft4.npy"

    # target = "Objects\\Deformables\\Nov19_Avocado_PostSoft_4.obj"
    # source = "Objects\\Deformables\\Nov19_Avocado_PostRigid_4.obj"
    # # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    # grasper = "rigid"
    # control_type = "open"
    # fname = "test_avo_post_rigid4.npy"

    target = "Objects\\Deformables\\Nov20_PlayDoh_Rigid_45mm_Prescan_4.obj"
    source = "Objects\\Deformables\\Nov20_PlayDoh_Rigid_45mm_Postscan_4.obj"
    # template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    grasper = "rigid"
    control_type = "open"
    fname = "test_rigid_open_45_n4.npy"
    clip_base_dim = [-np.inf, np.inf, 12, np.inf, -np.inf, np.inf]

    template = "Objects\\DeformableCylinder_Base_50mm_diameter.PLY"
    #template = "Objects\\DeformableCylinder_Base_50mm_diameter_with_placeholder.PLY"
    #template = "Objects\\DeformableCylinder_Base_50mm_diameter_sector.PLY"
    #template = "Objects\\DeformableCylinder_Base_50mm_diameter_sector.PLY"

    supply_transform = True  #supply transform == true uses the target and source homogeneous transforms applied above instead of doing RANSAC + ICP.
    save_transformation = True
    #align_method = "Manual Correspondence To Template then to each other"
    #align_method = "Crop Align Base"
    #align_method = "Default"
    align_method = "Manual Correspondence to template then auto align to each other"
    #align_method = "Crop base then align to each other"

    if supply_transform == False:
        deformable = DeformableClass(source_fname = source, target_fname = target, draw_intermediate = False,
                                     draw_final = True, grasper = grasper, control_type = control_type,
                                     center_pos = center_pos, zoom = zoom, projection_clip_box = projection_clip_box, projected_translate = projected_translate,
                                     camera_pos = cam_position,template_fname = template, clip_base_dim = clip_base_dim)
        if save_transformation == True:
            deformable.alignMesh(align_to_template = True, save_transform_name = fname, align_method = align_method)
        else:
            deformable.alignMesh(align_to_template=True, align_method = align_method)

        deformable.compute_chamfer_distance(source = deformable.source_mesh, target = deformable.target_mesh,crop = True, resample = True)
    else:
        with open(fname, 'rb') as f:
            t_transform = np.load(f)
            s_transform = np.load(f)
        deformable = DeformableClass(source_fname=source, target_fname=target, draw_intermediate=True,
                                     draw_final=True, grasper = grasper,control_type = control_type,
                                     center_pos = center_pos, zoom = zoom, projection_clip_box = projection_clip_box, projected_translate = projected_translate,
                                     camera_pos = cam_position, template_fname = template, clip_base_dim = clip_base_dim)
        source_original, source_ds, target_original, target_ds = deformable.center_rotate_meshes(
                                                                                                 source_fname=source,  target_fname=target, center_meshes=True, transform_source=s_transform,
                                                                                                 transform_target = t_transform)
        #deformable.draw_registration_result(source_original, target_original, transformation=np.eye(4))

        deformable.source_mesh_downsampled = source_ds
        deformable.target_mesh_downsampled = target_ds
        deformable.source_mesh = source_original
        deformable.target_mesh = target_original

        CD = deformable.compute_chamfer_distance(source = source_original, target = target_original,crop = True, resample = True) #need to crop with source_original and target_original (because these are triangle meshes that have the smaple_uniform command), and then downsample both with n after crop.

    deformable.convert_to_PyVista()
    deformable.compute_chamfer_distance_custom()
    md = deformable.PyVista_show(plot_style = "Separate",show_plot = True, debug = True,show_colorbar = True,show_axes = True,fname = "colorbar.svg",show_planes = False,fname_movie="test.mp4", clip_base = True,dmax_mm = 12 )

    pass















