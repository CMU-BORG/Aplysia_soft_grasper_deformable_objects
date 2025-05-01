from DeformableClass import DeformableClass
import pyvista as pv
import copy
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":

    fname = "MaxForce.xlsx"
    DF = pd.read_excel(fname)

    transform_directory = "Objects\Processed Scans"
    picture_directory = "Objects\Pictures"
    obj_dir = "Objects\Deformables"
    supply_transform = True
    save_transformations = True  # set to True if you want to save transformations
    debug = False #set to true if you want to look at individual rows in the dataframe
    debug_list = ["Soft Closed 15.0 n1",
                  "Soft Closed 5.0 n1",
                  "Soft w/ Rigid Jaws Open 35.0 n1",
                  "Soft w/ Rigid Jaws Open 45.0 n2",
                  "Soft Open 45.0 n2",
                  "Soft Open 35.0 n2",
                  "Soft Open 40.0 n3"]


    displaced_distances = [[], [], []]

    mean_by_category = DF.groupby(['Grasper ', 'Control Type', 'Experimental Condition'])['Chamfer Distance'].mean()
    std_by_category = DF.groupby(['Grasper ', 'Control Type', 'Experimental Condition'])['Chamfer Distance'].std()

    for index, row in DF.iterrows():

        if debug == True:
            # label = input("Enter row number") #manually select row number
            # index = int(label)
            # row = DF.iloc[index,:]
            if (row["Master ID"] not in debug_list):
                continue





        fi = os.path.join(obj_dir, row[
            "Name of Initial Scan"])  # initial scan is the target (one that we will transform the final scan to match)
        ff = os.path.join(obj_dir, row["Name of Final Scan"])  # final scan is the source

        fn = row["Master ID"].replace('/','') #doesn't work good for saving files if the forward slash in soft w/ Rigid Jaws is there
        transform_name = os.path.join(transform_directory, fn + ".npy")
        print("------ %s ------"%fn)

        if supply_transform == False:

            deformable = DeformableClass(source_fname=ff, target_fname=fi, draw_intermediate=False, draw_final=False, grasper = row["Grasper "])
            if save_transformations == True:
                deformable.alignMesh(align_to_template=True, save_transform_name=transform_name)
            else:
                deformable.alignMesh(align_to_template=True)

        else:  # if not supplying the transform, use the filename
            with open(transform_name, 'rb') as f:
                t_transform = np.load(f)
                s_transform = np.load(f)
            deformable = DeformableClass(source_fname=ff, target_fname=fi, draw_intermediate=False,
                                         draw_final=False, grasper = row["Grasper "], control_type = row["Control Type"])

            source_original, source_ds, target_original, target_ds = deformable.center_rotate_meshes(
                source_fname=ff, target_fname=fi, center_meshes=True, transform_source=s_transform,
                transform_target=t_transform)
            # deformable.draw_registration_result(source_original, target_original, transformation=np.eye(4))

            deformable.source_mesh_downsampled = source_ds
            deformable.target_mesh_downsampled = target_ds
            deformable.source_mesh = source_original
            deformable.target_mesh = target_original

            deformable.compute_chamfer_distance(source=source_ds, target=target_ds)

        deformable.convert_to_PyVista()
        deformable.compute_chamfer_distance_custom()
        picture_name = os.path.join(picture_directory, fn + ".svg") #if debug == False else None #only save screenshots when not in debug mode
        #picture_name = None #temporary to not save images, TODO: delete this line
        movie_name = os.path.join(picture_directory, fn + "_undeformed_overlay.mp4")
        mean_distance = deformable.PyVista_show(plot_style="Separate", show_colorbar = False,show_axes = False,
                                                show_plot=False, fname = picture_name,fname_movie = movie_name)  # get the mean distance for the three jaws.

        if row["Grasper "] == "Rigid":
            mean_distance.append(np.NaN)

        [displaced_distances[ix].append(x) for (ix, x) in enumerate(mean_distance)]

    for i in range(0, 3):
        DF["Mean Compression %i (mm)" % (i + 1)] = displaced_distances[i]

    DF.to_csv("MaxForce_w_displacement.csv")
    pass
