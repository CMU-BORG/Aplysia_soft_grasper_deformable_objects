"""
Mesh Alignment + Chamfer Distance Pipeline
------------------------------------------

This script processes deformable-object scan pairs (initial + final),
aligns meshes, computes Chamfer distances, extracts jaw compression metrics,
and saves results back into a CSV.

It uses:
- A DeformableClass object for mesh alignment, downsampling, and metrics
- An Excel file describing each experiment row
- OBJ files for initial and final scans
- Optional precomputed transformations
- PyVista for visualization and screenshot/video generation

Author: (your name)
"""

from DeformableClass import DeformableClass
import pyvista as pv
import copy
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":

    # ---------------------------------------------------------
    # LOAD EXPERIMENT MAP
    # ---------------------------------------------------------
    fname = "ScanMap_Playdoh.xlsx"
    DF = pd.read_excel(fname)

    # ---------------------------------------------------------
    # DIRECTORY CONFIGURATION
    # ---------------------------------------------------------
    transform_directory = "Objects\\Processed Scans"   # where .npy transforms are stored
    picture_directory   = "Objects\\Pictures"          # where screenshots/videos are saved
    obj_dir             = "Objects\\Deformables\\PlayDoh"       # where OBJ meshes are stored

    supply_transform      = True   # True = load saved transforms, False = compute new ones
    save_transformations  = True   # Save new transforms when computing them

    # ---------------------------------------------------------
    # DEBUGGING OPTIONS
    # ---------------------------------------------------------
    debug = False
    # debug_list = ["Rigid Closed 3.0 n1", ...]
    debug_list = ["Soft Closed 2.4 n1"]  # Only process these Master IDs when debug=True

    # ---------------------------------------------------------
    # ALIGNMENT SETTINGS
    # ---------------------------------------------------------
    # align_method = "Crop Align Base"
    align_method = "Manual Correspondence to template then auto align to each other"
    # align_method = "Default"

    n = 100000  # number of points for downsampling
    clip_base_dim = [-np.inf, np.inf, 12, np.inf, -np.inf, np.inf]  # cropping region

    # ---------------------------------------------------------
    # STORAGE FOR RESULTS
    # ---------------------------------------------------------
    displaced_distances     = [[], [], []]  # mean compression for each jaw
    max_displaced_distances = [[], [], []]  # max compression for each jaw
    chamfer_distances       = []            # Chamfer distance per row


    # Max visualization range per control type
    dmax_mm_control = {"Open": 8, "Closed": 12}

    # ---------------------------------------------------------
    # MAIN PROCESSING LOOP
    # ---------------------------------------------------------
    for index, row in DF.iterrows():

        # Skip rows unless debugging is disabled or row is in debug_list
        if debug:
            if row["Master ID"] not in debug_list:
                continue

        # -----------------------------------------------------
        # BUILD FILE PATHS
        # -----------------------------------------------------
        fi = os.path.join(obj_dir, row["Name of Initial Scan"])  # target mesh
        ff = os.path.join(obj_dir, row["Name of Final Scan"])    # source mesh

        # Clean Master ID for filenames (remove '/')
        fn = row["Master ID"].replace('/', '')
        transform_name = os.path.join(transform_directory, fn + ".npy")

        print("------ %s ------" % fn)

        # -----------------------------------------------------
        # ALIGNMENT: COMPUTE OR LOAD TRANSFORM
        # -----------------------------------------------------
        if not supply_transform:
            # Compute new transform
            deformable = DeformableClass(
                source_fname=ff,
                target_fname=fi,
                draw_intermediate=False,
                draw_final=False,
                grasper=row["Grasper "],
                n=n,
                clip_base_dim=clip_base_dim
            )

            if save_transformations:
                deformable.alignMesh(
                    align_to_template=True,
                    save_transform_name=transform_name,
                    align_method=align_method
                )
            else:
                deformable.alignMesh(
                    align_to_template=True,
                    align_method=align_method
                )

        else:
            # Load precomputed transform
            with open(transform_name, 'rb') as f:
                t_transform = np.load(f)
                s_transform = np.load(f)

            deformable = DeformableClass(
                source_fname=ff,
                target_fname=fi,
                draw_intermediate=False,
                draw_final=False,
                grasper=row["Grasper "],
                control_type=row["Control Type"],
                n=n
            )

            # Apply transforms and center meshes
            source_original, source_ds, target_original, target_ds = deformable.center_rotate_meshes(
                source_fname=ff,
                target_fname=fi,
                center_meshes=True,
                transform_source=s_transform,
                transform_target=t_transform
            )

            # Store aligned meshes
            deformable.source_mesh_downsampled = source_ds
            deformable.target_mesh_downsampled = target_ds
            deformable.source_mesh = source_original
            deformable.target_mesh = target_original

        # -----------------------------------------------------
        # CHAMFER DISTANCE
        # -----------------------------------------------------
        CD = deformable.compute_chamfer_distance(
            source=deformable.source_mesh,
            target=deformable.target_mesh,
            crop=True,
            resample=True
        )
        chamfer_distances.append(CD)
        print("*************\t CD = %f \t*************" % CD)

        # -----------------------------------------------------
        # VISUALIZATION + COMPRESSION METRICS
        # -----------------------------------------------------
        deformable.convert_to_PyVista()
        deformable.compute_chamfer_distance_custom()

        picture_name = os.path.join(picture_directory, fn + ".svg")
        movie_name   = os.path.join(picture_directory, fn + "_undeformed_overlay.mp4")

        dmax_mm_c = dmax_mm_control[row["Control Type"]]

        mean_distance, max_distance = deformable.PyVista_show(
            plot_style="Separate",
            show_colorbar=False,
            show_axes=False,
            show_plot=False,
            fname=picture_name,
            fname_movie=movie_name,
            clip_base=True,
            dmax_mm=dmax_mm_c,
            return_max_distance=True
        )

        # Rigid grasper has only 2 jaws → pad with NaN
        if row["Grasper "] == "Rigid":
            mean_distance.append(np.nan)
            max_distance.append(np.nan)

        # Store results
        for ix, x in enumerate(mean_distance):
            displaced_distances[ix].append(x)

        for ix, x in enumerate(max_distance):
            max_displaced_distances[ix].append(x)

    # ---------------------------------------------------------
    # SAVE RESULTS BACK INTO DATAFRAME
    # ---------------------------------------------------------
    for i in range(3):
        DF[f"Mean Compression {i+1} (mm)"] = displaced_distances[i]
        DF[f"Max Compression {i+1} (mm)"]  = max_displaced_distances[i]

    DF["Computed CD"] = chamfer_distances

    # Save final CSV
    DF.to_csv("ScanMap_w_displacements_PlayDoh.csv")

    pass
