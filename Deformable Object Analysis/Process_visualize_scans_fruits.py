"""
Mesh Alignment + Chamfer Distance Pipeline (Fruits Dataset)
-----------------------------------------------------------

This script processes deformable-object scan pairs (initial + final) for
fruit experiments (strawberry, avocado, tomato). It:

1. Loads experiment metadata from an Excel file
2. Loads OBJ meshes for initial and final scans
3. Either computes or loads precomputed alignment transforms
4. Aligns meshes using DeformableClass
5. Computes Chamfer distance between initial and final meshes
6. Extracts jaw compression metrics from PyVista visualization
7. Saves results back into a CSV

The script supports:
- Rigid and soft grasper types
- Open and closed control modes
- Object-specific visualization settings
- Debug mode for selective row processing

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
    fname = "ScanMap_Fruits.xlsx"
    DF = pd.read_excel(fname)

    # ---------------------------------------------------------
    # DIRECTORY CONFIGURATION
    # ---------------------------------------------------------
    transform_directory = "Objects\\Processed Scans"   # where .npy transforms are stored
    picture_directory   = "Objects\\Pictures"          # where screenshots/videos are saved
    obj_dir             = "Objects\\Deformables\\Fruits"       # where OBJ meshes are stored

    supply_transform      = True   # True = load saved transforms, False = compute new ones
    save_transformations  = True   # Save new transforms when computing them

    # ---------------------------------------------------------
    # DEBUGGING OPTIONS
    # ---------------------------------------------------------
    debug = False
    debug_list = ["Rigid Open Avocado n3"]
    # debug_list = ["Rigid Open Strawberry n3"]  # Only process these Master IDs when debug=True

    # ---------------------------------------------------------
    # ALIGNMENT SETTINGS
    # ---------------------------------------------------------
    align_method = "Default"
    # Other options:
    # "Default"
    # "Manual Correspondence To Template then to each other"
    # "Crop Align Base"
    # "Manual Correspondence of source to template then auto align to each other"

    # ---------------------------------------------------------
    # STORAGE FOR RESULTS
    # ---------------------------------------------------------
    displaced_distances = [[], [], []]  # mean compression for each jaw
    chamfer_distances   = []            # Chamfer distance per row


    # ---------------------------------------------------------
    # OBJECT-SPECIFIC VISUALIZATION SETTINGS
    # ---------------------------------------------------------
    # These control camera position, zoom, clipping, and translation
    # for each fruit × grasper type combination.
    visualization_dict = {
        "strawberry": {
            "rigid": {
                "zoom": 1.1,
                "projection_clip_box": {"x_length": 150, "y_length": 120, "z_length": 150},
                "projected_translate": [0, -30, 0],
                "center_pos": [
                    [21.88280869, 27.63329697, 0.48669434],
                    [-5, 27.63329697, -3]
                ],
                "camera_pos": [
                    (213.52721270317406, 98.87607063093205, -105.46126743208039),
                    (-4.705856150566401, 34.92994333668984, 2.27341546851139),
                    (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)
                ]
            },
            "soft": {
                "zoom": 1.3,
                "projection_clip_box": {"x_length": 120, "y_length": 90, "z_length": 120},
                "projected_translate": [0, -30, -5],
                "center_pos": [
                    [21.927, 27.633, 9.4167],
                    [-22.0046, 27.633, 11.706],
                    [-7, 27.633, -24.1]
                ],
                "camera_pos": [
                    (213.52721270317406, 98.87607063093205, -105.46126743208039),
                    (-4.705856150566401, 34.92994333668984, 2.27341546851139),
                    (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)
                ]
            }
        },

        # -----------------------------------------------------
        # Avocado visualization settings
        # -----------------------------------------------------
        "avocado": {
            "rigid": {
                "zoom": 0.81,
                "projection_clip_box": {"x_length": 120, "y_length": 90, "z_length": 120},
                "projected_translate": [0, -45, 0],
                "center_pos": [
                    [23.02794647, 56.3971138, -0.93143082],
                    [-23.02794647, 56.3971138, -0.93143082]
                ],
                "camera_pos": [
                    (257.2602019740628, 132.10817382580038, -128.60675452583604),
                    (-6.801811338963276, 40.21335979976731, 1.7522117838802544),
                    (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)
                ]
            },
            "soft": {
                "zoom": 0.95,
                "projection_clip_box": {"x_length": 120, "y_length": 90, "z_length": 120},
                "projected_translate": [0, -53, 0],
                "center_pos": [
                    [21.927, 56.397, 9.4167],
                    [-22.0046, 56.397, 11.706],
                    [-7, 56.397, -24.1]
                ],
                "camera_pos": [
                    (257.2602019740628, 132.10817382580038, -128.60675452583604),
                    (-6.801811338963276, 40.21335979976731, 1.7522117838802544),
                    (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)
                ]
            }
        },

        # -----------------------------------------------------
        # Tomato visualization settings
        # -----------------------------------------------------
        "tomato": {
            "rigid": {
                "zoom": 1.1,
                "projection_clip_box": {"x_length": 120, "y_length": 90, "z_length": 120},
                "projected_translate": [0, -47, 0],
                "center_pos": [
                    [23.02794647, 46.7, -0.93143082],
                    [-23.02794647, 46.7, -0.93143082]
                ],
                "camera_pos": [
                    (213.52721270317406, 98.87607063093205, -105.46126743208039),
                    (-4.705856150566401, 34.92994333668984, 2.27341546851139),
                    (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)
                ]
            },
            "soft": {
                "zoom": 1.1,
                "projection_clip_box": {"x_length": 120, "y_length": 90, "z_length": 120},
                "projected_translate": [0, -47, 0],
                "center_pos": [
                    [21.927, 46.7, 9.4167],
                    [-22.0046, 46.7, 11.706],
                    [-7, 46.7, -24.1]
                ],
                "camera_pos": [
                    (213.52721270317406, 98.87607063093205, -105.46126743208039),
                    (-4.705856150566401, 34.92994333668984, 2.27341546851139),
                    (-0.2591623357186034, 0.9646599675819324, 0.0476028432943153)
                ]
            }
        }
    }

    # ---------------------------------------------------------
    # MAIN PROCESSING LOOP
    # ---------------------------------------------------------
    for index, row in DF.iterrows():

        # Skip rows unless debugging is disabled or row is in debug_list
        if debug and row["Master ID"] not in debug_list:
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

        # Select visualization settings for this fruit × grasper
        vdict = visualization_dict[row["Experimental Condition"].lower()][row["Grasper "].lower()]

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
                **vdict
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
                draw_final=True,
                grasper=row["Grasper "],
                control_type=row["Control Type"],
                **vdict
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

        picture_name = os.path.join(picture_directory, fn + ".svg")
        movie_name   = os.path.join(picture_directory, fn + "_undeformed_overlay.mp4")

        mean_distance = deformable.PyVista_show(
            plot_style="Separate",
            show_colorbar=False,
            show_axes=False,
            show_plot=False,
            fname=picture_name,
            fname_movie=movie_name,
            clip_base=True,
            dmax_mm=12
        )

        # Rigid grasper has only 2 jaws → pad with NaN
        if row["Grasper "] == "Rigid":
            mean_distance.append(np.nan)

        # Store results
        for ix, x in enumerate(mean_distance):
            displaced_distances[ix].append(x)

    # ---------------------------------------------------------
    # SAVE RESULTS BACK INTO DATAFRAME
    # ---------------------------------------------------------
    for i in range(3):
        DF[f"Mean Compression {i+1} (mm)"] = displaced_distances[i]

    DF["Computed CD"] = chamfer_distances

    DF.to_csv("ScanMap_w_displacements_Fruits.csv")

    pass
