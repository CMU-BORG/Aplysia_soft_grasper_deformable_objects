# FRR-Adaptive-Grasping

## Overview
This repository provides an implementation of the soft grasper controlled by a Synthetic Nervous Systems (SNSs) controller.

<p align="center">
    <img src="pick_and_place.gif" width="289.2" height="316.8" />
</p>



This environment builds upon the Pybullet simulator (https://pybullet.org/wordpress) and the [keras ncp](https://github.com/mlech26l/keras-ncp) by Mathias Lechner, Institute of Science and Technology Austria (IST Austria) ([Paper](https://www.nature.com/articles/s42256-020-00237-3?ref=https://coder.social)).

The current release provides following features:
* Implementation of the gantry system and simple objects in Pybullet.
* Implementation of a soft grasper and its contact dynamics in Pybullet.
* Implementation of supervised learning for SNS.
* Support for using SNS control policies to accomplish pick-and-place tasks using the soft and rigid graspers.
* Code to align and measure the chamfer distance between pre-grasp and post-grasp scanned point clouds of plastically deformable objects.

## Training the SNS
The main environment for simulating a gantry system with a soft grasper and objects to pick is
in [GantrySimulation.py](SNS Training/Gantry/envs/GantrySimulation.py). The neural network parameters are defined in [SNS_layer.py](SNS Training/Gantry/controller/SNS_layer.py) and the supervised learning environment is defined in [train.py](SNS Training/Gantry/controller/train.py)

```bash
SNS Training
├── Gantry
    ├── controller
        ├── torchSNS
        ├── SNS_layer.py
        └── train.py
    ├── envs
        ├── GantrySimulation.py
        └── sinusoidgui_programmaticcheck.py
    test.ipynb

```

### Verifying the Gantry Environment

You can run the [PickAndPlace_SoftGrasperCheck.py](PickAndPlace_SoftGrasperCheck.py) script to verify your environment setup.

```bash
python -m PickAndPlace_SoftGrasperCheck.py
```

If it runs then you have set the gantry
environment correctly.

### Training a Model
To train the SNS layer, run [train.py](SNS Training/Gantry/controller/train.py).

```bash
python -m Gantry.controller.train
```

### Evaluating the Controller

To evaluate the designed SNS controller, run [test.ipynb](SNS Training/test.ipynb).

```bash
python test.ipynb
```


## Controlling the robot

The code for controlling the robot is run from [Robot Experiments/GUI/EmbeddedSystems/IntegratedSystem/IntegratedSystem.py]. The individual libraries are contained within 
[Robot Experiments/GUI/EmbeddedSystems].


### Gantry Control

Runs from [Robot Experiments/GUI/EmbeddedSystems/Gantry/GantryController.py].

### Joystick Control

Class is contained in [Robot Experiments/GUI/EmbeddedSystems/JoyCon/JoyCon.py]

### Rigid Grasper Control

Class for communicating with the rigid grasper from the PC is contained in [Robot Experiments/GUI/EmbeddedSystems/RigidGrasper/RigidGrasper.py]. Code to communicate with the Dynamixel motors is adapted from [here](https://github.com/ROBOTIS-GIT/emanual/blob/master/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python/python_read_write_protocol_2_0.md)

The Teensy code to read from the load cell is contained in [Robot Experiments/GUI/EmbeddedSystems/RigidGrasper/LoadCellForceSensor/src/main.cpp]

### Soft Grasper Control

Class for communicating with the soft grasper from the PC is contained in [Robot Experiments/GUI/EmbeddedSystems/SoftGrasper/SoftGrasper.py]. The Teensy controller code is contained in [Robot Experiments/GUI/EmbeddedSystems/SoftGrasper/SoftGrasperController/src/ManageSoftGrasper.cpp]

### SNS controllers

The different flavors of the SNS controller are contained within [Robot Experiments/GUI/EmbeddedSystems/SNS].

## Aligning Point Clouds and quantifying deformation

To align point clouds, enter the .obj of the pre- and post- grasp point clouds in [Deformable Object Analysis/MaxForce.xlsx].  Then, run [Deformable Object Analysis/Process_visualize_scans.py] to batch align, generate figures and videos, and compute Chamfer Distance for the different experiments. The .obj files and datalogs available upon request. 

## Support

For questions about the code, please create an issue in the repository.
