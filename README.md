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

## Code Structure
The main environment for simulating a gantry system with a soft grasper and objects to pick is
in [GantrySimulation.py](SNS Training/Gantry/envs/GantrySimulation.py). The neural network parameters are defined in [SNS_layer.py](SNS Training/Gantry/controller/SNS_layer.py) and the supervised learning environment is defined in [train.py](SNS Training/Gantry/controller/train.py)

```bash
Gantry
├── controller
    ├── torchSNS
    ├── SNS_layer.py
    └── train.py
├── envs
    ├── GantrySimulation.py
    └── sinusoidgui_programmaticcheck.py
PickAndPlace_SoftGrasperCheck.py
PickAndPlace.py

```

## Verifying the Gantry Environment

You can run the [PickAndPlace_SoftGrasperCheck.py](PickAndPlace_SoftGrasperCheck.py) script to verify your environment setup.

```bash
python -m PickAndPlace_SoftGrasperCheck.py
```

If it runs then you have set the gantry
environment correctly.

## Training a Model
To train the sensory layer, run [train.py](Gantry/controller/train.py).

```bash
python -m Gantry.controller.train
```

## Evaluating the Controller

To evaluate the designed SNS controller, run [PickAndPlace.py](PickAndPlace.py).

```bash
python PickAndPlace.py
```

## Support

For questions about the code, please create an issue in the repository.
