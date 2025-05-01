import time
import copy
import pathlib
import numpy as np
import torch
import torch.nn as nn
import pybullet as p
import pybullet_data
from . import torchSNS as tSNS
from .torchSNS.torch import SNSCell, SNSCell_modulation
from ...Gantry.envs.GantrySimulation import GantrySimulation
from pathlib import Path

#########################################################


THETA_MAX = np.array([0.2, 0.2, -1, 0.1])
THETA_MIN = np.array([-0.2, -0.2, 0, 0])
F_MAX = np.array([50, 50, 50])
F_MIN = np.array([0, 0, 0])
SENSORY_LAYER_1_INPUT_SIZE = 9
SENSORY_LAYER_1_SIZE = 12
SENSORY_LAYER_2_INPUT_SIZE = 12 + 3
SENSORY_LAYER_2_SIZE = 8
COMMAND_LAYER_INPUT_SIZE = 8
COMMAND_LAYER_SIZE = 8
INTER_LAYER_1_INPUT_SIZE = 8
INTER_LAYER_1_SIZE = 4
INTER_LAYER_2_INPUT_SIZE = 12
INTER_LAYER_2_SIZE = 10
MOTOR_LAYER_INPUT_SIZE = 12
MOTOR_LAYER_MODULATION_INPUT_SIZE = 13
MOTOR_LAYER_SIZE = 4
R = 20


#########################################################


# Parameter initialization for SNS layers
def layer_initialization(layer, theta_min_in, theta_max_in, theta_min_out, theta_max_out, tau, b, sensory_erev,
                         sensory_w, R):
    in_features = layer._wiring.input_dim
    out_features = layer._wiring.output_dim
    neuron_num = layer._wiring.units
    with torch.no_grad():
        layer._params["tau"].requires_grad = False
        layer._params["b"].requires_grad = False
        layer._params["input_w"].requires_grad = False
        layer._params["input_b"].requires_grad = False
        layer._params["output_w"].requires_grad = False
        layer._params["output_b"].requires_grad = False
        layer._params["sensory_mu"].requires_grad = False
        layer._params["sensory_sigma"].requires_grad = False
        layer._params["mu"].requires_grad = False
        layer._params["sigma"].requires_grad = False
        layer._params["sensory_erev"].requires_grad = False
        layer._params["sensory_w"].requires_grad = False
        layer._params["erev"].requires_grad = False
        layer._params["w"].requires_grad = False

        layer._params["tau"].data = tau
        layer._params["b"].data = b
        layer._params["input_w"].data = torch.Tensor(R / (theta_max_in - theta_min_in))
        layer._params["input_b"].data = torch.Tensor(R * theta_min_in / (theta_min_in - theta_max_in))
        layer._params["output_w"].data = torch.Tensor((theta_max_out - theta_min_out) / R)
        layer._params["output_b"].data = torch.Tensor(theta_min_out)
        layer._params["sensory_mu"].data = R / 2 * torch.ones((in_features, neuron_num))
        layer._params["sensory_sigma"].data = R / 2 * torch.ones((in_features, neuron_num))
        layer._params["mu"].data = R / 2 * torch.ones((neuron_num, neuron_num))
        layer._params["sigma"].data = R / 2 * torch.ones((neuron_num, neuron_num))
        layer._params["sensory_erev"].data = sensory_erev
        layer._params["sensory_w"].data = sensory_w
        layer._params["erev"].data = torch.zeros((neuron_num, neuron_num))
        layer._params["w"].data = torch.zeros((neuron_num, neuron_num))


# Parameter initialization for SNS layers with modulation inputs
def modulation_layer_initialization(layer, theta_min_in, theta_max_in, theta_min_out, theta_max_out, tau, b, tau_io,
                                    t_d, beta, k_io, sensory_erev, sensory_w, R):
    layer_initialization(layer, theta_min_in, theta_max_in, theta_min_out, theta_max_out, tau, b, sensory_erev,
                         sensory_w, R)
    # in_features = layer._wiring.input_dim
    # out_features = layer._wiring.output_dim
    # neuron_num = layer._wiring.units
    with torch.no_grad():
        layer._params["tau_io"].requires_grad = False
        layer._params["t_d"].requires_grad = False
        layer._params["beta"].requires_grad = False
        layer._params["k_io"].requires_grad = False

        layer._params["tau_io"].data = tau_io
        layer._params["t_d"].data = t_d
        layer._params["beta"].data = beta
        layer._params["k_io"].data = k_io


# Create a SNS layer with layer_input_size input neurons and layer_size neurons. sparsity_mask reflects sparsity in the input synaptic connections.
def SNS_layer(layer_input_size, layer_size, sparsity_mask, sparsity_mask_modulation=None, tau=None, tau_io=None,
              t_d=None, beta=None, k_io=None, theta_min_in=None, theta_max_in=None, theta_min_out=None,
              theta_max_out=None, R=20):
    if tau is None:
        tau = torch.zeros(layer_size)
    if theta_min_in is None:
        theta_min_in = torch.zeros(layer_input_size)
    if theta_max_in is None:
        theta_max_in = R * torch.ones(layer_input_size)
    if theta_min_out is None:
        theta_min_out = torch.zeros(layer_size)
    if theta_max_out is None:
        theta_max_out = R * torch.ones(layer_size)
    if sparsity_mask_modulation is not None:
        if tau_io is None:
            tau_io = 1.2 * torch.ones(layer_size)
        if t_d is None:
            t_d = 2 * torch.ones(layer_size)
        if beta is None:
            beta = 0.0 * torch.ones(layer_size)
        if k_io is None:
            k_io = 5 * torch.ones(layer_size) * 2

    wiring = tSNS.wirings.FullyConnected(
        layer_size, layer_size, self_connections=False, erev_init_seed=np.random.randint(0, 10000))
    layer = SNSCell(wiring, layer_input_size)
    config = layer._wiring.get_config()
    config["adjacency_matrix"] = np.zeros([layer_size, layer_size])
    config["sensory_adjacency_matrix"] = sparsity_mask * config["sensory_adjacency_matrix"]
    if sparsity_mask_modulation is not None:
        config["modulation_adjacency_matrix"] = sparsity_mask_modulation * np.ones([layer_input_size, layer_size])
    new_wiring = tSNS.wirings.Wiring.from_config(config)
    if sparsity_mask_modulation is None:
        layer = SNSCell(new_wiring, layer_input_size, ode_unfolds=1, elapsed_time=1 / 240)
    else:
        layer = SNSCell_modulation(new_wiring, layer_input_size, ode_unfolds=1, elapsed_time=1 / 240)
    b = torch.Tensor(R * theta_min_out / (theta_min_out - theta_max_out))
    sensory_erev = (R - b).reshape(1, -1).repeat(layer_input_size, 1) * torch.Tensor(sparsity_mask)
    sensory_w = torch.zeros((layer_input_size, layer_size))
    if sparsity_mask_modulation is None:
        layer_initialization(layer, theta_min_in=theta_min_in, theta_max_in=theta_max_in, theta_min_out=theta_min_out,
                             theta_max_out=theta_max_out, tau=tau, b=b, sensory_erev=sensory_erev, sensory_w=sensory_w,
                             R=R)
    else:
        modulation_layer_initialization(layer, theta_min_in=theta_min_in, theta_max_in=theta_max_in,
                                        theta_min_out=theta_min_out, theta_max_out=theta_max_out, tau=tau, b=b,
                                        tau_io=tau_io, t_d=t_d, beta=beta, k_io=k_io, sensory_erev=sensory_erev,
                                        sensory_w=sensory_w, R=R)

    return layer


# Perception network
class SNS_Perception(nn.Module):
    def __init__(self, sensory_layer_1, sensory_layer_2, command_layer):
        super(SNS_Perception, self).__init__()
        self._sensory_layer_1 = sensory_layer_1
        self._sensory_layer_2 = sensory_layer_2
        self._command_layer = command_layer
        self.reset()

    def forward(self, gripper_position, object_position, target_position, force):
        position_input = torch.cat((gripper_position, object_position, target_position), dim=1)
        _, self._sensory_layer_1_state = self._sensory_layer_1.forward(position_input, self._sensory_layer_1_state)
        sensory_layer_2_input = torch.cat((self._sensory_layer_1_state, force), dim=1)
        _, self._sensory_layer_2_state = self._sensory_layer_2.forward(sensory_layer_2_input,
                                                                       self._sensory_layer_2_state)
        output, self._command_layer_state = self._command_layer.forward(self._sensory_layer_2_state,
                                                                        self._command_layer_state)

        # return output, self._sensory_layer_2_state
        force_output = self._sensory_layer_2_state[:, -1]
        return output, force_output.reshape(-1, 1)

    def reset(self):
        self._sensory_layer_1_state = torch.zeros((1, self._sensory_layer_1.state_size))
        self._sensory_layer_2_state = torch.zeros((1, self._sensory_layer_2.state_size))
        self._command_layer_state = torch.zeros((1, self._command_layer.state_size))
        self._command_layer_state[0, -1] = -R
        self._command_layer_state[0, 1] = -R

    def set_tau(self, tau):
        self._sensory_layer_1._params["tau"] = tau * torch.ones_like(self._sensory_layer_1._params["tau"])
        self._sensory_layer_2._params["tau"] = tau * torch.ones_like(self._sensory_layer_2._params["tau"])
        self._command_layer._params["tau"] = tau * torch.ones_like(self._command_layer._params["tau"])


# Open loop control network
class SNS_Control_open_loop(nn.Module):
    def __init__(self, inter_layer_1, inter_layer_2, motor_layer):
        super(SNS_Control_open_loop, self).__init__()
        self._inter_layer_1 = inter_layer_1
        self._inter_layer_2 = inter_layer_2
        self._motor_layer = motor_layer
        self.reset()

    def forward(self, object_position, target_position, input):
        _, self._inter_layer_1_state = self._inter_layer_1.forward(input, self._inter_layer_1_state)
        _, self._inter_layer_2_state = self._inter_layer_2.forward(torch.cat((object_position.repeat_interleave(2,
                                                                                                                dim=1)[
                                                                              :, :-1],
                                                                              target_position.repeat_interleave(2,
                                                                                                                dim=1)[
                                                                              :, :-1],
                                                                              self._inter_layer_1_state[:, :2]), dim=1),
                                                                   self._inter_layer_2_state)
        output, self._motor_layer_state = self._motor_layer.forward(
            torch.cat((self._inter_layer_2_state, self._inter_layer_1_state[:, -2:]), dim=1), self._motor_layer_state)

        # return output.squeeze(dim=0), self._inter_layer_2_state
        return output.squeeze(dim=0)

    def reset(self):
        self._inter_layer_1_state = torch.zeros((1, self._inter_layer_1.state_size))
        self._inter_layer_1_state[0, 0] = R
        self._inter_layer_1_state[0, 1] = -R
        self._inter_layer_2_state = torch.zeros((1, self._inter_layer_2.state_size))
        self._motor_layer_state = torch.Tensor([[R / 2, R / 2, 0, 0]])


# Closed loop control network v1
class SNS_Control_closed_loop_v1(nn.Module):
    def __init__(self, inter_layer_1, inter_layer_2, motor_layer):
        super(SNS_Control_closed_loop_v1, self).__init__()
        self._inter_layer_1 = inter_layer_1
        self._inter_layer_2 = inter_layer_2
        self._motor_layer = motor_layer
        self.reset()

    def forward(self, object_position, target_position, input, force_input):
        closed_loop_input = torch.cat((input, force_input), dim=1)
        _, self._inter_layer_1_state = self._inter_layer_1.forward(closed_loop_input, self._inter_layer_1_state)
        _, self._inter_layer_2_state = self._inter_layer_2.forward(torch.cat((object_position.repeat_interleave(2,
                                                                                                                dim=1)[
                                                                              :, :-1],
                                                                              target_position.repeat_interleave(2,
                                                                                                                dim=1)[
                                                                              :, :-1],
                                                                              self._inter_layer_1_state[:, :2]), dim=1),
                                                                   self._inter_layer_2_state)
        output, self._motor_layer_state = self._motor_layer.forward(
            torch.cat((self._inter_layer_2_state, self._inter_layer_1_state[:, -2:]), dim=1), self._motor_layer_state)

        # return output.squeeze(dim=0), self._inter_layer_2_state
        return output.squeeze(dim=0)

    def reset(self):
        self._inter_layer_1_state = torch.zeros((1, self._inter_layer_1.state_size))
        self._inter_layer_1_state[0, 0] = R
        self._inter_layer_1_state[0, 1] = -R
        self._inter_layer_2_state = torch.zeros((1, self._inter_layer_2.state_size))
        self._motor_layer_state = torch.Tensor([[R / 2, R / 2, 0, 0]])


# Closed loop control network v2
class SNS_Control_closed_loop_v2(nn.Module):
    def __init__(self, inter_layer_1, inter_layer_2, motor_layer):
        super(SNS_Control_closed_loop_v2, self).__init__()
        self._inter_layer_1 = inter_layer_1
        self._inter_layer_2 = inter_layer_2
        self._motor_layer = motor_layer

    def forward(self, object_position, target_position, input):
        _, self._inter_layer_1_state = self._inter_layer_1.forward(input, self._inter_layer_1_state)
        _, self._inter_layer_2_state = self._inter_layer_2.forward(torch.cat((object_position.repeat_interleave(2,
                                                                                                                dim=1)[
                                                                              :, :-1],
                                                                              target_position.repeat_interleave(2,
                                                                                                                dim=1)[
                                                                              :, :-1], self._inter_layer_1_state[:, :2],
                                                                              self._inter_layer_1_state[:, -2:]),
                                                                             dim=1), self._inter_layer_2_state)
        output, self._motor_layer_state = self._motor_layer.forward(torch.cat(
            (self._inter_layer_2_state[:, :-2], self._inter_layer_1_state[:, [-3]], self._inter_layer_2_state[:, -2:]),
            dim=1), self._motor_layer_state)

        # return output.squeeze(dim=0), self._inter_layer_2_state
        return output.squeeze(dim=0)

    def reset(self):
        self._inter_layer_1_state = torch.zeros((1, self._inter_layer_1.state_size))
        self._inter_layer_1_state[0, 0] = R
        self._inter_layer_1_state[0, 1] = -R
        self._inter_layer_2_state = torch.zeros((1, self._inter_layer_2.state_size))
        self._inter_layer_2_state[0, -2:] = R / 2
        self._motor_layer_state = torch.Tensor([[R / 2, R / 2, 0, 0]])


# Modulation control network
class SNS_Control_modulation(nn.Module):
    def __init__(self, inter_layer_1, inter_layer_2, motor_layer):
        super(SNS_Control_modulation, self).__init__()
        self._inter_layer_1 = inter_layer_1
        self._inter_layer_2 = inter_layer_2
        self._motor_layer = motor_layer
        self.reset()

    def forward(self, object_position, target_position, input):
        _, self._inter_layer_1_state = self._inter_layer_1.forward(input, self._inter_layer_1_state)
        _, self._inter_layer_2_state = self._inter_layer_2.forward(torch.cat((object_position.repeat_interleave(2,
                                                                                                                dim=1)[
                                                                              :, :-1],
                                                                              target_position.repeat_interleave(2,
                                                                                                                dim=1)[
                                                                              :, :-1],
                                                                              self._inter_layer_1_state[:, :2]), dim=1),
                                                                   self._inter_layer_2_state)
        output, self._motor_layer_state, self._motor_layer_a, self._motor_layer_buffer = self._motor_layer.forward(
            torch.cat((self._inter_layer_2_state, self._inter_layer_1_state[:, -2:], input[:, 2].reshape(1, -1)),
                      dim=1), self._motor_layer_state, self._motor_layer_a, self._motor_layer_buffer)

        # return output.squeeze(dim=0), self._inter_layer_2_state.squeeze(dim=0), self._inter_layer_1_state.squeeze(dim=0)
        return output.squeeze(dim=0)

    def reset(self):
        self._inter_layer_1_state = torch.zeros((1, self._inter_layer_1.state_size))
        self._inter_layer_1_state[0, 0] = R
        self._inter_layer_1_state[0, 1] = -R
        self._inter_layer_2_state = torch.zeros((1, self._inter_layer_2.state_size))
        self._motor_layer_state = torch.Tensor([[R / 2, R / 2, 0, 0]])
        self._motor_layer_a = torch.zeros((1, self._motor_layer.state_size))
        self._motor_layer_buffer = torch.zeros((1, self._motor_layer.buffer_size, self._motor_layer.state_size))


# Closed loop modulation control
class SNS_Control_closed_loop_modulation(nn.Module):
    def __init__(self, inter_layer_1, inter_layer_2, motor_layer):
        super(SNS_Control_closed_loop_modulation, self).__init__()
        self._inter_layer_1 = inter_layer_1
        self._inter_layer_2 = inter_layer_2
        self._motor_layer = motor_layer

    def forward(self, object_position, target_position, input):
        _, self._inter_layer_1_state = self._inter_layer_1.forward(input, self._inter_layer_1_state)
        _, self._inter_layer_2_state = self._inter_layer_2.forward(torch.cat((object_position.repeat_interleave(2,
                                                                                                                dim=1)[
                                                                              :, :-1],
                                                                              target_position.repeat_interleave(2,
                                                                                                                dim=1)[
                                                                              :, :-1], self._inter_layer_1_state[:, :2],
                                                                              self._inter_layer_1_state[:, -2:]),
                                                                             dim=1), self._inter_layer_2_state)
        output, self._motor_layer_state, self._motor_layer_a, self._motor_layer_buffer = self._motor_layer.forward(
            torch.cat((self._inter_layer_2_state[:, :-2], self._inter_layer_1_state[:, [-3]],
                       self._inter_layer_2_state[:, -2:], input[:, 1].reshape(1, -1)), dim=1), self._motor_layer_state,
            self._motor_layer_a, self._motor_layer_buffer)

        # return output.squeeze(dim=0), self._inter_layer_2_state
        return output.squeeze(dim=0)

    def reset(self):
        self._inter_layer_1_state = torch.zeros((1, self._inter_layer_1.state_size))
        self._inter_layer_1_state[0, 0] = R
        self._inter_layer_1_state[0, 1] = -R
        self._inter_layer_2_state = torch.zeros((1, self._inter_layer_2.state_size))
        self._inter_layer_2_state[0, -2] = R / 2 + R / 16
        self._inter_layer_2_state[0, -1] = R / 2 - R / 16
        self._motor_layer_state = torch.Tensor([[R / 2, R / 2, 0, R / 8, R]])
        self._motor_layer_a = torch.zeros((1, self._motor_layer.state_size))
        self._motor_layer_buffer = torch.zeros((1, self._motor_layer.buffer_size, self._motor_layer.state_size))


#########################################################


# sensory_layer_1
sparsity_mask = np.zeros([SENSORY_LAYER_1_INPUT_SIZE, SENSORY_LAYER_1_SIZE], dtype=np.int32)
for i in range(3):
    sparsity_mask[i, i] = 1
    sparsity_mask[i + 3, i] = -1
for i in range(3, 6):
    sparsity_mask[i - 3, i] = -1
    sparsity_mask[i, i] = 1
for i in range(6, 9):
    sparsity_mask[i - 6, i] = 1
    sparsity_mask[i, i] = -1
for i in range(9, 12):
    sparsity_mask[i - 9, i] = -1
    sparsity_mask[i - 3, i] = 1
tau = torch.Tensor(0.1 * np.ones(SENSORY_LAYER_1_SIZE))
theta_min_in = np.tile(THETA_MIN[:3], 3)
theta_max_in = np.tile(THETA_MAX[:3], 3)
sensory_layer_1 = SNS_layer(layer_input_size=SENSORY_LAYER_1_INPUT_SIZE, layer_size=SENSORY_LAYER_1_SIZE,
                            sparsity_mask=sparsity_mask, theta_min_in=theta_min_in, theta_max_in=theta_max_in, R=R,
                            tau=tau)

# sensory_layer_2
sparsity_mask = np.zeros([SENSORY_LAYER_2_INPUT_SIZE, SENSORY_LAYER_2_SIZE], dtype=np.int32)
sparsity_mask[0:6, 0:2] = 1
sparsity_mask[6:12, 2:4] = 1
sparsity_mask[12:15, 4:8] = 1
tau = torch.Tensor(0.1 * np.ones(SENSORY_LAYER_2_SIZE))
theta_max_in = torch.Tensor(np.concatenate((R * np.ones(SENSORY_LAYER_1_SIZE), F_MAX)))
sensory_layer_2 = SNS_layer(layer_input_size=SENSORY_LAYER_2_INPUT_SIZE, layer_size=SENSORY_LAYER_2_SIZE,
                            sparsity_mask=sparsity_mask, theta_max_in=theta_max_in, R=R)

p = Path(__file__).with_name("sensory_layer_2_param")
sensory_layer_2.load_state_dict(torch.load(p))
sensory_layer_2._params["tau"].data = tau

# command_layer
# neuron 0 move to the pregrasp position
# neuron 1 move to the grasp position
# neuron 2 grasp
# neuron 3 move to the postgrasp position
# neuron 4 move to the prerelease position
# neuron 5 move to the release position
# neuron 6 release
# neuron 7 move to the postrelease position
sparsity_mask = np.zeros([COMMAND_LAYER_INPUT_SIZE, COMMAND_LAYER_SIZE], dtype=np.int32)
# far away from the object position + far away from the target position + no force = move to the pregrasp position
sparsity_mask[[0, 2, 7], 0] = [1, 1, -1]
# not far away from the object position + not very close to the object + no force = move to the grasp position
sparsity_mask[[0, 1, 7], 1] = [-1, 1, -1]
# very close to the object position + no force = grasp
sparsity_mask[[1, 4, 5, 6], 2] = [-1, -1, -1, -1]
# not far away from the object position + force = move to the postgrasp position
sparsity_mask[[0, 7], 3] = [-1, 1]
# far away from the object position + far away from the target position + force = move to the prerelease position
sparsity_mask[[0, 2, 7], 4] = [1, 1, 1]
# not far away from the target position + not very close to the target position + force = move to the release position
sparsity_mask[[2, 3, 7], 5] = [-1, 1, 1]
# very close to the target position + force = release
sparsity_mask[[3, 7], 6] = [-1, 1]
# far away from the target position + no force = move to the postrelease position
sparsity_mask[[2, 7], 7] = -1
tau = torch.Tensor(0.1 * np.ones(COMMAND_LAYER_SIZE))
command_layer = SNS_layer(layer_input_size=COMMAND_LAYER_INPUT_SIZE, layer_size=COMMAND_LAYER_SIZE,
                          sparsity_mask=sparsity_mask, R=R, tau=tau)

p = Path(__file__).with_name("output_mu_param")
command_layer._params["sensory_mu"].data = torch.load(p).data.reshape(-1,1).repeat(1,COMMAND_LAYER_SIZE)

p = Path(__file__).with_name("output_sigma_param")
command_layer._params["sensory_sigma"].data = torch.load(p).data.reshape(-1,1).repeat(1,COMMAND_LAYER_SIZE)


command_layer._params["sensory_mu"].requires_grad = False
command_layer._params["sensory_sigma"].requires_grad = False
command_layer._params["sensory_erev"].data[1, 2] = -3 * R
command_layer._params["b"].data[[0, 2, 4, 5, 7]] = torch.Tensor([-R, 3 * R, -2 * R, -R, R])
command_layer_modulation = copy.deepcopy(command_layer)
command_layer._params["erev"].data[2, 3] = -R
command_layer._params["sparsity_mask"].data[2, 3] = 1

# inter_layer_1
# neuron 0 move to the object position
# neuron 1 move to the target position
# neuron 2 lift the gripper up
# neuron 3 open/close the gripper
sparsity_mask = np.zeros([INTER_LAYER_1_INPUT_SIZE, INTER_LAYER_1_SIZE], dtype=np.int32)
# move to the pregrasp position = move to the object position + lift the gripper up + open the gripper
sparsity_mask[0, [0, 2]] = [1, 1]
# move to the grasp position = move to the object position + open the gripper
sparsity_mask[1, 0] = 1
# grasp the object = move to the object position + close the gripper
sparsity_mask[2, [0, 3]] = [1, 1]
# move to the postgrasp position = move to the object position + lift the gripper up + close the gripper
sparsity_mask[3, [0, 2, 3]] = [1, 1, 1]
# move to the prerelease position = move to the target position + lift the gripper up + close the gripper
sparsity_mask[4, [1, 2, 3]] = [1, 1, 1]
# move to the release position = move to the target position + close the gripper
sparsity_mask[5, [1, 3]] = [1, 1]
# release = move to the target position + open the gripper
sparsity_mask[6, 1] = 1
# move to the postrelease position = move to the target position + lift the gripper up + open the gripper
sparsity_mask[7, [1, 2]] = 1
tau = torch.Tensor(0.1 * np.ones(INTER_LAYER_1_SIZE))
inter_layer_1 = SNS_layer(layer_input_size=INTER_LAYER_1_INPUT_SIZE, layer_size=INTER_LAYER_1_SIZE,
                          sparsity_mask=sparsity_mask, R=R, tau=tau)
inter_layer_1._params["sensory_erev"].data[0, 2] = 10
inter_layer_1._params["sensory_erev"].data[4, 2] = 10

# inter_layer_1 for closed loop controller v_1
sparsity_mask = np.vstack((sparsity_mask, np.zeros((1, sparsity_mask.shape[1]))))
sparsity_mask[-1, -1] = 1
theta_min_in = torch.Tensor(np.concatenate((np.zeros(INTER_LAYER_1_INPUT_SIZE), 0 * np.ones(1))))
inter_layer_1_closed_loop_v1 = SNS_layer(layer_input_size=INTER_LAYER_1_INPUT_SIZE + 1, layer_size=INTER_LAYER_1_SIZE,
                                         sparsity_mask=sparsity_mask, R=R, tau=tau, theta_min_in=theta_min_in)
inter_layer_1_closed_loop_v1._params["sensory_erev"].data[0, 2] = 10
inter_layer_1_closed_loop_v1._params["sensory_erev"].data[4, 2] = 10
inter_layer_1_closed_loop_v1._params["sensory_erev"].data[-1, -1] = -0.72 * R

# inter_layer_1 for closed loop controller v_2
sparsity_mask = np.zeros([INTER_LAYER_1_INPUT_SIZE, INTER_LAYER_1_SIZE + 1], dtype=np.int32)
# move to the pregrasp position = move to the object position + lift the gripper up
sparsity_mask[0, [0, 2]] = [1, 1]
# move to the grasp position = move to the object position + open the gripper
sparsity_mask[1, [0, 4]] = [1, 1]
# grasp the object = move to the object position + close the gripper
sparsity_mask[2, [0, 3]] = [1, 1]
# move to the postgrasp position = move to the object position + lift the gripper up
sparsity_mask[3, [0, 2]] = [1, 1]
# move to the prerelease position = move to the target position + lift the gripper up
sparsity_mask[4, [1, 2]] = [1, 1]
# move to the release position = move to the target position
sparsity_mask[5, 1] = 1
# release = move to the target position + open the gripper
sparsity_mask[6, [1, 4]] = [1, 1]
# move to the postrelease position = move to the target position + lift the gripper up + open the gripper
sparsity_mask[7, [1, 2, 4]] = [1, 1, 1]

tau = torch.Tensor(0.1 * np.ones(INTER_LAYER_1_SIZE + 1))
inter_layer_1_closed_loop_v2 = SNS_layer(layer_input_size=INTER_LAYER_1_INPUT_SIZE, layer_size=INTER_LAYER_1_SIZE + 1,
                                         sparsity_mask=sparsity_mask, R=R, tau=tau)
inter_layer_1_closed_loop_v2._params["sensory_erev"].data[0, 2] = 10
inter_layer_1_closed_loop_v2._params["sensory_erev"].data[4, 2] = 10
inter_layer_1_closed_loop_v2._params["sensory_erev"].data[1, 4] = R / 1.2

# inter_layer_2
# neuron 0/1 object_x(+-)
# neuron 2/3 object_y(+-)
# neuron 4 object_z
# neuron 5/6 target_x(+-)
# neuron 7/8 target_y(+-)
# neuron 9 target_z
sparsity_mask = np.zeros([INTER_LAYER_2_INPUT_SIZE, INTER_LAYER_2_SIZE], dtype=np.int32)
for i in range(2):
    # positive object_x feedback = object_x(+), positive object_y feedback = object_y(+)
    sparsity_mask[2 * i, 4 * i] = 1
    # negative object_x feedback = object_x(-), negative object_y feedback = object_y(-)
    sparsity_mask[2 * i + 1, 4 * i + 1] = 1
    # positive target_x feedback = target_x(+), # positive target_y feedback = target_y(+)
    sparsity_mask[2 * i + 5, 4 * i + 2] = 1
    # negative target_x feedback = target_x(-), # negative target_y feedback = target_y(-)
    sparsity_mask[2 * i + 5 + 1, 4 * i + 2 + 1] = 1
    # move to the object position = target_x(+-), target_y(+-) inhibited
    sparsity_mask[10, [4 * i + 2, 4 * i + 2 + 1]] = -1
    # move to the target position = object_x(+-), object_y(+-) inhibited
    sparsity_mask[11, [4 * i, 4 * i + 1]] = -1
sparsity_mask[4, 8] = 1  # object_z feedback = object_z
sparsity_mask[9, 9] = 1  # target_z feedback = target_z
# move to the object position = target_z inhibited
sparsity_mask[10, 9] = -1
# move to the target position = object_z inhibited
sparsity_mask[11, 8] = -1
tau = torch.Tensor(0.1 * np.ones(INTER_LAYER_2_SIZE))
theta_max_in = np.concatenate((np.stack((THETA_MAX[0:3], THETA_MIN[0:3])).transpose().reshape(1, -1).squeeze()[:-1],
                               np.stack((THETA_MAX[0:3], THETA_MIN[0:3])).transpose().reshape(1, -1).squeeze()[:-1],
                               [R, R]))
theta_max_in = torch.Tensor(theta_max_in)
inter_layer_2 = SNS_layer(layer_input_size=INTER_LAYER_2_INPUT_SIZE, layer_size=INTER_LAYER_2_SIZE,
                          sparsity_mask=sparsity_mask, theta_max_in=theta_max_in, R=R, tau=tau)

# inter_layer_2 for closed loop controller v_2
sparsity_mask = np.hstack((sparsity_mask, np.zeros([sparsity_mask.shape[0], 2], dtype=np.int32)))
sparsity_mask = np.vstack((sparsity_mask, np.zeros([2, sparsity_mask.shape[1]], dtype=np.int32)))
sparsity_mask[12, 10] = 1
sparsity_mask[13, 11] = 1
tau = torch.Tensor(0.1 * np.ones(INTER_LAYER_2_SIZE + 2))
tau.data[-2:] = 1
theta_max_in = np.concatenate((np.stack((THETA_MAX[0:3], THETA_MIN[0:3])).transpose().reshape(1, -1).squeeze()[:-1],
                               np.stack((THETA_MAX[0:3], THETA_MIN[0:3])).transpose().reshape(1, -1).squeeze()[:-1],
                               [R, R, R, R]))
theta_max_in = torch.Tensor(theta_max_in)
inter_layer_2_closed_loop_v2 = SNS_layer(layer_input_size=INTER_LAYER_2_INPUT_SIZE + 2,
                                         layer_size=INTER_LAYER_2_SIZE + 2, sparsity_mask=sparsity_mask,
                                         theta_max_in=theta_max_in, R=R, tau=tau)
inter_layer_2_closed_loop_v2._params["sparsity_mask"].data[-2, -1] = 1
inter_layer_2_closed_loop_v2._params["sparsity_mask"].data[-1, -2] = 1
inter_layer_2_closed_loop_v2._params["erev"].data[-2, -1] = -R
inter_layer_2_closed_loop_v2._params["erev"].data[-1, -2] = -R
inter_layer_2_closed_loop_v2._params["b"].data[-2:] = R

# motor_layer
# neuron 0 x joint command
# neuron 1 y joint command
# neuron 2 z joint command
# neuron 3 jaw joint command
# neuron 4 vertical joint command
sparsity_mask = np.zeros([MOTOR_LAYER_INPUT_SIZE, MOTOR_LAYER_SIZE], dtype=np.int32)
for i in range(
        2):  # object_x = x joint command, object_y = y joint command/target_x = x joint command, target_y = y joint command
    sparsity_mask[[4 * i, 4 * i + 2], i] = 1
    sparsity_mask[[4 * i + 1, 4 * i + 3], i] = -1
# object_z = z joint command/target_z = z joint command
sparsity_mask[8:10, 2] = 1
sparsity_mask[10, 2] = -1  # lift the gripper up = decrease z joint command
# open the gripper = negative left claw joint command + positive right claw joint command
sparsity_mask[11, 3] = 1
theta_min_out = torch.Tensor(THETA_MIN)
theta_max_out = torch.Tensor(THETA_MAX)
tau = torch.Tensor([0.1, 0.1, 0.1, 0.1])
motor_layer = SNS_layer(layer_input_size=MOTOR_LAYER_INPUT_SIZE, layer_size=MOTOR_LAYER_SIZE,
                        sparsity_mask=sparsity_mask, tau=tau, theta_min_out=theta_min_out, theta_max_out=theta_max_out,
                        R=R)
motor_layer._params["sensory_erev"].data[10, 2] = -R / 10

# motor_layer for closed loop controller v_2
sparsity_mask = np.vstack((sparsity_mask, np.zeros([1, sparsity_mask.shape[1]], dtype=np.int32)))
sparsity_mask[12, 3] = -1
motor_layer_closed_loop_v2 = SNS_layer(layer_input_size=MOTOR_LAYER_INPUT_SIZE + 1, layer_size=MOTOR_LAYER_SIZE,
                                       sparsity_mask=sparsity_mask, tau=tau, theta_min_out=theta_min_out,
                                       theta_max_out=theta_max_out, R=R)
motor_layer_closed_loop_v2._params["sensory_erev"].data[10, 2] = -R / 10

# motor_layer for the modulation controller
sparsity_mask[12, :] = 0
sparsity_mask_modulation = np.zeros([MOTOR_LAYER_MODULATION_INPUT_SIZE, MOTOR_LAYER_SIZE], dtype=np.int32)
sparsity_mask_modulation[12, 3] = 1
theta_max_out[-1] *= 0.2
motor_layer_modulation = SNS_layer(layer_input_size=MOTOR_LAYER_MODULATION_INPUT_SIZE, layer_size=MOTOR_LAYER_SIZE,
                                   sparsity_mask=sparsity_mask, sparsity_mask_modulation=sparsity_mask_modulation,
                                   tau=tau, theta_min_out=theta_min_out, theta_max_out=theta_max_out, R=R,
                                   k_io=5 * torch.ones(MOTOR_LAYER_SIZE))
motor_layer_modulation._params["sensory_erev"].data[10, 2] = -R / 10

# # motor_layer for closed loop controller modulation
# sparsity_mask[12, 3] = -1
# sparsity_mask = np.hstack((sparsity_mask, np.zeros([sparsity_mask.shape[0], 1], dtype=np.int32)))
# sparsity_mask = np.vstack((sparsity_mask, np.zeros([1, sparsity_mask.shape[1]], dtype=np.int32)))
# sparsity_mask_modulation = np.zeros([MOTOR_LAYER_MODULATION_INPUT_SIZE + 1, MOTOR_LAYER_SIZE + 1], dtype=np.int32)
# sparsity_mask_modulation[13, 4] = 1
# theta_min_out = torch.cat((torch.Tensor(THETA_MIN), torch.Tensor((0,))), dim=-1)
# theta_max_out = torch.cat((torch.Tensor(THETA_MAX), torch.Tensor((0.125,))), dim=-1)
# tau = torch.Tensor([0.1, 0.1, 0.1, 0.1, 0.1])
# motor_layer_closed_loop_modulation = SNS_layer(layer_input_size=MOTOR_LAYER_MODULATION_INPUT_SIZE + 1,
#                                                layer_size=MOTOR_LAYER_SIZE + 1, sparsity_mask=sparsity_mask,
#                                                sparsity_mask_modulation=sparsity_mask_modulation, tau=tau,
#                                                theta_min_out=theta_min_out, theta_max_out=theta_max_out, R=R,
#                                                tau_io=torch.tensor((2.5,)), k_io=20 * torch.ones(MOTOR_LAYER_SIZE + 1))
# motor_layer_closed_loop_modulation._params["sensory_erev"].data[10, 2] = -R / 10
# motor_layer_closed_loop_modulation._params["b"].data[-1] = R

#########################################################


perceptor = SNS_Perception(sensory_layer_1, sensory_layer_2, command_layer)
perceptor_modulation = SNS_Perception(sensory_layer_1, sensory_layer_2, command_layer_modulation)
controller_open_loop = SNS_Control_open_loop(inter_layer_1, inter_layer_2, motor_layer)
controller_closed_loop_v1 = SNS_Control_closed_loop_v1(inter_layer_1_closed_loop_v1, inter_layer_2, motor_layer)
controller_closed_loop_v2 = SNS_Control_closed_loop_v2(inter_layer_1_closed_loop_v2, inter_layer_2_closed_loop_v2,
                                                       motor_layer_closed_loop_v2)
controller_modulation = SNS_Control_modulation(inter_layer_1, inter_layer_2, motor_layer_modulation)
# controller_closed_loop_modulation = SNS_Control_closed_loop_modulation(inter_layer_1_closed_loop_v2,
#                                                                        inter_layer_2_closed_loop_v2,
#                                                                        motor_layer_closed_loop_modulation)

perceptor.eval()
perceptor_modulation.eval()
controller_open_loop.eval()
controller_closed_loop_v1.eval()
controller_closed_loop_v2.eval()
controller_modulation.eval()
# controller_closed_loop_modulation.eval()


#########################################################


def Create_Open_Loop_Modulation(max_radial_scaling = 0.2):
    theta_max_out[-1] *= max_radial_scaling

    sparsity_mask[12, :] = 0
    sparsity_mask_modulation = np.zeros([MOTOR_LAYER_MODULATION_INPUT_SIZE, MOTOR_LAYER_SIZE], dtype=np.int32)
    sparsity_mask_modulation[12, 3] = 1
    motor_layer_modulation = SNS_layer(layer_input_size=MOTOR_LAYER_MODULATION_INPUT_SIZE, layer_size=MOTOR_LAYER_SIZE,
                                       sparsity_mask=sparsity_mask, sparsity_mask_modulation=sparsity_mask_modulation,
                                       tau=tau, theta_min_out=theta_min_out, theta_max_out=theta_max_out, R=R,
                                       k_io=5 * torch.ones(MOTOR_LAYER_SIZE))
    motor_layer_modulation._params["sensory_erev"].data[10, 2] = -R / 10

    open_loop_controller_modulation = SNS_Control_modulation(inter_layer_1, inter_layer_2, motor_layer_modulation)
    return(open_loop_controller_modulation, perceptor_modulation)

# pick_and_place function runs a pick and place simulation.
def pick_and_place(position_o, position_t, perceptor, controller, PressureValue=2.5, end_time=8, force_threshold_gain=1,
                   inhibitory_gain=1, grasper_closing_speed=1, zero_time_constant=False, mass=1, sizeScaling=0.6):
    """
    Parameters
        ----------
        position_o: [x_o, y_o, z_o]
            Object position.

        position_t: [x_t, y_t, z_t]
            Target position.

        perceptor: SNS_Perception
            The perceptor model that contains the sensory layer 1, sensory layer 2 and command layer.

        controller: SNS_Control_open_loop or SNS_Control_closed_loop_v1 or SNS_Control_closed_loop_v2
            The controller model that contains the inter layer 1, inter layer 2 and motor layer.

        end_time: float (optional)
            The time to stop simulation. Default is 12 sec.

        force_threshold_gain: float
            force_threshold_gain scales the force threshold. The force feedback divided by this parameter is sent to the network so that the force threshold = force_threshold_gain * original force threshold. Default is 1

        inhibitory_gain: float
            inhibitory_gain tunes the weight of the inhibitory synapse from the force sensory neuron to the grasper interneuron. This synapse only exists in instances of SNS_Control_closed_loop_v1. Default is 1

        grasper_closing_speed: float
            grasper_closing_speed determines how fast the grasper closes or opens if an instance of SNS_Control_closed_loop_v2 is selected as the controller model. Default is 1

        zero_time_constant: Boolean
            If true, set the time constants of controller neurons in the jaw loop to zero so that the jaw can respond more quickly.

        mass: float
            The mass of the object. Default is 1 kg.

        sizeScaling: float
            An additional variable to change the scale of the body.  Default is 0.6.
    """

    if isinstance(controller, SNS_Control_closed_loop_v1):
        controller._inter_layer_1._params["sensory_erev"].data[-1, -1] = -inhibitory_gain * R
    if isinstance(controller, SNS_Control_closed_loop_v2) or isinstance(controller, SNS_Control_closed_loop_modulation):
        controller._inter_layer_2._params["tau"].data[-2:] = 1 / grasper_closing_speed
    if zero_time_constant is True:
        controller._inter_layer_1._params["tau"].data[-2:] = 0
        controller._motor_layer._params["tau"].data[3] = 0
    gS = GantrySimulation()  # gantryURDFfile = "URDF//GrasperAndGantry//urdf//GrasperAndGantry.urdf"
    # add object to the simulation at the center of the plate
    gS.addObjectsToSim("PickupCube", startPos=[position_o[0], position_o[1], (0.063 + 0.02)], mass_kg=mass,
                       sizeScaling=sizeScaling, sourceFile=str(
            pathlib.Path.cwd() / "Gantry\\envs\\URDF\\PickUpObject_URDF\\urdf\\PickUpObject_URDF.urdf"))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath("C://Users//Ravesh//BulletPhysics//bullet3//examples//pybullet//gym//pybullet_data")

    neuron_history = []
    sensory_history = []
    command_history = []
    GUI_control = True
    gS.simCounter = 0
    ts = gS.timeStep  # time step of the simulation in seconds
    object_position = torch.Tensor(position_o).unsqueeze(dim=0)
    target_position = torch.Tensor([0, 0, position_o[-1]]).unsqueeze(dim=0)
    # target_position = torch.Tensor(position_t).unsqueeze(dim=0)

    while (not gS.CheckStopSim()):  # check to see if the button was pressed to close the sim
        timeStart = time.perf_counter()
        nsteps = gS.simCounter  # of simulation steps taken so far
        if ts * nsteps > 1:
            target_position = torch.Tensor(position_t).unsqueeze(dim=0)

        x = gS.bulletClient.getJointState(gS.gantryId, gS.GantryLinkIndex_dict["GantryHeadIndex"])[0]
        y = gS.bulletClient.getJointState(gS.gantryId, gS.GantryLinkIndex_dict["BasePositionIndex"])[0]
        z = gS.bulletClient.getJointState(gS.gantryId, gS.GantryLinkIndex_dict["ZAxisBarIndex"])[0]
        gripper_position = torch.Tensor([x, y, z]).unsqueeze(dim=0)
        force_feedback_1 = gS.bulletClient.getContactPoints(gS.gantryId, gS.objects["PickupCube"].objId,
                                                            gS.gantryLinkDict["SJ1"], -1)
        force_feedback_2 = gS.bulletClient.getContactPoints(gS.gantryId, gS.objects["PickupCube"].objId,
                                                            gS.gantryLinkDict["SJ2"], -1)
        force_feedback_3 = gS.bulletClient.getContactPoints(gS.gantryId, gS.objects["PickupCube"].objId,
                                                            gS.gantryLinkDict["SJ3"], -1)
        if len(force_feedback_1) != 0:
            force_1 = np.linalg.norm(sum(np.array([np.array(x[7]) * x[9] for x in force_feedback_1])), 2)
        else:
            force_1 = 0
        if len(force_feedback_2) != 0:
            force_2 = np.linalg.norm(sum(np.array([np.array(x[7]) * x[9] for x in force_feedback_2])), 2)
        else:
            force_2 = 0
        if len(force_feedback_3) != 0:
            force_3 = np.linalg.norm(sum(np.array([np.array(x[7]) * x[9] for x in force_feedback_3])), 2)
        else:
            force_3 = 0
        force = torch.Tensor([force_1, force_2, force_3]).unsqueeze(dim=0) / force_threshold_gain

        commands, force_output = perceptor.forward(gripper_position, object_position, target_position, force)
        [move_to_pre_grasp, move_to_grasp, grasp, lift_after_grasp, move_to_pre_release, move_to_release, release,
         lift_after_release] = commands.squeeze(dim=0).numpy()
        force_sum = force_output.squeeze(dim=0).numpy()
        if isinstance(controller, SNS_Control_closed_loop_v1):
            motor_states = controller.forward(object_position, target_position, commands, force_output)
        else:
            motor_states = controller.forward(object_position, target_position, commands)
        [x_d, y_d, z_d, JawRadialPos] = motor_states.numpy()[:4]
        JawRadialPos = np.maximum(0, JawRadialPos)
        if len(motor_states.numpy()) > 4:
            PressureValue = motor_states.numpy()[-1]
        if lift_after_release > 10:
            object_position = torch.Tensor([0, 0, 0]).unsqueeze(dim=0)

        sensory_history.append([x, y, z, force_1, force_2, force_3, force_sum])
        command_history.append([x_d, y_d, z_d, JawRadialPos, PressureValue])
        neuron_history.append(
            [move_to_pre_grasp, move_to_grasp, grasp, lift_after_grasp, move_to_pre_release, move_to_release, release,
             lift_after_release])

        GrasperArguments = {"frictionCoefficient": 1, "PressureValue": PressureValue, "TargetJawPosition": JawRadialPos,
                            "MaxJawForce": 20, "MaxVel": 0.1, "MaxVertForce": 100, "TargetVertPosition": 0,
                            "MaxVertVel": 0.1}
        ArgumentDict = {"x_gantryHead": x_d, "y_BasePos": y_d, "z_AxisBar": z_d, "x_force": 50, "y_force": 500,
                        "z_force": 500, "GrasperArguments": GrasperArguments}

        # ---------step the simulation----------
        gS.stepSim(usePositionControl=True, GUI_override=True, **ArgumentDict)  # pass argument dict to function
        if ts * nsteps > end_time:
            gS.bulletClient.disconnect()
            break

    return np.array(neuron_history), np.array(sensory_history), np.array(command_history)