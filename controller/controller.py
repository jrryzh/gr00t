import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print("sys add path,", os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random
import numpy as np
import pickle
from PIL import Image as PILImage
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as R
from torch.nn.parallel import DistributedDataParallel as DDP

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset
import torch



class Gr00TController:
    def __init__(self, args):
        device = args.device
        model_path = args.model_path
        action_horizon = args.action_horizon
        
        super().__init__()
        EMBODIMENT_TAG = EmbodimentTag.NEW_EMBODIMENT


        data_config = DATA_CONFIG_MAP["gemanip"]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()


        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=EMBODIMENT_TAG,
            modality_config=modality_config,
            modality_transform=modality_transform,
            device=device,
        )
        
        self.action_horizon = action_horizon
        # self.action_horizon = 1

    def random_seed(self, seed=42, rank=0):
        torch.manual_seed(seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + rank)
            torch.cuda.manual_seed_all(seed + rank)  # if you are using multi-GPU.
        np.random.seed(seed + rank)  # Numpy module.
        random.seed(seed + rank)  # Python random module.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @torch.no_grad()
    def forward(self, obs_dict, timestep=0):
        if timestep % self.action_horizon == 0:
            obs_dict["state.arm"] = obs_dict["state.arm"][None]
            obs_dict["state.gripper"] = obs_dict["state.gripper"][None]
            obs_dict["video.first_view"] = obs_dict["video.first_view"][None]
            obs_dict["video.wrist_view"] = obs_dict["video.wrist_view"][None]
            obs_dict["annotation.human.action.task_description"] = [obs_dict["annotation.human.action.task_description"], ]

            # with torch.cuda.amp.autocast(enabled=True):
            ret_acts = self.policy.get_action(obs_dict)
            self.action_arm_chunk = ret_acts['action.arm']
            self.action_gripper_chunk = ret_acts['action.gripper']
            assert len(self.action_arm_chunk) == 16, "action_horizon is {}".format(len(self.action_arm_chunk))
                
        action_arm = self.action_arm_chunk[timestep % self.action_horizon]
        action_gripper = self.action_gripper_chunk[timestep % self.action_horizon]
        target_joints = action_arm
        target_gripper = action_gripper
        is_terminal = -1.0
        return target_joints, target_gripper, is_terminal


    def reset(self):
        # self.policy.reset()
        pass

import pickle
import struct
import socket
import time

def send_message(send_socket: socket.socket, data: dict):
    serialized_data = pickle.dumps(data)
    message_size = struct.pack("Q", len(serialized_data))
    send_socket.sendall(message_size + serialized_data)

def wait_message(conn: socket.socket):
    data = b""
    payload_size = struct.calcsize("Q")
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    received_data = pickle.loads(frame_data)

    return received_data

def create_send_port_and_wait(port: int):
    serial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serial.bind(("localhost", port))
    serial.listen(1)
    print("Waiting for a connection...")
    conn, addr = serial.accept()
    print("Connected by", addr)
    return conn

def create_receive_port_and_attach(port:int):
    serial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serial.connect(("localhost", port))
    print("connected port ", port)
    return serial

def generate_action(data, agent: Gr00TController, obs_type):
    obs = {}
    # import ipdb; ipdb.set_trace()
    obs["state.arm"] = data["joint_position_state"][:7]
    obs["state.gripper"] = data["joint_position_state"][7:9]
    obs["video.first_view"] = data['camera_data'][obs_type]['rgb']
    obs["video.wrist_view"] = data['camera_data']['realsense']['rgb']
    obs['annotation.human.action.task_description'] = data["instruction"]
    # obs['annotation.human.action.task_description'] = "What is the key object to finish the task: put the banana on the top of the basket. Ouput the bbox to localize the banana."
    timestep = data["timestep"]
    reset = data["reset"]
    if reset:
        agent.reset()
    output, gripper, _ = agent.forward(obs, timestep)
    result = np.array(output)
    return result, gripper


def process_data(data, agent, obs_type):
    # import pdb; pdb.set_trace()
    try:
        processed_data, gripper = generate_action(data, agent, obs_type)
        # print(processed_data, gripper)
        if processed_data is None:
            return {"message": "No action generated!"}
        action = processed_data.tolist() + ([0.4, 0.4] if gripper < 0 else [0.0, 0.0])
        return {"action": action}
    except Exception as e:
        print(str(e))
        return {"error": str(e)}

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # 添加所有需要的参数
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--action_horizon', type=int, default=16)
    parser.add_argument("--receive_port", type=int, default=10000)
    parser.add_argument("--send_port", type=int, default=10001)
    parser.add_argument("--obs_type", type=str, default="obs_camera")

    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    agent = Gr00TController(args=args)
    obs_type = args.obs_type
    send_socket = create_send_port_and_wait(port=args.send_port)
    time.sleep(1)
    receive_socket = create_receive_port_and_attach(port=args.receive_port)
    from tqdm import tqdm
    while True:
        data = wait_message(receive_socket)
        # for _ in tqdm(range(1000)):
        actions = process_data(data, agent, obs_type)
        send_message(send_socket, actions)


