"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import os

import glob
import tqdm
import lmdb
import pickle
import numpy as np
import cv2
from pathlib import Path
from loguru import logger
import copy
import debugpy 

# debugpy.listen(("0.0.0.0", 10092))  # 监听端口 
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()  # 等待 VS Code 附加


# from lerobot import LEROBOT_HOME
LEROBOT_HOME = "data/lerobot_data"
LEROBOT_HOME = Path(LEROBOT_HOME)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

REPO_NAME = "s2r_3L_banana_basket_2025_none_wot_panda_upGrasp_0.095_0611"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    # "eval_overfitting",
    "render"
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def get_episode_iterator(episode_dir):    
    lmdb_env = lmdb.open(
        f"{episode_dir}/lmdb", 
        readonly=True, 
        lock=False, 
        readahead=False, 
        meminit=False
    )
    meta_info = pickle.load(
        open(
        f"{episode_dir}/meta_info.pkl", 
        "rb"
        )
    )
    arm_index = meta_info["keys"]["scalar_data"].index(b'arm_action')
    arm_key = meta_info["keys"]["scalar_data"][arm_index]
    qpos_index = meta_info["keys"]["scalar_data"].index(b'observation/robot/qpos')
    qpos_key = meta_info["keys"]["scalar_data"][qpos_index]
    gripper_index = meta_info["keys"]["scalar_data"].index(b'gripper_close')
    gripper_key = meta_info["keys"]["scalar_data"][gripper_index]
    primary_index =  meta_info["keys"]["observation/obs_camera/color_image"]
    wrist_index = meta_info["keys"]["observation/realsense/color_image"]
    num_steps = meta_info['num_steps']
    step_id = 0
    with lmdb_env.begin(write=False) as txn:
        # 读取所有数据
        arm_actions = pickle.loads(txn.get(arm_key))
        gripper_actions = pickle.loads(txn.get(gripper_key))
        qpos_data = pickle.loads(txn.get(qpos_key))

        for step_id in range(num_steps):
            step = {}

            arm_action = arm_actions[step_id].tolist()
            gripper_action = gripper_actions[step_id]
            arm_action.append(gripper_action)
            arm_action = np.array(arm_action, dtype=np.float32)
            qpos = qpos_data[step_id]

            primary_data = pickle.loads(txn.get(primary_index[step_id]))
            primary_data = cv2.imdecode(np.frombuffer(primary_data, np.uint8), cv2.IMREAD_COLOR)
            wrist_data = pickle.loads(txn.get(wrist_index[step_id]))
            wrist_data = cv2.imdecode