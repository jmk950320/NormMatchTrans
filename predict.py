import math
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DistributedSampler
import time
from pathlib import Path
import os
from datetime import timedelta

from data.data_loader_multigraph import GMDataset, get_dataloader
import eval
from model import NMT

from utils.config import cfg
from utils.utils import update_params_from_cmdline



if __name__ == "__main__":
    cfg = update_params_from_cmdline(default_params=cfg)

    model = NMT()

    device = 'cuda'

    model = model.to(device)

  #  model.eval()

    model_parms = torch.load("params.pt", map_location='cuda:0')
        # 2. 'module.' 제거
    new_state_dict = {}
    for k, v in model_parms.items():
        if k.startswith("module."):
            new_key = k[7:]  # 'module.' 길이 7
        else:
            new_key = k
        new_state_dict[new_key] = v

    # 3. 모델에 로드
    model.load_state_dict(new_state_dict)

    dataset_len = 1000*1 #cfg.EVAL.SAMPLES * world_size}
    x = 'test'
    ###############
    # x: GMDataset(x, cfg.DATASET_NAME, sets=x, length=dataset_len[x], obj_resize=(384, 384)) for x in ("train", "test")
    dataset = GMDataset(x, 'test', sets=x, length=dataset_len, obj_resize=(384, 384))

    print('len(dataset): ', len(dataset))

  


    