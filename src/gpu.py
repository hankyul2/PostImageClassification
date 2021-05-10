#!/usr/bin/env python
# encoding: utf-8

import subprocess
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class GPU:

    def give_me_maximum_gpu(self, gpu_id):
        pass

    def show_gpu_memory(self):
        print(self.remaining_memory)

    def can_use_it(self, gpu_id: int, memory: int):
        assert isinstance(gpu_id, int) and isinstance(memory, int), "gpu_id, memory should be int"
        assert 25 > memory > 0, "memory should be between 1 and 24"
        memory_in_gb = memory * 1000
        if memory_in_gb < self.remaining_memory[gpu_id]:
            return True
        else:
            return False

    def get_device(self, gpu_id: int = -1):
        if torch.cuda.is_available():
            if gpu_id == -1:
                raise Exception("not implemented yet")
            else:
                if self.can_use_it(gpu_id, 20):
                    device = "cuda:" + str(gpu_id)
                    torch.backends.cudnn.benchmark = True
                else:
                    raise Exception("please use other devices")
        else:
            device = "cpu"
        return device

    @property
    def remaining_memory(self):
        return self.nividia_query('free')

    @property
    def used_memory(self):
        return self.nividia_query('used')

    @property
    def total_memory(self):
        return self.nividia_query('total')

    def nividia_query(self, query_type: str) -> Dict:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.{}'.format(query_type),
                '--format=csv,nounits,noheader'
            ]).decode('utf-8')
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map
