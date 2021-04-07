#!/usr/bin/env python
# encoding: utf-8

import subprocess
from dataclasses import dataclass

import GPUtil
from threading import Thread
import time

class Monitor(Thread):
    """
    10초 단위로 gpu 사용량을 알려주는 쓰레드 생성
    monitor = Monitor(10)
    """
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

@dataclass
class GPUChecker:
    gpu_id: int

    def check_gpu_by_id(self, id, memory):
        gpu_used = get_gpu_memory_map()
        assert gpu_used[id] >= memory
        return gpu_

    def get_gpu_memory_map(self):
        """Get the current gpu usage.

        Returns
        -------
        usage: dict
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,nounits,noheader'
            ]).decode('utf-8')
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map


