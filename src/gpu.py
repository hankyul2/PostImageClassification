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


class GPU:
    def give_me_maximum_gpu(self, gpu_id):
        pass

    def show_gpu_memory(self):
        print(self.remaining_memory)

    def can_use_it(self, gpu_id:int, memory:int):
        assert isinstance(gpu_id, int) and isinstance(memory, int), "gpu_id, memory should be int"
        assert 25 > memory > 0, "memory should be between 1 and 24"
        assert memory*1000 < self.remaining_memory[gpu_id]
        return True

    @property
    def remaining_memory(self):
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,nounits,noheader'
            ]).decode('utf-8')
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map
