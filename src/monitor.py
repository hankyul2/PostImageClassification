import time
from threading import Thread

import GPUtil

from src.gpu import GPU


class Monitor(Thread):
    """
    10초 단위로 gpu 사용량을 알려주는 쓰레드 생성
    monitor = Monitor(10)
    """

    def __init__(self, delay: int, gpu_id: int):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.gpu_id = gpu_id
        self.start()
        self.max_gpu_memory_used = 0
        self.gpu = GPU()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            self.max_gpu_memory_used = max(self.gpu.used_memory[self.gpu_id], self.max_gpu_memory_used)
            print("MAX gpu {} memory used:", self.max_gpu_memory_used)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

    @property
    def max_memory_used(self):
        return self.max_gpu_memory_used
