from assimp.post_process.post_process import PostProcessSteps
import time
from functools import partial

ASSIMP_LOAD_FLAGS = PostProcessSteps.aiProcess_Triangulate | PostProcessSteps.aiProcess_GenSmoothNormals | PostProcessSteps.aiProcess_JoinIdenticalVertices


class TimeMetric:
    def __init__(self):
        self.total_time = 0
        self.cnt = 0

    def update(self, time):
        self.total_time += time
        self.cnt += 1

    def __str__(self):
        if self.cnt == 0:
            average_time = 0
        else:
            average_time = self.total_time / self.cnt
        return f"cnt={self.cnt}, total_time={self.total_time}, average_time={average_time}"


class TimeScope:
    def __init__(self, name, logger: 'TimeLogger'):
        self.name = name
        self.logger = logger
        if name not in self.logger.data:
            self.logger.data[name] = TimeMetric()
        self.start = 0
        self.end = 0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.logger.data[self.name].update(self.end - self.start)


class TimeLogger:
    def __init__(self):
        self.data = {}

    def time_function(self, func, func_name=None):
        if func_name is None:
            func_name = func.__name__
        if func_name not in self.data:
            self.data[func_name] = TimeMetric()

        def func_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            self.data[func_name].update(end_time - start)
            return result
        return func_wrapper

    def time(self, name):
        if isinstance(name, str):
            partial_func = partial(self.time_function, func_name=name)
            return partial_func
        else:
            return self.time_function(name)

    def scope(self, name):
        return TimeScope(name, self)

    def log(self):
        for name, metric in self.data.items():
            print(f'{name}: {metric}')

