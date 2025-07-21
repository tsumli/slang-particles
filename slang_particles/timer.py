import time


class Timer:
    def __init__(self):
        self.frames = 0.0
        self.time = 0.0
        self.fps = 0.0
        self.elapsed = 0.0
        self.last_time = time.time()

    def tick(self):
        now = time.time()
        self.frames += 1
        self.time += now - self.last_time
        self.elapsed += now - self.last_time
        self.last_time = now
        if self.time >= 1:
            self.fps = self.frames / self.time

    def get_fps(self) -> float:
        return self.fps

    def get_elapsed(self) -> float:
        return self.elapsed

    def avg_dt(self) -> float:
        return self.elapsed / self.frames

    def get_frame_count(self) -> int:
        return int(self.frames)
