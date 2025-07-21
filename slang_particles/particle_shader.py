from pathlib import Path

import slangpy as spy
import numpy as np

from slang_particles.particles import Particle


class ParticleShader:
    def __init__(
        self,
        device: spy.Device,
        shader_path: Path,
        initial_particles: list[Particle],
        thread_count: tuple[int, int, int],
    ):
        super().__init__()
        self.device = device
        self.program = self.device.load_program(str(shader_path), ["compute_main"])
        self.kernel = self.device.create_compute_kernel(self.program)
        self.buffer_dt = self.device.create_buffer(
            element_count=1,
            struct_type=self.kernel.reflection.frame_info.dt,
            usage=spy.BufferUsage.shader_resource,
            label="dt",
        )
        self.particle_buffer = self.device.create_buffer(
            element_count=len(initial_particles),
            struct_type=self.kernel.reflection.particles,
            usage=spy.BufferUsage.unordered_access,
            label="particles",
            data=np.array([p.pack() for p in initial_particles]),
        )
        self.thread_count = thread_count

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        dt: float,
    ):
        self.buffer_dt.copy_from_numpy(np.array([dt], dtype=np.float32))

        self.kernel.dispatch(
            thread_count=self.thread_count,
            vars={
                "particles": self.particle_buffer,
                "frame_info": {
                    "dt": self.buffer_dt,
                },
            },
            command_encoder=command_encoder,
        )
