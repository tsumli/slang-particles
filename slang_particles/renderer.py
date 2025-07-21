import slangpy as spy
import enum
import numpy as np
import numpy.typing as npt

from slang_particles.shader import SHADER_DIR


class PrepareFrame:

    class Strategy(enum.Enum):
        CLEAR = enum.auto()
        ATTENUATE = enum.auto()
        STORE = enum.auto()

    def __init__(self, device: spy.Device, strategy: Strategy):
        super().__init__()
        self.device = device
        self.strategy = strategy

        if self.strategy != self.Strategy.STORE:
            self.program = self.device.load_program(
                str(SHADER_DIR / "prepare_frame.slang"),
                [self._get_entry_point(strategy)],
            )
            self.kernel = self.device.create_compute_kernel(self.program)

    def _get_entry_point(self, strategy: Strategy):
        if strategy == self.Strategy.CLEAR:
            return "clear_main"
        elif strategy == self.Strategy.ATTENUATE:
            return "attenuate_main"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        output: spy.Texture,
    ):
        if self.strategy == self.Strategy.STORE:
            return

        self.kernel.dispatch(
            thread_count=[output.width, output.height, 1],
            vars={
                "output": output,
            },
            command_encoder=command_encoder,
        )


class ParticleRenderer:
    def __init__(self, device: spy.Device):
        super().__init__()
        self.device = device
        self.program = self.device.load_program(
            str(SHADER_DIR / "particle_renderer.slang"), ["compute_main"]
        )
        self.kernel = self.device.create_compute_kernel(self.program)
        self.camera_buffer = self.device.create_buffer(
            element_count=1,
            struct_type=self.kernel.reflection.camera.position,
            usage=spy.BufferUsage.shader_resource,
            label="camera",
        )

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        output: spy.Texture,
        particles: spy.Buffer,
        world_view_projection: spy.Buffer,
        position: npt.NDArray[np.float32],
    ):
        self.camera_buffer.copy_from_numpy(position)
        num_particles = particles.size // particles.struct_size
        self.kernel.dispatch(
            thread_count=[num_particles, 1, 1],
            vars={
                "output": output,
                "particles": particles,
                "camera": {
                    "world_view_projection": world_view_projection,
                    "position": self.camera_buffer,
                },
            },
            command_encoder=command_encoder,
        )
