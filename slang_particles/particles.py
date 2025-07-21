import numpy as np
from dataclasses import dataclass
import numpy.typing as npt


@dataclass
class Particle:
    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def pack(self):
        return np.array(
            [
                self.mass,
                self.position[0],
                self.position[1],
                self.position[2],
                self.velocity[0],
                self.velocity[1],
                self.velocity[2],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def unpack(data: npt.NDArray[np.int8]) -> "Particle":
        data_float = data.view(np.float32)

        return Particle(
            mass=data_float[0],
            position=data_float[1:4],
            velocity=data_float[4:7],
        )
