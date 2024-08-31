import numpy as np
from dataclasses import dataclass
from ogldev.lib.ogldev_math_3d import Vector3f


@dataclass
class BaseLight:
    name: str = ""
    color: Vector3f = Vector3f()
    ambient_intensity: float = 0
    diffuse_intensity: float = 0

    def __str__(self):
        return " ".join([
            f"name={self.name}",
            f"color={self.color}",
            f"ambient_intensity={self.ambient_intensity}",
            f"diffuse_intensity={self.diffuse_intensity}",
        ])


@dataclass
class DirectionalLight(BaseLight):
    direction: Vector3f = Vector3f()

    def __str__(self):
        return " ".join([
            super().__str__(),
            f"direction={self.direction}",
        ])
