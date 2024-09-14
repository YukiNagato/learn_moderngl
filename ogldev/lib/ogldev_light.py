import numpy as np
from ogldev.lib.ogldev_math_3d import Vector3f
from dataclasses import dataclass, asdict, field


@dataclass
class BaseLight:
    name: str = ""
    color: Vector3f = field(default_factory=Vector3f)
    ambient_intensity: float = 0
    diffuse_intensity: float = 0

    def __str__(self):
        return "".join([f"{name}={value}" for name, value in asdict(self).items()])

@dataclass
class DirectionalLight(BaseLight):
    direction: Vector3f = field(default_factory=Vector3f)


@dataclass
class LightAttenuation:
    constant: float = 1.0
    linear: float = 0.0
    exp: float = 0.0


@dataclass
class PointLight(BaseLight):
    position: Vector3f = field(default_factory=Vector3f)
    attenuation: LightAttenuation = LightAttenuation()


@dataclass
class SpotLight(PointLight):
    direction: Vector3f = field(default_factory=Vector3f)
    cutoff: float = 0.0


COLOR_WHITE = Vector3f([1.0, 1.0, 1.0])
COLOR_RED = Vector3f([1.0, 0.0, 0.0])
COLOR_GREEN = Vector3f([0.0, 1.0, 0.0])
COLOR_CYAN = Vector3f([0.0, 1.0, 1.0])
COLOR_BLUE = Vector3f([0.0, 0.0, 1.0])