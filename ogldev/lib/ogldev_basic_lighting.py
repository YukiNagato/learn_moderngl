import os
import math
import numpy as np
from ogldev.lib.ogldev_math_3d import Vector, Vector3f
from ogldev.lib.ogldev_technique import Technique
from ogldev.lib import SHADER_DIR
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from typing import List
from ogldev.lib.ogldev_light import DirectionalLight, PointLight, SpotLight
import glm


MAX_POINT_LIGHTS = 2
MAX_SPOT_LIGHTS  = 2


@dataclass
class UniformBaseLight:
    Color: str = ""
    AmbientIntensity: str = ""
    DiffuseIntensity: str = ""


@dataclass
class UniformDirectionalLight:
    Base: UniformBaseLight = field(default_factory=UniformBaseLight)
    Direction: str = ""


@dataclass
class UniformAttenuation:
    Constant: str = ""
    Linear: str = ""
    Exp: str = ""


@dataclass
class UniformPointLight:
    Base: UniformBaseLight = field(default_factory=UniformBaseLight)
    Position: str = ""
    Atten: UniformAttenuation = field(default_factory=UniformAttenuation)


@dataclass
class UniformSpotLight:
    Base: UniformBaseLight = field(default_factory=UniformBaseLight)
    Direction: str = ""
    Cutoff: str = ""


@dataclass
class Uniforms:
    gNumPointLights: str = 'gNumPointLights'
    gNumSpotLights: str = 'gNumSpotLights'
    gDirectionalLight: UniformDirectionalLight= field(default_factory=UniformDirectionalLight)
    gPointLights: List[UniformPointLight] = field(default_factory=lambda: [UniformPointLight() for _ in range(MAX_POINT_LIGHTS)])
    gSpotLights: List[UniformSpotLight] = field(default_factory=lambda: [UniformSpotLight() for _ in range(MAX_SPOT_LIGHTS)])
    gEyeWorldPos: str = 'gEyeWorldPos'
    gMatSpecularIntensity: str = 'gMatSpecularIntensity'
    gSpecularPower: str = 'gSpecularPower'
    gColorMod: str = 'gColorMod'
    gWVP: str = 'gWVP'
    gWorld: str = 'gWorld'

    def __post_init__(self):
        prefix = ""

        def set_name(object, prefix):
            if isinstance(object, list):
                for idx in range(len(object)):
                    value = object[idx]
                    if isinstance(value, str) and not value:
                        object[idx] = prefix + f'[{idx}]'
                    else:
                        set_name(object[idx], prefix + f'[{idx}].')
            elif is_dataclass(object):
                for f in fields(object):
                    name = f.name
                    value = getattr(object, f.name)

                    if isinstance(value, str):
                        if not value:
                            setattr(object, f.name, prefix + name)
                    elif is_dataclass(value):
                        set_name(value, prefix=prefix+name+'.')
                    elif isinstance(value, list):
                        set_name(value, prefix=prefix + name)
                    else:
                        raise NotImplementedError(value)
            else:
                raise NotImplementedError

        set_name(self, prefix)

    def get_lines(self):
        lines = []

        def object_to_lines(object, lines):
            if isinstance(object, str):
                lines.append(object)
            elif isinstance(object, list):
                for o in object:
                    object_to_lines(o, lines)
            elif is_dataclass(object):
                for field in fields(object):
                    value = getattr(object, field.name)
                    object_to_lines(value, lines)
            else:
                raise NotImplementedError

        object_to_lines(self, lines)
        return lines

    def check(self):
        lines = self.get_lines()
        for line in lines:
            cmd = f'self.{line}'
            value = eval(cmd)
            assert value == line

    def __str__(self):
        return '\n'.join(self.get_lines())


class BasicLightingTechnique(Technique):
    def init(self):
        vs_file = os.path.join(SHADER_DIR, 'basic_lighting.vs')
        fs_file = os.path.join(SHADER_DIR, 'basic_lighting.fs')
        self.init_from_files(
            vs_file=vs_file,
            fs_file=fs_file,
        )
        self.uniforms = Uniforms()
        self.program[self.uniforms.gColorMod].write(np.array([1, 1, 1, 1], dtype=np.float32))

    def set_wvp(self, matrix):
        glm_matrix = glm.mat4(np.ascontiguousarray(matrix))
        self.program[self.uniforms.gWVP].write(glm_matrix)

    def set_world_matrix(self, matrix):
        matrix = glm.mat4(np.ascontiguousarray(matrix))
        self.program[self.uniforms.gWorld].write(matrix)

    def set_directional_light(self, light: DirectionalLight):
        self.program['gDirectionalLight.Base.Color'].write(light.color)
        self.program['gDirectionalLight.Base.AmbientIntensity'] = light.ambient_intensity
        direction = light.direction.copy()
        direction.normalize_()
        self.program['gDirectionalLight.Direction'].write(direction)
        self.program['gDirectionalLight.Base.DiffuseIntensity'] = light.diffuse_intensity

    def set_point_lights(self, pl_list: List[PointLight]):
        self.program['gNumPointLights'] = len(pl_list)
        for idx in range(len(pl_list)):
            self.program[f'gPointLights[{idx}].Base.Color'].write(pl_list[idx].color)
            self.program[f'gPointLights[{idx}].Base.AmbientIntensity'] = pl_list[idx].ambient_intensity
            self.program[f'gPointLights[{idx}].Position'].write(pl_list[idx].position)
            self.program[f'gPointLights[{idx}].Base.DiffuseIntensity'] = pl_list[idx].diffuse_intensity
            self.program[f'gPointLights[{idx}].Atten.Constant'] = pl_list[idx].attenuation.constant
            self.program[f'gPointLights[{idx}].Atten.Linear'] = pl_list[idx].attenuation.linear
            self.program[f'gPointLights[{idx}].Atten.Exp'] = pl_list[idx].attenuation.exp

    def set_spot_lights(self, sl_list: List[SpotLight]):
        self.program['gNumSpotLights'] = len(sl_list)

        for idx in range(len(sl_list)):
            self.program[f'gSpotLights[{idx}].Base.Base.Color'].write(sl_list[idx].color)
            self.program[f'gSpotLights[{idx}].Base.Base.AmbientIntensity'] = sl_list[idx].ambient_intensity
            self.program[f'gSpotLights[{idx}].Base.Base.DiffuseIntensity'] = sl_list[idx].diffuse_intensity
            self.program[f'gSpotLights[{idx}].Base.Position'].write(sl_list[idx].position)

            direction = sl_list[idx].direction.copy()
            direction.normalize_()
            self.program[f'gSpotLights[{idx}].Direction'].write(direction)
            self.program[f'gSpotLights[{idx}].Cutoff'] = math.cos(sl_list[idx].cutoff)
            self.program[f'gSpotLights[{idx}].Base.Atten.Constant'] = sl_list[idx].attenuation.constant
            self.program[f'gSpotLights[{idx}].Base.Atten.Linear'] = sl_list[idx].attenuation.linear
            self.program[f'gSpotLights[{idx}].Base.Atten.Exp'] = sl_list[idx].attenuation.exp

    def set_eye_world_pos(self, pos: np.ndarray):
        self.program[self.uniforms.gEyeWorldPos].write(pos)

    def set_mat_specular_intensity(self, intensity: float):
        self.program[self.uniforms.gMatSpecularIntensity] = intensity

    def set_mat_specular_power(self, power: float):
        self.program[self.uniforms.gSpecularPower] = power

    def set_color_mod(self, color_mod: np.ndarray):
        self.program[self.uniforms.gColorMod].write(color_mod)


if __name__ == '__main__':
    uniforms = Uniforms()
    uniforms.check()

