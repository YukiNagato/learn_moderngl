from enum import Enum, auto
import numpy as np
from dataclasses import dataclass, field
from assimp.math import Vector2f, Vector3f, Quaternion
from typing import List, Tuple
import ctypes
from assimp.types import AiReturn
from collections import namedtuple


def limits(c_int_type):
    signed = c_int_type(-1).value < c_int_type(0).value
    bit_size = ctypes.sizeof(c_int_type) * 8
    signed_limit = 2 ** (bit_size - 1)
    return (-signed_limit, signed_limit - 1) if signed else (0, 2 * signed_limit - 1)


class AiTextureOp(Enum):
    aiTextureOp_Multiply = 0
    aiTextureOp_Add = 1
    aiTextureOp_Subtract = 2
    aiTextureOp_Divide = 3
    aiTextureOp_SmoothAdd = 4
    aiTextureOp_SignedAdd = 5


class AiTextureMapMode(Enum):
    aiTextureMapMode_Wrap = 0
    aiTextureMapMode_Clamp = 1
    aiTextureMapMode_Decal = 2
    aiTextureMapMode_Mirror = 3


class AiTextureMapping(Enum):
    aiTextureMapping_UV = 0
    aiTextureMapping_SPHERE = 1
    aiTextureMapping_CYLINDER = 2
    aiTextureMapping_BOX = 3
    aiTextureMapping_PLANE = 4
    aiTextureMapping_OTHER = 5


class AiTextureType(Enum):
    aiTextureType_NONE = 0
    aiTextureType_DIFFUSE = 1
    aiTextureType_SPECULAR = 2
    aiTextureType_AMBIENT = 3
    aiTextureType_EMISSIVE = 4
    aiTextureType_HEIGHT = 5
    aiTextureType_NORMALS = 6
    aiTextureType_SHININESS = 7
    aiTextureType_OPACITY = 8
    aiTextureType_DISPLACEMENT = 9
    aiTextureType_LIGHTMAP = 10
    aiTextureType_REFLECTION = 11

    aiTextureType_BASE_COLOR = 12
    aiTextureType_NORMAL_CAMERA = 13
    aiTextureType_EMISSION_COLOR = 14
    aiTextureType_METALNESS = 15
    aiTextureType_DIFFUSE_ROUGHNESS = 16
    aiTextureType_AMBIENT_OCCLUSION = 17

    aiTextureType_UNKNOWN = 18

    aiTextureType_SHEEN = 19

    aiTextureType_CLEARCOAT = 20

    aiTextureType_TRANSMISSION = 21

    aiTextureType_MAYA_BASE = 22
    aiTextureType_MAYA_SPECULAR = 23
    aiTextureType_MAYA_SPECULAR_COLOR = 24
    aiTextureType_MAYA_SPECULAR_ROUGHNESS = 25


    @staticmethod
    def type_to_string(texture_type: 'AiTextureType') -> str:
        match texture_type:
            case AiTextureType.aiTextureType_NONE:
                return "n/a"
            case AiTextureType.aiTextureType_DIFFUSE:
                return "Diffuse"
            case AiTextureType.aiTextureType_SPECULAR:
                return "Specular"
            case AiTextureType.aiTextureType_AMBIENT:
                return "Ambient"
            case AiTextureType.aiTextureType_EMISSIVE:
                return "Emissive"
            case AiTextureType.aiTextureType_OPACITY:
                return "Opacity"
            case AiTextureType.aiTextureType_NORMALS:
                return "Normals"
            case AiTextureType.aiTextureType_HEIGHT:
                return "Height"
            case AiTextureType.aiTextureType_SHININESS:
                return "Shininess"
            case AiTextureType.aiTextureType_DISPLACEMENT:
                return "Displacement"
            case AiTextureType.aiTextureType_LIGHTMAP:
                return "Lightmap"
            case AiTextureType.aiTextureType_REFLECTION:
                return "Reflection"
            case AiTextureType.aiTextureType_BASE_COLOR:
                return "BaseColor"
            case AiTextureType.aiTextureType_NORMAL_CAMERA:
                return "NormalCamera"
            case AiTextureType.aiTextureType_EMISSION_COLOR:
                return "EmissionColor"
            case AiTextureType.aiTextureType_METALNESS:
                return "Metalness"
            case AiTextureType.aiTextureType_DIFFUSE_ROUGHNESS:
                return "DiffuseRoughness"
            case AiTextureType.aiTextureType_AMBIENT_OCCLUSION:
                return "AmbientOcclusion"
            case AiTextureType.aiTextureType_SHEEN:
                return "Sheen"
            case AiTextureType.aiTextureType_CLEARCOAT:
                return "Clearcoat"
            case AiTextureType.aiTextureType_TRANSMISSION:
                return "Transmission"
            case AiTextureType.aiTextureType_UNKNOWN:
                return "Unknown"
        return 'BUG'


class AiShadingMode(Enum):
    aiShadingMode_Flat = 0x1
    aiShadingMode_Gouraud = 0x2
    aiShadingMode_Phong = 0x3
    aiShadingMode_Blinn = 0x4
    aiShadingMode_Toon = 0x5
    aiShadingMode_OrenNayar = 0x6
    aiShadingMode_Minnaert = 0x7
    aiShadingMode_CookTorrance = 0x8
    aiShadingMode_NoShading = 0x9
    aiShadingMode_Unlit = aiShadingMode_NoShading
    aiShadingMode_Fresnel = 0xa
    aiShadingMode_PBR_BRDF = 0xb


class AiTextureFlags(Enum):
    aiTextureFlags_Invert = 0x1
    aiTextureFlags_UseAlpha = 0x2
    iTextureFlags_IgnoreAlpha = 0x4


class AiBlendMode(Enum):
    aiBlendMode_Default = 0x1
    aiBlendMode_Additive = 0x2


@dataclass
class AiUVTransform:
    translation: Vector2f = field(default_factory=Vector2f)
    scaling: Vector2f = field(default=Vector2f([1, 0]))
    rotation: float = 0.0


class AiPropertyTypeInfo(Enum):
    aiPTI_Float = 0x1
    aiPTI_Double = 0x2
    aiPTI_String = 0x3
    aiPTI_Integer = 0x4
    aiPTI_Buffer = 0x5


@dataclass
class AiMaterialProperty:
    key: str = ""
    semantic: int = 0
    index: int = 0
    type: AiPropertyTypeInfo = AiPropertyTypeInfo.aiPTI_Float
    data: np.ndarray = field(default=np.array([], dtype=np.uint8))


DefaultNumAllocated = 5

_AI_MATKEY_TEXTURE_BASE       = "$tex.file"
_AI_MATKEY_UVWSRC_BASE        = "$tex.uvwsrc"
_AI_MATKEY_TEXOP_BASE         = "$tex.op"
_AI_MATKEY_MAPPING_BASE       = "$tex.mapping"
_AI_MATKEY_TEXBLEND_BASE      = "$tex.blend"
_AI_MATKEY_MAPPINGMODE_U_BASE = "$tex.mapmodeu"
_AI_MATKEY_MAPPINGMODE_V_BASE = "$tex.mapmodev"
_AI_MATKEY_TEXMAP_AXIS_BASE   = "$tex.mapaxis"
_AI_MATKEY_UVTRANSFORM_BASE   = "$tex.uvtrafo"
_AI_MATKEY_TEXFLAGS_BASE      = "$tex.flags"

UINT_MAX = limits(ctypes.c_uint)[1]

def AI_MATKEY_TEXTURE(type, N):
    return _AI_MATKEY_TEXTURE_BASE, type, N

def AI_MATKEY_MAPPING(type, N):
    return  _AI_MATKEY_MAPPING_BASE, type, N

def AI_MATKEY_MAPPING_DIFFUSE(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_DIFFUSE, N)

def AI_MATKEY_MAPPING_SPECULAR(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_SPECULAR, N)

def AI_MATKEY_MAPPING_AMBIENT(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_AMBIENT, N)

def AI_MATKEY_MAPPING_EMISSIVE(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_EMISSIVE, N)

def AI_MATKEY_MAPPING_NORMALS(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_NORMALS, N)

def AI_MATKEY_MAPPING_HEIGHT(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_HEIGHT, N)

def AI_MATKEY_MAPPING_SHININESS(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_SHININESS, N)

def AI_MATKEY_MAPPING_OPACITY(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_OPACITY, N)

def AI_MATKEY_MAPPING_DISPLACEMENT(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_DISPLACEMENT, N)

def AI_MATKEY_MAPPING_LIGHTMAP(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_LIGHTMAP, N)

def AI_MATKEY_MAPPING_REFLECTION(N):
    return AI_MATKEY_MAPPING(AiTextureType.aiTextureType_REFLECTION, N)

def GET_MATKEY_BASE(prefix):
    match prefix:
        case 'MAPPING':
            return _AI_MATKEY_MAPPING_BASE
        case 'UVWSRC':
            return _AI_MATKEY_UVWSRC_BASE
        case 'TEXTURE':
            return _AI_MATKEY_TEXTURE_BASE
        case 'TEXBLEND':
            return _AI_MATKEY_TEXBLEND_BASE
        case 'TEXOP':
            return _AI_MATKEY_TEXOP_BASE
        case 'MAPPINGMODE_U':
            return _AI_MATKEY_MAPPINGMODE_U_BASE
        case 'MAPPINGMODE_V':
            return _AI_MATKEY_MAPPINGMODE_V_BASE
        case 'TEXFLAGS':
            return _AI_MATKEY_TEXFLAGS_BASE
        case 'TEXTURE':
            return _AI_MATKEY_TEXTURE_BASE

    raise NotImplementedError

def GET_MATKEY(prefix_str, texture_str, N) -> Tuple[str, AiTextureType, int]:
    base = GET_MATKEY_BASE(prefix_str)
    texture_type = getattr(AiTextureType, f'aiTextureType_{texture_str}', None)
    if texture_type is None:
        raise ValueError(f'Unknown texture type {texture_str}')
    return base, texture_type, N


AI_MATKEY_NAME = "?mat.name", 0, 0
AI_MATKEY_TWOSIDED = "$mat.twosided", 0, 0
AI_MATKEY_SHADING_MODEL = "$mat.shadingm", 0, 0
AI_MATKEY_ENABLE_WIREFRAME = "$mat.wireframe", 0, 0
AI_MATKEY_BLEND_FUNC = "$mat.blend", 0, 0
AI_MATKEY_OPACITY = "$mat.opacity", 0, 0
AI_MATKEY_TRANSPARENCYFACTOR = "$mat.transparencyfactor", 0, 0
AI_MATKEY_BUMPSCALING = "$mat.bumpscaling", 0, 0
AI_MATKEY_SHININESS = "$mat.shininess", 0, 0
AI_MATKEY_REFLECTIVITY = "$mat.reflectivity", 0, 0
AI_MATKEY_SHININESS_STRENGTH = "$mat.shinpercent", 0, 0
AI_MATKEY_REFRACTI = "$mat.refracti", 0, 0
AI_MATKEY_COLOR_DIFFUSE = "$clr.diffuse", 0, 0
AI_MATKEY_COLOR_AMBIENT = "$clr.ambient", 0, 0
AI_MATKEY_COLOR_SPECULAR = "$clr.specular", 0, 0
AI_MATKEY_COLOR_EMISSIVE = "$clr.emissive", 0, 0
AI_MATKEY_COLOR_TRANSPARENT = "$clr.transparent", 0, 0
AI_MATKEY_COLOR_REFLECTIVE = "$clr.reflective", 0, 0
AI_MATKEY_GLOBAL_BACKGROUND_IMAGE = "?bg.global", 0, 0
AI_MATKEY_GLOBAL_SHADERLANG = "?sh.lang", 0, 0
AI_MATKEY_SHADER_VERTEX = "?sh.vs", 0, 0
AI_MATKEY_SHADER_FRAGMENT = "?sh.fs", 0, 0
AI_MATKEY_SHADER_GEO = "?sh.gs", 0, 0
AI_MATKEY_SHADER_TESSELATION = "?sh.ts", 0, 0
AI_MATKEY_SHADER_PRIMITIVE = "?sh.ps", 0, 0
AI_MATKEY_SHADER_COMPUTE = "?sh.cs", 0, 0


@dataclass
class MaterialTextureResult:
    path: str = None
    mapping: AiTextureMapping = None
    uv_index: int = None
    blend: float = None
    op: AiTextureType = None
    map_mode: List[AiTextureMapMode] = field(default_factory=list)
    flags: int = None


class AiMaterial:
    def __init__(self):
        self.properties: List[AiMaterialProperty] = []

    def get(self, key: str, type: int, idx: int, p_max=None):
        pass

    def add(self, data, key: str, type: int, index: int):
        dst_property_idx = None
        for property_idx, property in enumerate(self.properties):
            if property.key == key and property.index == index:
                dst_property_idx = property_idx

        p_type = None
        if isinstance(data, np.ndarray):
            match data.dtype:
                case np.float32:
                    p_type = AiPropertyTypeInfo.aiPTI_Float
                case np.int32:
                    p_type = AiPropertyTypeInfo.aiPTI_Integer
                case np.float64:
                    p_type = AiPropertyTypeInfo.aiPTI_Double
                case np.uint8:
                    p_type = AiPropertyTypeInfo.aiPTI_Buffer
            if p_type is None:
                raise ValueError(f'Unknown type {data.dtype}')
            data = data.view(np.uint8)

        elif isinstance(data, str):
            p_type = AiPropertyTypeInfo.aiPTI_String
            data = np.fromstring(data, np.uint8)
        else:
            raise NotImplementedError

        new_property = AiMaterialProperty(
            key=key,
            semantic=type,
            index=index,
            type=p_type,
            data=data,
        )

        if dst_property_idx is not None:
            self.properties[dst_property_idx] = new_property
        else:
            self.properties.append(new_property)






        new_property = AiMaterialProperty(
            key=key, semantic=self
        )

    def get_texture_count(self, type: AiTextureType):
        return get_material_texture_count(self, type)

    def get_texture(
                         self,
                         type: AiTextureType,
                         index: int,
                         uv_index: bool = False,
                         blend: bool = False,
                         op: bool = False,
                         map_mode: bool = False)->MaterialTextureResult:
        material_texture_result = MaterialTextureResult()
        get_material_texture(
            material_texture_result,
            self,
            type=type,
            index=index,
            uv_index=uv_index,
            blend=blend,
            op=op,
            map_mode=map_mode,
            flags=False)
        return material_texture_result



    @property
    def num_properties(self) -> int:
        return len(self.properties)


def get_material_texture_count(p_mat: AiMaterial, type: AiTextureType):
    max_ = 0
    for i in range(p_mat.num_properties):
        prop = p_mat.properties[i]
        if prop.key == _AI_MATKEY_TEXTURE_BASE and AiTextureType(prop.semantic) == type:
            max_ = max(max_, prop.index+1)
    return max_


def get_material_property(p_mat: AiMaterial, key: str, type: int, index: int):
    for i in range(p_mat.num_properties):
        prop = p_mat.properties[i]
        if (type == UINT_MAX or prop.semantic == type) and \
                prop.key == key and \
                (index == UINT_MAX or prop.index == index):
            return prop
    return None


def get_material_array(p_mat: AiMaterial, key: str, type: int, index: int, p_max: int=None):
    prop = get_material_property(p_mat, key, type, index)
    if prop is None:
        return None, p_max

    i_write = max(1, len(prop.data))
    if p_max is not None:
        i_write = min(p_max, i_write)

    return prop.data[:i_write], i_write


def get_material_integer(p_mat: AiMaterial, key: str, type: int, index: int):
    ret = get_material_array(p_mat, key, type, index, 0)
    if ret is None:
        return None
    return ret[0]


def get_material_color(p_mat: AiMaterial, key: str, type: int, index: int):
    i_max = 4
    array, i_max = get_material_array(p_mat, key, type, index, i_max)
    if array is None:
        return None

    if i_max == 3:
        array[3] = 1
    return array


def get_material_uv_transform(p_mat: AiMaterial, key: str, type: int, index: int):
    i_max = 5
    return get_material_array(p_mat, key, type, index, i_max)[0]


def get_material_string(p_mat: AiMaterial, key: str, type: int, index: int):
    prop = get_material_property(p_mat, key, type, index)
    if prop is None:
        return None

    if prop.type == AiPropertyTypeInfo.aiPTI_String:
        return prop.data.tobytes().decode('utf-8')
    else:
        print(f"Material property {key} was found, but is no string")
        return None


def get_material_texture(
                         material_texture_result:MaterialTextureResult,
                         mat: AiMaterial,
                         type: AiTextureType,
                         index: int,
                         uv_index: bool,
                         blend: bool,
                         op: bool,
                         map_mode: bool,
                         flags: bool):

    path = get_material_string(mat, *AI_MATKEY_TEXTURE(type, index))
    mapping = AiTextureMapping.aiTextureMapping_UV.value
    mapping_: int = get_material_integer(mat, *AI_MATKEY_MAPPING(type, index))
    if mapping_ is not None:
        mapping = mapping_
    mapping = AiTextureMapping(mapping)

    material_texture_result.path = path
    material_texture_result.mapping = mapping_

    if uv_index and mapping == AiTextureMapping.aiTextureMapping_UV:
        material_texture_result.uv_index = get_material_integer(mat, GET_MATKEY_BASE('UVWSRC'), type.value, index)

    if blend is not None:
        material_texture_result.blend = get_material_integer(mat, GET_MATKEY_BASE('TEXBLEND'), type.value, index)

    if op:
        material_texture_result.op = get_material_integer(mat, GET_MATKEY_BASE('TEXOP'), type.value, index)

    if map_mode:
        u = get_material_integer(mat, GET_MATKEY_BASE('MAPPINGMODE_U'), type.value(), index)
        v = get_material_integer(mat, GET_MATKEY_BASE('MAPPINGMODE_V'), type.value(), index)

        material_texture_result.map_mode[0] = AiTextureMapMode(u)
        material_texture_result.map_mode[1] = AiTextureMapMode(v)

    if flags:
        material_texture_result.flags = get_material_integer(mat, GET_MATKEY_BASE('TEXFLAGS'), type.value(), index)

    return material_texture_result

