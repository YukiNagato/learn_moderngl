import math

import moderngl
import pygame
import numpy as np
import glm
from ogldev.lib.ogldev_pipeline import Pipeline
from ogldev.lib.ogldev_math_3d import PersProjInfo
from ogldev.lib.ogldev_camera import Camera
from PIL import Image
from ogldev.lib.ogldev_light import DirectionalLight, PointLight, SpotLight
from ogldev.lib.ogldev_math_3d import Vector3f, Vector2f, Vector
from ogldev.lib.ogldev_callbacks import ICallbacks
from ogldev.lib.pygame_backend import PyGameBackend
from typing import List

HEIGHT = 800
WIDTH = 800
FieldDepth = 20.0
FieldWidth = 10.0


class ImageTexture:
    def __init__(self, path):
        self.ctx = moderngl.get_context()

        img = Image.open(path).convert('RGBA')
        self.texture = self.ctx.texture(img.size, 4, img.tobytes())
        self.sampler = self.ctx.sampler(texture=self.texture)

    def use(self):
        self.sampler.use()


class Vertex(np.ndarray):
    def __new__(cls, pos=None, tex=None, normal=None):
        values = np.zeros(8)
        if pos is not None:
            values[:3] = np.array(pos)
        if tex is not None:
            values[3:5] = np.array(tex)
        if normal is not None:
            values[5:8] = np.array(normal)

        obj = np.array(values, dtype=np.float32).view(cls)
        return obj

    @property
    def pos(self):
        return self[:3].view(Vector3f)

    @property
    def tex(self):
        return self[3:5].view(Vector3f)

    @property
    def normal(self):
        return self[5:8].view(Vector3f)

    @normal.setter
    def normal(self, value):
        self[5:8] = value


def calculate_normals(vertices: List[Vertex], indices):
    for i in range(0, len(indices), 3):
        idx_0 = indices[i]
        idx_1 = indices[i + 1]
        idx_2 = indices[i + 2]

        v1 = vertices[idx_1].pos - vertices[idx_0].pos
        v2 = vertices[idx_2].pos - vertices[idx_0].pos
        normal = v1.cross(v2)
        normal.normalize_()
        vertices[idx_0].normal += normal
        vertices[idx_1].normal += normal
        vertices[idx_2].normal += normal

    for i in range(len(vertices)):
        vertices[i].normal.normalize_()


vertex_shader = '''
#version 330                                                                        

layout (location = 0) in vec3 Position;                                             
layout (location = 1) in vec2 TexCoord;                                             
layout (location = 2) in vec3 Normal;                                               

uniform mat4 gWVP;                                                                  
uniform mat4 gWorld;                                                                

out vec2 TexCoord0;                                                                 
out vec3 Normal0;                                                                   
out vec3 WorldPos0;                                                                 

void main()                                                                         
{                                                                                   
    gl_Position = gWVP * vec4(Position, 1.0);                                       
    TexCoord0   = TexCoord;                                                         
    Normal0     = (gWorld * vec4(Normal, 0.0)).xyz;                                 
    WorldPos0   = (gWorld * vec4(Position, 1.0)).xyz;                               
}
'''

fragment_shader = '''
#version 330                                                                        

const int MAX_POINT_LIGHTS = 2;
const int MAX_SPOT_LIGHTS = 2;

in vec2 TexCoord0;
in vec3 Normal0;
in vec3 WorldPos0;

out vec4 FragColor;

struct BaseLight
{
    vec3 Color;
    float AmbientIntensity;
    float DiffuseIntensity;
};

struct DirectionalLight
{
    BaseLight Base;
    vec3 Direction;
};

struct Attenuation
{
    float Constant;
    float Linear;
    float Exp;
};

struct PointLight
{
    BaseLight Base;
    vec3 Position;
    Attenuation Atten;
};

struct SpotLight
{
    PointLight Base;
    vec3 Direction;
    float Cutoff;
};

uniform int gNumPointLights;
uniform int gNumSpotLights;
uniform DirectionalLight gDirectionalLight;
uniform PointLight gPointLights[MAX_POINT_LIGHTS];
uniform SpotLight gSpotLights[MAX_SPOT_LIGHTS];
uniform sampler2D gSampler;
uniform vec3 gEyeWorldPos;
uniform float gMatSpecularIntensity;
uniform float gSpecularPower;

vec4 CalcLightInternal(BaseLight Light, vec3 LightDirection, vec3 Normal)                   
{                                     
    vec4 AmbientColor = vec4(Light.Color * Light.AmbientIntensity, 1.0f);
    float DiffuseFactor = dot(Normal, -LightDirection);

    vec4 DiffuseColor = vec4(0, 0, 0, 0);
    vec4 SpecularColor = vec4(0, 0, 0, 0);

    if (DiffuseFactor > 0) {
        DiffuseColor = vec4(Light.Color * Light.DiffuseIntensity * DiffuseFactor, 1.0f);

        vec3 VertexToEye = normalize(gEyeWorldPos - WorldPos0);
        vec3 LightReflect = normalize(reflect(LightDirection, Normal));
        float SpecularFactor = dot(VertexToEye, LightReflect);
        if (SpecularFactor > 0) {
            SpecularFactor = pow(SpecularFactor, gSpecularPower);
            SpecularColor = vec4(Light.Color * gMatSpecularIntensity * SpecularFactor, 1.0f);
        }                                                                                   
    }                                                                                       
                                                                                            
    return (AmbientColor + DiffuseColor + SpecularColor);                                   
}                                                                                           
                                                                                            
vec4 CalcDirectionalLight(vec3 Normal)                                                      
{                                                                                           
    return CalcLightInternal(gDirectionalLight.Base, gDirectionalLight.Direction, Normal);  
}                                                                                           
                                                                                            
vec4 CalcPointLight(PointLight l, vec3 Normal)                                              
{                                                                                           
    vec3 LightDirection = WorldPos0 - l.Position;                                           
    float Distance = length(LightDirection);                                                
    LightDirection = normalize(LightDirection);                                             
                                                                                            
    vec4 Color = CalcLightInternal(l.Base, LightDirection, Normal);                         
    float AttenuationFactor =  l.Atten.Constant +                                                 
                         l.Atten.Linear * Distance +                                        
                         l.Atten.Exp * Distance * Distance;                                 
                                                                                            
    return Color / AttenuationFactor;                                                             
}                                                                                           
                                                                                            
vec4 CalcSpotLight(SpotLight l, vec3 Normal)                                                
{                                                                                           
    vec3 LightToPixel = normalize(WorldPos0 - l.Base.Position);
    float SpotFactor = dot(LightToPixel, l.Direction);

    if (SpotFactor > l.Cutoff) {                                                            
        vec4 Color = CalcPointLight(l.Base, Normal);
        return Color * (1.0 - (1.0 - SpotFactor) * 1.0/(1.0 - l.Cutoff));
    }                                                                                       
    else {                                                                                  
        return vec4(0,0,0,0);
    }                                                                                       
}                                                                                           

void main()                                                                                 
{                                                                                           
    vec3 Normal = normalize(Normal0);
    vec4 TotalLight = CalcDirectionalLight(Normal);

    for (int i = 0 ;i < gNumPointLights ;i++) {                                           
        TotalLight += CalcPointLight(gPointLights[i], Normal);
    }                                                                                       

    for (int i = 0 ;i < gNumSpotLights ;i++) {                                            
        TotalLight += CalcSpotLight(gSpotLights[i], Normal);
    }                                                                                       

    FragColor = texture2D(gSampler, TexCoord0.xy) * TotalLight;
}

'''


class Scene(ICallbacks):
    def __init__(self):
        self.ctx = moderngl.get_context()
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

        normal = np.array([0.0, 1.0, 0.0])
        vertices = [
            Vertex((0.0, 0.0, 0.0), (0.0, 0.0), normal),
            Vertex((0.0, 0.0, FieldDepth), (0.0, 1.0), normal),
            Vertex((FieldWidth, 0.0, 0.0), (1.0, 0.0), normal),
            Vertex((FieldWidth, 0.0, 0.0), (1.0, 0.0), normal),
            Vertex((0.0, 0.0, FieldDepth), (0.0, 1.0), normal),
            Vertex((FieldWidth, 0.0, FieldDepth), (1.0, 1.0), normal),
        ]

        self.vertices = vertices

        vertices = np.ascontiguousarray(np.stack(vertices).flatten())

        self.scale = 0.0
        self.delta = 0.0057
        self.pipline = Pipeline()
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f 2f 3f', 'Position', 'TexCoord', 'Normal')])
        self.pers_proj_info = PersProjInfo(
            fov=60.0,
            height=HEIGHT,
            width=WIDTH,
            z_near=1.0,
            z_far=50.0
        )
        self.camera = Camera(WIDTH, HEIGHT,
                             pos=(5.0, 1.0, -3.0),
                             target=(0.0, 0.0, 1.0),
                             up=(0.0, 1.0, 0.0))
        self.texture = ImageTexture('ogldev/data/test.png')
        self.direction_light = DirectionalLight(
            color=Vector3f([1.0, 1.0, 1.0]),
            ambient_intensity=0.0,
            diffuse_intensity=0.1,
            direction=Vector3f([1.0, -1.0, 0.0])
        )

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

    def render_scene_callback(self):
        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.texture.use()
        self.camera.on_render()
        self.scale += self.delta

        pl = [PointLight() for _ in range(2)]
        pl[0].diffuse_intensity = 0.5
        pl[0].color = Vector3f([1.0, 0.5, 0.0])
        pl[0].position = Vector3f([3.0, 1.0, FieldDepth * (np.cos(self.scale) + 1.0) / 2.0])
        pl[0].attenuation.linear = 0.1

        pl[1].diffuse_intensity = 0.5
        pl[1].color = Vector3f([0.0, 0.5, 1.0])
        pl[1].position = Vector3f([7.0, 1.0, FieldDepth * (np.sin(self.scale) + 1.0) / 2.0])
        pl[1].attenuation.linear = 0.1

        sl = [SpotLight() for _ in range(2)]
        sl[0].diffuse_intensity = 0.9
        sl[0].color = Vector3f([0.0, 1.0, 1.0])
        sl[0].position = Vector3f(self.camera.pos)
        sl[0].direction = Vector3f(self.camera.target)
        sl[0].attenuation.linear = 0.1
        sl[0].cutoff = 10.0

        sl[1].diffuse_intensity = 0.9
        sl[1].color = Vector3f([1.0, 1.0, 1.0])
        sl[1].position = Vector3f([5.0, 3.0, 10.0])
        sl[1].direction = Vector3f([0.0, -1.0, 0.0])
        sl[1].attenuation.linear = 0.1
        sl[1].cutoff = 20.0

        self.set_point_lights(pl)
        self.set_spot_lights(sl)

        self.pipline.set_world_pos((0.0, 0.0, 1.0))
        self.pipline.set_camera(
            **self.camera.get_camera_dict()
        )
        self.pipline.set_perspective_projection(self.pers_proj_info)
        g_wvp = self.pipline.get_wvp_trans()
        g_wvp = glm.mat4(np.ascontiguousarray(g_wvp))
        self.program['gWVP'].write(g_wvp)

        g_world = self.pipline.get_world_trans()
        g_world = glm.mat4(np.ascontiguousarray(g_world))
        self.program['gWorld'].write(g_world)

        self.set_directional_light(self.direction_light)
        self.program['gEyeWorldPos'].write(self.camera.pos.astype('float32'))

        self.program['gMatSpecularIntensity'] = 0.0
        self.program['gSpecularPower'] = 0

        self.vao.render()

    def keyboard_callback(self, key, *args, **kwargs):
        if key == pygame.K_a:
            self.direction_light.ambient_intensity += 0.05
        elif key == pygame.K_s:
            self.direction_light.ambient_intensity -= 0.05
        elif key == pygame.K_z:
            self.direction_light.diffuse_intensity += 0.05
        elif key == pygame.K_x:
            self.direction_light.diffuse_intensity -= 0.05

    def passive_mouse_callback(self, x, y):
        self.camera.on_mouse(x, y)


backend = PyGameBackend(height=HEIGHT, width=WIDTH)
backend.run(Scene())
