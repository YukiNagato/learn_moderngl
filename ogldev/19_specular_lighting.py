import moderngl
import pygame
import numpy as np
import glm
from ogldev.lib.ogldev_pipeline import Pipeline
from ogldev.lib.ogldev_math_3d import PersProjInfo
from ogldev.lib.ogldev_camera import Camera
from PIL import Image
from ogldev.lib.ogldev_light import DirectionalLight
from ogldev.lib.ogldev_math_3d import Vector3f, Vector2f, Vector
from ogldev.lib.ogldev_callbacks import ICallbacks
from ogldev.lib.pygame_backend import PyGameBackend
from typing import List

HEIGHT = 800
WIDTH = 800


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


class Scene(ICallbacks):
    def __init__(self):
        self.ctx = moderngl.get_context()
        self.program = self.ctx.program(
            vertex_shader='''
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
            ''',
            fragment_shader='''
                #version 330                                                                        
                                                                                                    
                in vec2 TexCoord0;                                                                  
                in vec3 Normal0;                                                                    
                in vec3 WorldPos0;                                                                  
                                                                                                    
                out vec4 FragColor;                                                                 
                                                                                                    
                struct DirectionalLight                                                             
                {                                                                                   
                    vec3 Color;                                                                     
                    float AmbientIntensity;                                                         
                    float DiffuseIntensity;                                                         
                    vec3 Direction;                                                                 
                };                                                                                  
                                                                                                    
                uniform DirectionalLight gDirectionalLight;                                         
                uniform sampler2D gSampler;                                                         
                uniform vec3 gEyeWorldPos;                                                          
                uniform float gMatSpecularIntensity;                                                
                uniform float gSpecularPower;                                                       
                                                                                                    
                void main()                                                                         
                {                                                                                   
                    vec4 AmbientColor = vec4(gDirectionalLight.Color * gDirectionalLight.AmbientIntensity, 1.0f);
                    vec3 LightDirection = -gDirectionalLight.Direction;                             
                    vec3 Normal = normalize(Normal0);                                               
                                                                                                    
                    float DiffuseFactor = dot(Normal, LightDirection);                              
                                                                                                    
                    vec4 DiffuseColor  = vec4(0, 0, 0, 0);                                          
                    vec4 SpecularColor = vec4(0, 0, 0, 0);                                          
                                                                                                    
                    if (DiffuseFactor > 0) {                                                        
                        DiffuseColor = vec4(gDirectionalLight.Color * gDirectionalLight.DiffuseIntensity * DiffuseFactor, 1.0f);
                                                                                                    
                        vec3 VertexToEye = normalize(gEyeWorldPos - WorldPos0);                     
                        vec3 LightReflect = normalize(reflect(gDirectionalLight.Direction, Normal));
                        float SpecularFactor = dot(VertexToEye, LightReflect);                      
                        if (SpecularFactor > 0) {                                                   
                            SpecularFactor = pow(SpecularFactor, gSpecularPower);
                            SpecularColor = vec4(gDirectionalLight.Color * gMatSpecularIntensity * SpecularFactor, 1.0f);
                        }                                                                           
                    }                                                                               
                                                                                                    
                    FragColor = texture2D(gSampler, TexCoord0.xy) *                                 
                                (AmbientColor + DiffuseColor + SpecularColor);                      
                }


            ''',
        )

        vertices = [
            Vertex((-1.0, -1.0, 0.5773), (0.0, 0.0)),
            Vertex((0.0, -1.0, -1.15475), (0.5, 0.0)),
            Vertex((1.0, -1.0, 0.5773), (1.0, 0.0)),
            Vertex((0.0, 1.0, 0.0), (0.5, 1.0)),
        ]
        self.vertices = vertices
        indices = np.array([0, 3, 1, 1, 3, 2, 2, 3, 0, 0, 1, 2], dtype='i4')
        calculate_normals(vertices, indices)
        vertices = np.ascontiguousarray(np.stack(vertices).flatten())

        self.scale = 0.0
        self.delta = 0.1
        self.pipline = Pipeline()
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(indices)
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f 2f 3f', 'Position', 'TexCoord', 'Normal')],
                                         self.ibo)
        self.pers_proj_info = PersProjInfo(
            fov=60.0,
            height=HEIGHT,
            width=WIDTH,
            z_near=1.0,
            z_far=100.0
        )
        self.camera = Camera(800, 800,
                             pos=(0.0, 0.0, -3.0),
                             target=(0.0, 0.0, 1.0),
                             up=(0.0, 1.0, 0.0))
        self.texture = ImageTexture('ogldev/data/test.png')
        self.direction_light = DirectionalLight(
            color=Vector3f([1.0, 1.0, 1.0]),
            ambient_intensity=0.0,
            diffuse_intensity=0.2,
            direction=Vector3f([0.0, 0.0, 1.0])
        )

    def set_directional_light(self, light: DirectionalLight):
        self.program['gDirectionalLight.Color'].write(light.color)
        self.program['gDirectionalLight.AmbientIntensity'] = light.ambient_intensity
        direction = light.direction.copy()
        direction.normalize_()
        self.program['gDirectionalLight.Direction'].write(direction)
        self.program['gDirectionalLight.DiffuseIntensity'] = light.diffuse_intensity

    def render_scene_callback(self):
        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.texture.use()
        self.camera.on_render()
        self.scale += self.delta
        self.pipline.set_world_pos((0.0, 0.0, 5.0))
        self.pipline.set_rotate((0.0, self.scale, 0.0))
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

        self.program['gMatSpecularIntensity'] = 1.0
        self.program['gSpecularPower'] = 32

        self.vao.render()

        # vertices = self.vertices
        # points = np.array(vertices)[:, :3]
        # points = np.concatenate(
        #     [
        #         points,
        #         np.ones_like(points[:, :1])
        #     ],
        #     1
        # )
        #
        # wvp = self.pipline.get_wvp_trans()
        #
        # trans_points = points @ wvp.transpose()
        # trans_points = trans_points[:, :3] / trans_points[:, -1:]
        # print(trans_points)

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
