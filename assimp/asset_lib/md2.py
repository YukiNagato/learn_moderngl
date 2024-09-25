import struct
from pprint import pprint
from dataclasses import dataclass, field
from typing import List
from assimp.scene import AiScene, AiNode, AiMesh, AiMaterial
from assimp.mesh import AiPrimitiveState, AiFace
from assimp.light import AiLight
import assimp.material as am
import numpy as np
from assimp.color import AiColor3D, AiColor4D
from assimp.math import Vector3f, Vector4f
from assimp.asset_lib.md2_normal_table import anorms
from typing import ClassVar
from assimp.base_importer import BaseImporter


@dataclass
class Header:
    magic : int = 0
    version : int = 0
    skin_width : int = 0
    skin_height : int = 0
    frame_size : int = 0
    num_skins : int = 0
    num_vertices : int = 0
    num_tex_coords : int = 0
    num_triangles : int = 0
    num_gl_commands : int = 0
    num_frames : int = 0
    offset_skins : int = 0
    offset_tex_coords : int = 0
    offset_triangles : int = 0
    offset_frames : int = 0
    offset_gl_commands : int = 0
    offset_end : int = 0

    @classmethod
    def read(cls, bytes: memoryview):
        format = "<4s16i"
        size = struct.calcsize(format)
        header_data = bytes[:size]
        header_struct = struct.unpack(format, header_data)
        return Header(*header_struct)

    def is_valid(self):
        if self.version != 8 :
            return False
        if self.magic != b'IDP2':
            return False
        if self.num_frames != 1:
            return False
        return True


class Vertex:
    def __init__(self, x, y, z, light_normal_index):
        self.vertex = np.array([x, y, z], dtype=np.float32)
        self.light_normal_index = light_normal_index

    @classmethod
    def read(cls, buffer: memoryview):
        format = '<4B'
        tri_vertex_struct = struct.unpack(format, buffer[:4])
        return cls(*tri_vertex_struct)


@dataclass
class Frame:
    scale: np.ndarray = field(default=np.zeros(3, dtype=np.float32))
    translate: np.ndarray = field(default=np.zeros(3, dtype=np.float32))
    name: str = ""
    vertices: List[Vertex] = field(default_factory=list)

    @classmethod
    def read(cls, buffer: memoryview, vertex_num: int):
        format = '<6f16s'
        size = struct.calcsize(format)
        frame_data = buffer[:size]
        frame_struct = struct.unpack(format, frame_data)
        frame = cls(
            scale=np.array(frame_struct[:3], dtype=np.float32),
            translate=np.array(frame_struct[3:6], dtype=np.float32),
            name=frame_struct[6].split(b'\00')[0].decode('ascii'),
        )
        buffer = buffer[size:]
        for i in range(vertex_num):
            frame.vertices.append(Vertex.read(buffer))
            buffer = buffer[4:]
        return frame


class Skins:
    @classmethod
    def read(cls, buffer, num_skins)->List[str]:
        format = '<64s'
        size = struct.calcsize(format)
        skins = []
        for i in range(num_skins):
            data = buffer[:size]
            skin = struct.unpack(format, data)[0]
            skin = skin.split(b'\x00')[0].decode('ascii')
            skins.append(skin)
            buffer = buffer[64:]
        return skins


class Triangle:
    format = '<6h'
    size = struct.calcsize(format)
    def __init__(self, vertex_0, vertex_1, vertex_2, texture_0, texture_1, texture_2):
        self.vertex_indices = [vertex_0, vertex_1, vertex_2]
        self.texture_indices = [texture_0, texture_1, texture_2]

    @classmethod
    def read(cls, buffer: memoryview):
        triangle_data = buffer[:cls.size]
        triangle_struct = struct.unpack(cls.format, triangle_data)
        return cls(*triangle_struct)


class Triangles:
    @classmethod
    def read(cls, buffer: memoryview, num_triangles):
        triangles = []
        for i in range(num_triangles):
            tri = Triangle.read(buffer)
            buffer = buffer[Triangle.size:]
            triangles.append(tri)
        return triangles


@dataclass
class TexCoord:
    format: ClassVar = '<2h'
    size: ClassVar = struct.calcsize(format)

    s: int = 0
    t: int = 0

    @classmethod
    def read(cls, buffer: memoryview):
        st_vertex_data = buffer[:cls.size]
        st_vertex_struct = struct.unpack(cls.format, st_vertex_data)
        return cls(*st_vertex_struct)


class TexCoords:
    @classmethod
    def read(cls, buffer: memoryview, num_tex_coords):
        tex_coords = []
        for i in range(num_tex_coords):
            tex_coord = TexCoord.read(buffer)
            buffer = buffer[TexCoord.size:]
            tex_coords.append(tex_coord)
        return tex_coords


class Md2(BaseImporter):
    def read(self, file_path)->AiScene:
        fp = open(file_path, 'rb')
        buffers = fp.read()
        buffers_view = memoryview(buffers)
        header = Header.read(buffers_view)
        pprint(header)
        assert header.is_valid()
        scene = AiScene()
        scene.root_node = AiNode()
        scene.root_node.meshes = [0]
        scene.materials = [AiMaterial()]
        mesh = AiMesh()
        scene.meshes = [mesh]
        mesh.primitive_types.triangle = True
        mesh.faces = [AiFace() for _ in range(header.num_triangles)]
        mesh.vertices = [Vector3f() for _ in range(header.num_triangles * 3)]
        mesh.normals = [Vector3f() for _ in range(header.num_triangles * 3)]

        frames_buffer_view = buffers_view[header.offset_frames:]
        frame = Frame.read(frames_buffer_view, header.num_vertices)

        triangles_buffer_view = buffers_view[header.offset_triangles:]
        triangles = Triangles.read(triangles_buffer_view, header.num_triangles)

        tex_coords_buffer_view = buffers_view[header.offset_tex_coords:]
        tex_coords = TexCoords.read(tex_coords_buffer_view, header.num_tex_coords)

        skin_buffer_view = buffers_view[header.offset_skins:]
        skins = Skins.read(skin_buffer_view, header.num_skins)

        material = scene.materials[0]
        i_mode = am.AiShadingMode.aiShadingMode_Gouraud.value
        material.add(np.array([i_mode], dtype=np.int32), *am.AI_MATKEY_SHADING_MODEL)
        if header.num_tex_coords > 0 and header.num_skins > 0:
            color = AiColor3D([1, 1, 1])
            material.add(color, *am.AI_MATKEY_COLOR_DIFFUSE)
            material.add(color, *am.AI_MATKEY_COLOR_SPECULAR)
            material.add(AiColor3D([0.05, 0.05, 0.05]), *am.AI_MATKEY_COLOR_AMBIENT)
            if len(skins) > 0:
                material.add(skins[0], *am.GET_MATKEY('TEXTURE', 'DIFFUSE', 0))
        else:
            color = AiColor3D([0.6, 0.6, 0.6])
            material.add(color, *am.AI_MATKEY_COLOR_DIFFUSE)
            material.add(color, *am.AI_MATKEY_COLOR_SPECULAR)
            material.add(AiColor3D([0.05, 0.05, 0.05]), *am.AI_MATKEY_COLOR_AMBIENT)
            material.add('DefaultMaterial', *am.AI_MATKEY_NAME)
            material.add('$texture_dummy.bmp', *am.GET_MATKEY('TEXTURE', 'DIFFUSE', 0))

        # now read all triangles of the first frame, apply scaling and translation
        i_current = 0
        divisor_u = 1.0
        divisor_v = 1.0

        if header.num_tex_coords > 0:
            mesh.num_uv_components = [2]
            mesh.texture_coords = [[Vector3f() for _ in range(header.num_triangles * 3)]]
            if not header.skin_width or not header.skin_height:
                raise Exception('MD2: No valid skin width or height given')

            else:
                divisor_u = float(header.skin_width)
                divisor_v = float(header.skin_height)

        for i in range(header.num_triangles):
            scene.meshes[0].faces[i].indices = [0, 0, 0]

            for c in range(3):
                i_index = triangles[i].vertex_indices[c]
                if i_index > header.num_vertices:
                    raise Exception('MD2: Triangle index out of range')

                vec = frame.vertices[i_index].vertex * frame.scale + frame.translate
                mesh.vertices[i_current] = vec.view(Vector3f)
                light_normal_index = frame.vertices[i_index].light_normal_index
                assert light_normal_index < len(anorms)
                v_normal = anorms[light_normal_index]
                mesh.normals[i_current] = v_normal.view(Vector3f)

                if header.num_tex_coords:
                    i_index = triangles[i].texture_indices[c]
                    if i_index > header.num_triangles:
                        raise Exception('MD2: Texture index out of range')

                    tc = mesh.texture_coords[0][i_current]
                    tc.x = tex_coords[i_index].s / divisor_u
                    tc.y = 1-tex_coords[i_index].t / divisor_v
                scene.meshes[0].faces[i].indices[c] = i_current
                i_current += 1

            scene.meshes[0].faces[i].indices[0], scene.meshes[0].faces[i].indices[2] = \
                scene.meshes[0].faces[i].indices[2], scene.meshes[0].faces[i].indices[0]

            scene.root_node.transformation = np.array(
                [
                    1, 0, 0, 0,
                    0, 0, 1, 0,
                    0, -1, 0, 0,
                    0, 0, 0, 1
                ],
                dtype=np.float32
            ).reshape((4, 4))
        return scene

    def intern_read_file(self, filename):
        return self.read(filename)

    def get_extension_list(self):
        return ['.md2']


if __name__ == '__main__':
    file_path = "D:/work/ogldev/Content/phoenix_ugv.md2"
    md2 = Md2()
    scene = md2.read(file_path)







