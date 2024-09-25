from .join_identical_vertices import JoinVerticesProcess
from .triangulate_process import TriangulateProcess
from .gen_vertex_normals_process import GenVertexNormalsProcess


def get_post_processing_step_instance_list():
    return [
        TriangulateProcess(),
        GenVertexNormalsProcess(),
        JoinVerticesProcess(),
    ]
