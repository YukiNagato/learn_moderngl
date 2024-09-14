import moderngl


class Technique:
    def __init__(self):
        pass

    def init(self):
        raise NotImplementedError

    def init_from_str(self, vertex_shader=None, fragment_shader=None):
        self.ctx = moderngl.get_context()
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

    def init_from_files(self, vs_file, fs_file):
        def load_file(file):
            if file is None:
                return None
            with open(file, 'r') as f:
                return f.read()

        vs = load_file(vs_file)
        fs = load_file(fs_file)
        return self.init_from_str(vs, fs)

