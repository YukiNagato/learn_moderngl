import os

CURRENT_DIR = os.path.realpath(os.path.dirname(__file__))
SHADER_DIR = os.path.join(CURRENT_DIR, 'shaders')
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'data'))
