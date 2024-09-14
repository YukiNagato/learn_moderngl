from dataclasses import dataclass, field
import numpy as np


@dataclass
class AiTexel:
    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d

    def __ne__(self, other):
        return not self == other

    def ai_color_4d(self):
        return np.array([self.a/255, self.b/255, self.c/255, self.d/255], dtype=np.float32)


@dataclass
class AiTexture:
    width: int = 0
    height: int = 0
    ach_format_hint: str = ""
    ai_texel: AiTexel = field(default_factory=AiTexel)
    filename: str = ""




