from typing import List


class ListPointer:
    def __init__(self, data: List):
        self.offset = 0
        self.data = data

    def get(self):
        return self.data[self.offset]

    def __iadd__(self, other: int):
        self.offset += other

    def rinc(self):
        obj = self.data[self.offset]
        self.offset += 1
        return obj

    def new(self):
        new_obj = ListPointer(self.data)
        new_obj.offset = self.offset
        return new_obj
