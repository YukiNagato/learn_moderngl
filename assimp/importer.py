from assimp.post_process.post_process import PostProcessSteps, BaseProcess
from enum import Enum, IntFlag, auto
from typing import List
from assimp.base_importer import BaseImporter
from assimp.asset_lib import get_importer_instances
from assimp.post_process import get_post_processing_step_instance_list


class Importer:
    def __init__(self):
        self.importers: List[BaseImporter] = get_importer_instances()
        self.post_processing_steps: List[BaseProcess] = get_post_processing_step_instance_list()

    def get_importer(self, file):
        for importer in self.importers:
            extensions = importer.get_extension_list()
            for extension in extensions:
                if file.endswith(extension):
                    return importer
        return None

    def read_file(self, file, flags: IntFlag):
        importer = self.get_importer(file)
        if importer is None:
            raise IOError(f'No importer found for {file}')

        scene = importer.intern_read_file(file)
        self.apply_post_processing(scene, flags)
        return scene

    def apply_post_processing(self, scene, flags: IntFlag):
        for post_processing_step in self.post_processing_steps:
            if post_processing_step.is_active(flags):
                post_processing_step.execute(scene)






