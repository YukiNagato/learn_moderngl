class BaseImporter:
    def get_extension_list(self):
        raise NotImplementedError

    def intern_read_file(self, filename):
        raise NotImplementedError