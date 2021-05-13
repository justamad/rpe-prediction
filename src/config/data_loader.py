from os.path import join

import os


class LoadingException(Exception):
    pass


class RPELoader(object):

    def __init__(self, root_path):
        if not os.path.isfile(join(root_path, "rpe.json")):
            raise LoadingException("JSON file does not exists!")


class AzureLoader(object):

    def __init__(self, root_path, cam_type="sub"):
        if not os.path.exists(join(root_path, "azure")):
            raise LoadingException("Azure file not present.")


data_loaders = [RPELoader, AzureLoader]


class DataLoader(object):

    def __init__(self, root_path):
        """
        Create instance of DataLoader for given root path
        @param root_path: current path to subject folder
        """
        self.file_loaders = []
        for loader in data_loaders:
            try:
                current_loader = loader(root_path)
                self.file_loaders.append(current_loader)
            except LoadingException:
                print("Could not load file loader.")

        self._sets = 9

    def iterate_over_sets(self):
        for current_set in range(self._sets):
            yield current_set
