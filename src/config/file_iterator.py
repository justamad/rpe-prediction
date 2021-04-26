from os.path import join
import os


class FileIterator(object):

    def __init__(self, input_path, train_percentage: float = 0.7):
        self._input_path = input_path
        self._folders = os.listdir(self._input_path)
        print(f"Found the following folders: {self._folders}")

    def iterate_over_files(self):
        for subject in self._folders:
            pass


if __name__ == '__main__':
    iterator = FileIterator("../../data")
