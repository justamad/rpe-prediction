from os.path import exists

import os


def create_folder_if_not_already_exists(path: str):
    if not exists(path):
        os.makedirs(path)
