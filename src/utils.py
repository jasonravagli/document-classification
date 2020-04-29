import os
import configuration as config


def get_path_from_property(config_property):
    return os.path.join(config["paths"]["root"].get(), config["paths"][config_property].get())
