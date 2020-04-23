import confuse
import logging
import sys
import os

# Read configuration file
config = confuse.Configuration('document-classification')
config.set_file("config_default.yml")

def add_command_line_args(args):
    config.set_args(args)


def set_application_root_path(root_path: str):
    config["paths"]["root"] = root_path


def configure_logger(log_level):
    # Log both on file and console
    fh = logging.FileHandler(os.path.join(config["paths"]["root"], config["paths"]["log_file"]))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logging.basicConfig(level=log_level, handlers=[fh, ch])
