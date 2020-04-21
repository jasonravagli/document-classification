import confuse
import logging
import sys

# Read configuration file
config = confuse.Configuration('document-classification')
config.set_file("../config_default.yml")

# Configure logging
# Log both on file and console
fh = logging.FileHandler('log.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[fh, ch])


def add_command_line_args(args):
    config.set_args(args)


def set_application_root_path(root_path: str):
    config["paths"]["root"] = root_path


def set_log_level(log_level):
    logging.getLogger().setLevel(log_level)
