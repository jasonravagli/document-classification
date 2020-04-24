import os
import argparse
import logging
import configuration as config

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Document Classification')
    parser.add_argument("--generate_dataset", action="store_true", help="Generate the dataset to be used from the CNN training process")
    parser.add_argument("--train_network", action="store_true", help="Train the CNN")
    parser.add_argument("--resume_training", action="store_true", help="Resume the trainining of the CNN from the last checkpoint")
    args = parser.parse_args()

    config.add_command_line_args(args)
    config.set_application_root_path(os.path.abspath("../"))

    config.configure_logger(logging.DEBUG)

    import training_vgg16
