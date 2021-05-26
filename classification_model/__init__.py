import logging

from classification_model.config.core import config, PACKAGE_ROOT
from classification_model.config import logging_config

VERSION_PATH = PACKAGE_ROOT / 'VERSION'

# logger
logger = logging.getLogger(config.app_config.PACKAGE_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False

# Create version variable
with open(VERSION_PATH,'r') as version_file:
    __version__ = version_file.read().strip()