import os
import os.path as osp

PACKAGE_DIR = osp.dirname(osp.abspath(__file__))
BASE_DIR = osp.dirname(PACKAGE_DIR)

# default parent directory for saving experiment results
DEFAULT_DATA_DIR = osp.join(BASE_DIR, 'data')

# creates default data directory if it doesn't exist already
if not osp.exists(DEFAULT_DATA_DIR):
    os.mkdir(DEFAULT_DATA_DIR)
