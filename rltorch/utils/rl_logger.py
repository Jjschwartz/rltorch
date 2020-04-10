import time
import os.path as osp
from prettytable import PrettyTable

import rltorch.utils.file_utils as fu
from rltorch.user_config import DEFAULT_DATA_DIR


RESULTS_FILE_NAME = "results"
RESULTS_FILE_EXT = "tsv"
CONFIG_FILE_NAME = "config"
CONFIG_FILE_EXT = "yaml"
CONFIG_FILE = f"{CONFIG_FILE_NAME}.{CONFIG_FILE_EXT}"
RESULTS_FILE = f"{RESULTS_FILE_NAME}.{RESULTS_FILE_EXT}"


class RLLogger:

    def __init__(self, env_name, alg=None):
        self.env_name = env_name
        self.alg = alg
        self.setup_save_file()
        self.headers = []
        self.log_buffer = dict()
        self.headers_written = False

    def setup_save_file(self):
        ts = time.strftime("%Y%m%d-%H%M")
        self.save_dir = osp.join(DEFAULT_DATA_DIR,
                                 f"{self.alg}_{self.env_name}_{ts}")
        fu.make_dir(self.save_dir)
        self.save_file_path = fu.generate_file_path(self.save_dir,
                                                    RESULTS_FILE_NAME,
                                                    RESULTS_FILE_EXT)

    def get_save_path(self, filename=None, ext=None):
        if filename is None:
            ts = time.strftime("%Y%m%d-%H%M")
            filename = f"{self.alg}_{self.env_name}_{ts}"
        return fu.generate_file_path(self.save_dir, filename, ext)

    def save_config(self, cfg):
        cfg_file = self.get_save_path(CONFIG_FILE_NAME, CONFIG_FILE_EXT)
        fu.write_yaml(cfg_file, cfg)

    def add_header(self, header):
        assert header not in self.headers
        self.headers.append(header)

    def log(self, header, value):
        assert header in self.headers, \
            "Cannot log value of new header, use add_header first."
        self.log_buffer[header] = value

    def flush(self, display=False):
        if display:
            self.display()

        save_file = open(self.save_file_path, "a+")
        if not self.headers_written:
            save_file.write("\t".join(self.headers) + "\n")
            self.headers_written = True

        row = []
        for header in self.headers:
            row.append(str(self.log_buffer[header]))

        save_file.write("\t".join(row) + "\n")
        save_file.close()

    def display(self):
        table = PrettyTable()
        table.field_name = ["Metric", "Value"]
        for header in self.headers:
            val = self.log_buffer[header]
            val = f"{val:.6f}" if isinstance(val, float) else str(val)
            table.add_row([header, val])
        print()
        print(table)
        print()
