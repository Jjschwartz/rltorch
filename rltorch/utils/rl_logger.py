import time
import os.path as osp
from prettytable import PrettyTable

from rltorch.user_config import DEFAULT_DATA_DIR


class RLLogger:

    def __init__(self, env_name, alg=None):
        self.env_name = env_name
        self.alg = alg
        self.setup_save_file()
        self.headers = []
        self.log_buffer = dict()
        self.headers_written = False

    def setup_save_file(self):
        self.save_file_path = self.get_save_path("tsv")

    def get_save_path(self, ext=None):
        ts = time.strftime("%Y%m%d-%H%M")
        save_file_path = osp.join(DEFAULT_DATA_DIR, f"{self.alg}_{self.env_name}_{ts}")
        if ext:
            save_file_path += f".{ext}"
        return save_file_path

    def add_header(self, header):
        assert header not in self.headers
        self.headers.append(header)

    def log(self, header, value):
        assert header in self.headers, "Cannot log value of new header, use add_header first."
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
            val = f"{val:.4f}" if isinstance(val, float) else str(val)
            table.add_row([header, val])
        print()
        print(table)
        print()
