import time
import os.path as osp

RESULTS_DIR = "data"


class DQNLogger:

    def __init__(self, env_name):
        self.env_name = env_name
        self.setup_save_file()
        self.headers = []
        self.log_buffer = dict()
        self.headers_written = False

    def setup_save_file(self):
        ts = time.strftime("%Y%m%d-%H%M")
        self.save_file_path = osp.join(RESULTS_DIR, f"{self.env_name}_{ts}.tsv")

    def add_header(self, header):
        assert header not in self.headers
        self.headers.append(header)

    def log(self, header, value):
        assert header in self.headers, "Cannot log value of new header, use add_header first."
        self.log_buffer[header] = value

    def flush(self):
        save_file = open(self.save_file_path, "a+")
        if not self.headers_written:
            save_file.write("\t".join(self.headers) + "\n")
            self.headers_written = True

        row = []
        for header in self.headers:
            row.append(str(self.log_buffer[header]))

        save_file.write("\t".join(row) + "\n")
        save_file.close()
