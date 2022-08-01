# coding=utf8

import os
import glob
import gzip
import json
import logging
from typing import Dict
from overrides import overrides


logger = logging.getLogger(__name__)


class BaseWriter:
    def __init__(self, line_per_file: int, output_directory: str, suffix: str = "txt", enable_gzip: bool = True):
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        self.output_directory = output_directory
        self.line_per_file = line_per_file
        self.current_fd = None
        self.file_count = 0
        self.current_file_line_count = 0
        self.total_line = 0
        self._suffix = suffix
        self._enable_gzip = enable_gzip

    def _write(self, content) -> None:
        raise NotImplementedError

    def write(self, content) -> None:
        if self.current_file_line_count > self.line_per_file:
            self.current_fd.close()
            self.current_fd = None
            self.total_line += self.current_file_line_count
            self.current_file_line_count = 0
        if self.current_fd is None:
            if self._enable_gzip:
                path = os.path.join(self.output_directory, "%d.%s.gz" % (self.file_count, self._suffix))
                self.current_fd = gzip.open(path, 'wt', encoding='utf-8')
            else:
                path = os.path.join(self.output_directory, "%d.%s" % (self.file_count, self._suffix))
                self.current_fd = open(path, "w", encoding='utf-8')
            self.file_count += 1

        self._write(content)
        self.current_file_line_count += 1

    def close(self):
        if self.current_fd is not None:
            self.current_fd.close()
            self.current_fd = None
            self.total_line += self.current_file_line_count
            self.current_file_line_count = 0
        logger.info("Total Files: %d; Total Lines: %d." % (self.file_count, self.total_line))


class JsonlWriter(BaseWriter):

    def __init__(self, line_per_file: int, output_directory: str, enable_gzip: bool = True):
        super().__init__(line_per_file=line_per_file, output_directory=output_directory, 
                         suffix="jsonl", enable_gzip=enable_gzip)

    @overrides
    def _write(self, content: Dict) -> None:
        self.current_fd.write(json.dumps(content) + '\n')


class TxtWriter(BaseWriter):

    def __init__(self, line_per_file: int, output_directory: str, enable_gzip: bool = True, suffix: str = "txtpb"):
        super().__init__(line_per_file=line_per_file, output_directory=output_directory, 
                         suffix=suffix, enable_gzip=enable_gzip)

    @overrides
    def _write(self, content: str) -> None:
        self.current_fd.write(content + '\n')


def jsonl_lines(input_files, completed_files=None, limit=0, report_every=100000):
    return read_lines(jsonl_files(input_files, completed_files), limit=limit, report_every=report_every)


def jsonl_files(input_files, completed_files=None):
    return expand_files(input_files, '*.jsonl*', completed_files)


def expand_files(input_files, file_pattern='*', completed_files=None):
    """
    expand the list of files and directories
    :param input_files:
    :param file_pattern: glob pattern for recursive example '*.jsonl*' for jsonl and jsonl.gz
    :param completed_files: these will not be returned in the final list
    :return:
    """
    if type(input_files) is str:
        input_files = [input_files]
    # expand input files recursively
    all_input_files = []
    if completed_files is None:
        completed_files = []
    for input_file in input_files:
        if input_file in completed_files:
            continue
        if os.path.isdir(input_file):
            sub_files = glob.glob(input_file + "/**/" + file_pattern, recursive=True)
            sub_files = [f for f in sub_files if not os.path.isdir(f)]
            sub_files = [f for f in sub_files if f not in input_files and f not in completed_files]
            all_input_files.extend(sub_files)
        else:
            all_input_files.append(input_file)
    return all_input_files


def read_lines(input_files, limit=0, report_every=100000):
    """
    This takes a list of input files and iterates over the lines in them
    :param input_files: Directory name or list of file names
    :param completed_files: The files we have already processed; We won't read these again.
    :param limit: maximum number of examples to load
    :return:
    """
    count = 0
    for input_file in input_files:
        if input_file.endswith(".gz"):
            reader = gzip.open(input_file, "rt")
        else:
            reader = open(input_file, "r")
        with reader:
            for line in reader:
                yield line
                count += 1
                if count % report_every == 0:
                    logger.info(f'On line {count} in {input_file}')
                if 0 < limit <= count:
                    return


def read_specific_line(input_file: str, line_number: int) -> str:
    if input_file.endswith(".gz"):
        reader = gzip.open(input_file, "rt")
    else:
        reader = open(input_file, "r")
    with reader:
        lines = reader.readlines()
        return lines[line_number]
