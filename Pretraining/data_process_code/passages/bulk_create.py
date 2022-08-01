# coding=utf8

import os
import json
import logging
import argparse
import collections
import multiprocessing
from pprint import pprint
from typing import List, Dict
from create_passages import create
from util.line_corpus import expand_files, read_lines, JsonlWriter


logging.basicConfig(format='%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def reader(msg_queue: multiprocessing.Queue) -> None:
    failure_statistics = collections.Counter()
    while True:
        msg = msg_queue.get()
        if isinstance(msg, str) and msg == 'kill':
            pprint(failure_statistics)
            break
        else:
            assert isinstance(msg, dict)
            # Aggregate failure statistic
            for key, value in msg.items():
                failure_statistics[key] = value + failure_statistics[key]


def worker(rank: int, rank_jobs: List[str], rank_output_path: str, msg_queue: multiprocessing.Queue, 
           line_per_file: int, num_words: int, minimum_num_words: int) -> None:

    logger.info("Start Worker: %d" % rank)
    rank_writer = JsonlWriter(line_per_file, output_directory=rank_output_path)

    exception_fd = open(os.path.join(rank_output_path, "%d_create_exceptions_list.txt" % rank), 'w')
    total_documents, total_passages, exception_count = 0, 0, 0
    failure_statistics = collections.Counter()
    
    reg_error=0
    for line in read_lines(rank_jobs):
        total_documents += 1

        doc = json.loads(line.strip())

        try:
            passages, reg_error = create(doc, num_words, minimum_num_words, reg_error)
            total_passages += len(passages)
            for psge in passages:
                rank_writer.write(psge)
        except Exception as e:
            reason = str(e)
            failure_statistics[reason] += 1
            exception_fd.write("%s, %s, %s\n" % (doc['title'], doc['url'], reason,))
            exception_count += 1

        if total_passages % 1000 == 0:
            logger.info("Rank %d: processes %d documents and %d passages. Valid Documents: %d; Failed Documents: %d, Reg_error: %d" % (rank, total_documents, 
                        total_passages, total_documents - exception_count, exception_count, reg_error))

    rank_writer.close()
    exception_fd.close()
    msg_queue.put(failure_statistics)
    msg_queue.put({"total_documents": total_documents, "total_passages": total_passages, "valid_documents": total_documents - exception_count})


def main(input_path: str, output_path: str, line_per_file: int, num_words: int, window_size: int, num_processes: int,
         minimum_num_words: int) -> None:

    os.makedirs(output_path, exist_ok=True)

    world_size = num_processes
    world_rank = [i for i in range(world_size)]
    total_jobs = expand_files([input_path], file_pattern="*")
    with multiprocessing.Pool(processes=world_size + 1) as pool:
        m = multiprocessing.Manager()
        msg_queue = m.Queue()
        reader_proc = pool.apply_async(reader, (msg_queue,)) # Aggregate information from worker

        proc_pool = list()
        for rank in world_rank:
            rank_jobs = total_jobs[rank::world_size]
            rank_output_path = os.path.join(output_path, "rank_%d" % rank)
            logger.info("Number of Jobs for Rank %d: %d" % (rank, len(rank_jobs),))
            worker_proc = pool.apply_async(worker, (rank, rank_jobs, rank_output_path, msg_queue, line_per_file, num_words, minimum_num_words))
            proc_pool.append(worker_proc)

        # collect results from the workers through the pool result queue
        for proc in proc_pool: 
            proc.get()

        print("now we are done, kill the reader")
        # now we are done, kill the reader
        msg_queue.put('kill')
        pool.close()
        pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", 
                        # default="../../data/wikiextracteddir", 
                        default="/home/hk/workshop_43/My_project/Wiki/data/wikiextracteddir",
                        type=str, 
                        # required=True,
                        help="The input data dir. Should contain the json output of WikiExtractor.")
    parser.add_argument("--output", "-o", 
                        # default="../../data/wiki_passages_with_link_120", 
                        default="/home/hk/workshop_43/My_project/Wiki/data/wiki_3", 
                        type=str, 
                        # required=True,
                        help="The output directory.")
    parser.add_argument("--passages_per_file", default=100000, type=int, 
                        # required=True,
                        help="Number of passages in each file.")
    parser.add_argument("--num_words", "-n", default=120, type=int, 
                        # required=True,
                        help="Expected number of words of each passage.")
    parser.add_argument("--window_size", "-w", default=0, type=int,
                        help="Sliding window to passages")
    parser.add_argument("--num_processes", "-p", default=10, type=int,
                        # required=True,
                        help="Number of processes to parse page")
    parser.add_argument("--minimum_num_words", "-m", default=10, type=int,
                        help="Minimum number of words in passages")

    args = parser.parse_args()
    main(args.input, args.output, args.passages_per_file, args.num_words, args.window_size, 
         args.num_processes, args.minimum_num_words)
    
    with open("./finsied.txt", "w") as F:
        F.write("FINISH")