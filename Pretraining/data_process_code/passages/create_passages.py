# coding=utf8

import os
import copy
import json
import gzip
import math
import time
import logging
import argparse
import unicodedata
import regex as re
from typing import Dict, List, Tuple
from urllib.parse import unquote
import numpy as np
import spacy
from util.reporting import Reporting
from util.line_corpus import expand_files, read_lines, JsonlWriter

"""
(Only split paragraphs)
Split documents into passages. Each passage has a fixed length (e.g., 100 words).
We can configure a sliding window for the passage. (Should try it)
We can also filter out those passages without any wikilink.

TODO: We may improve the creation process of passages with the following Sentence Parser. 
It identifies sentences with some heuristic rules.
https://github.com/spencermountain/wtf_wikipedia/blob/master/src/04-sentence/parse.js

TODO: Fix bug, the current way to break paragraphs may break "anchor link <a>"

"""


logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


SPLIT_PATTERN = re.compile(r"""\s*\n\s*\n\s*""")
WIKILINK_PATTERN = re.compile(r'<a\s+href=\"(.*?)"\>(.*?)<\/a>')

nlp = spacy.load("en_core_web_sm")

def re_search_token(token):
    token = re.sub("\(", "\\(", token)
    token = re.sub("\)", "\\)", token)
    token = re.sub("\?", "\\?", token)
    token = re.sub("\+", "\\+", token)
    token = re.sub("\$", "\\$", token)
    token = re.sub("\[", "\\[", token)
    token = re.sub("\]", "\\]", token)
    token = re.sub("\.", "\\.", token)
    return token


def process_wikilink(text: str, reg_error: int) -> Tuple[str, List, int]:
    
    # matches = WIKILINK_PATTERN.findall(text)
    # entities = [(unquote(m[0]),"None_type") for m in matches]
    # processed_text = re.sub(WIKILINK_PATTERN, r'[ENT] \2 [/ENT]', text)

    # try:
    #     extra_entity_span_list = []
    #     doc = nlp(text)
    #     for ent in doc.ents:
    #         if (ent.start_char, ent.end_char) not in extra_entity_span_list:
    #             extra_entity_span_list.append((ent.start_char, ent.end_char))
    #             entities.append((ent.text,ent.label_))
    #     for my_entity in entities:
    #         processed_text = re.sub(re_search_token(my_entity[0]), r'[ENT] '+ my_entity[0] + ' [/ENT]', processed_text)
    # except:
    #     reg_error+=1
    #     return processed_text, entities
    
    processed_text = text    
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text,ent.label_,(ent.start_char, ent.end_char)))
        
    return processed_text, entities, reg_error


def create(document: Dict, num_words: int, minimum_num_words: int = 1, reg_error: int = 0) -> List[Dict]:

    title = document['title'].strip()
    text = document['text'].strip()

    # strip unicode combos
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])

    passages = list()
    paragraphs = re.split(SPLIT_PATTERN, text)
    for paragraph in paragraphs:

        # TODO: Use advanced tokenizers
        paragraph_tokens = paragraph.strip().split()
        expected_num_passages = math.ceil(len(paragraph_tokens) / num_words)

        # TODO: Support siliding windows
        for idx in range(expected_num_passages):
            beg, end = idx * num_words, (idx + 1) * num_words
            passage_tokens = paragraph_tokens[beg:end]
            # Empirical settings.
            diff = num_words - len(passage_tokens)
            if diff > 0:
                beg = max(0, beg - diff)
                passage_tokens = paragraph_tokens[beg:]
            # Drop passages with too few words
            if len(passage_tokens) < minimum_num_words:
                continue

            # NER
            passage = " ".join(passage_tokens).strip()
            processed_passage, entities, reg_error = process_wikilink(passage, reg_error)
            if not any([e[0].lower() == title.lower() for e in entities]):
                entities.append((title, "None_type"))

            passage_id = "wp_%s_%d" % (document['id'], len(passages),)
            adoc = {
                'id': document['id'], 'url': document['url'], 'title': document['title'],
                'text': processed_passage, 'entities': entities, 'passage_id': passage_id
            }
            passages.append(adoc)

    return passages, reg_error


def main(input_path: str, output_path: str, passages_per_file: int, num_words: int, window_size: int,
         has_wikilink: bool = False) -> None:
    """
    input format: {"id": "", "revid": "", "url":"", "title": "", "text": "..."}
    output format: {"id": "", "url":"", "title": "", "text": ""}
    both jsonl
    :return:
    """
    print(passages_per_file)
    writer = JsonlWriter(line_per_file=passages_per_file, output_directory=output_path)
    report = Reporting()
    total_passage_count = 0
    para_split = re.compile(r"""\s*\n\s*\n\s*""")

    for line in read_lines(expand_files(input_path, '*')):
        if report.is_time():
            report.display()
            logger.info(f'On document {report.check_count}, '
                        f'{report.check_count/(time.time()-report.start_time)} documents per second')
        jobj = json.loads(line)
        text = jobj['text'].strip()

        # strip unicode combos
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])

        paragraphs = re.split(para_split, text)
        passage_count = 0
        for paragraph in paragraphs:

            # TODO: Use advanced tokenizers
            paragraph_tokens = paragraph.strip().split()
            expected_num_passages = math.ceil(len(paragraph_tokens) / num_words)

            # TODO: Support siliding windows
            for idx in range(expected_num_passages):
                beg, end = idx * num_words, (idx + 1) * num_words
                passage_tokens = paragraph_tokens[beg:end]
                # Empirical settings.
                diff = num_words - len(passage_tokens)
                if diff > 0:
                    beg = max(0, beg - diff)
                    passage_tokens = paragraph_tokens[beg:]

                passage = " ".join(passage_tokens).strip()
                if has_wikilink:
                    # TODO: Remove those without wikilink
                    pass

                # Process passages

                passage_count += 1
                # build passage document
                adoc = {
                    'id': jobj['id'], 'url': jobj['url'], 'title': jobj['title'], 'text': passage
                }
                # write to file
                writer.write(adoc)

        total_passage_count += passage_count
        report.moving_averages(paragraphs_per_doc=len(paragraphs), passages_per_doc=passage_count)

    writer.close()
    logger.info("Total number of passages: %d" % total_passage_count)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", default="/home/hk/workshop_43/My_project/FSL/data/Wiki/data/wikiextracteddir", type=str, 
#                         # required=True,
#                         help="The input data dir. Should contain the json output of WikiExtractor.")
#     parser.add_argument("--output", default="/home/hk/workshop_43/My_project/FSL/data/Wiki/data/wiki_passages_with_link_120", type=str, 
#                         # required=True,
#                         help="The output directory.")
#     parser.add_argument("--passages_per_file", default=100000, type=int,
#                         help="Number of passages in each file.")
#     parser.add_argument("--num_words", default=120, type=int,
#                         help="Expected number of words of each passage.")
#     parser.add_argument("--window_size", default=0, type=int,
#                         help="Sliding window to passages")
#     args = parser.parse_args()
#     main(args.input, args.output, args.passages_per_file, args.num_words, args.window_size)
