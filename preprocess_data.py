# Imports

import urllib.request
import tarfile
from xml.dom import minidom
from xml import sax
import re
import xml.etree.cElementTree as ET
import pandas as pd
from collections import Counter
import random
from noise_maker import noise_maker
import datetime as dt
import pickle

import os

DATA_LOCATION = './data'

data = ''

WHITESPACE_FILTER = re.compile(r'[^\S\n]+', re.UNICODE)
DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
QUOTE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(chr(768), chr(769), chr(832), chr(833), chr(2387), chr(5151), chr(5152), chr(65344), chr(8242)), re.UNICODE)
LEFT_BRACKET_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RIGHT_BRACKET_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
ALLOWED_SPECIAL_CHARS = """-!?/;"'%&<>.()[]{}@#:,|=*"""
CLEANER = re.compile(r'[^\w\s{}]'.format(re.escape(ALLOWED_SPECIAL_CHARS)), re.UNICODE)

def parse_xml(location = DATA_LOCATION):
    words = 0
    log_every = 100000
    start_time = dt.datetime.now()
    for subdir, dirs, files in os.walk(location):
        for file in files:
            if file.endswith(".xml"):
                with open(subdir + "/" + file, buffering=200000) as xml_file:
                    sentences = ''
                    named_entities = []
                    print("Processing: " + subdir + "/" + file)
                    #tree = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml_file) + "</root>")
                    context = ET.iterparse(xml_file, events=('start','end'))
                    context = iter(context)
                    event, root = next(context)
                    for event, element in context:
                        if event == 'end' and element.tag == 'ne' and element.attrib['ex'] == 'ENAMEX':
                            named_entities.append(element.attrib['name'])
                        elif event == 'end' and element.tag == 'w':
                            if isinstance(element.text, str):
                                sentences += str(element.text + ' ')
                                element.clear()
                                root.clear()
                                words += 1
                                if words == log_every:
                                    print("{0} words has been added to the vocabulary. {1}% of 100%".format(words, round((words / 543800000) * 100, 3)))
                                    log_every += 100000
                        elif event == 'end' and element.tag == 'sentence':
                            sentences += '\n'
                            element.clear()
                            root.clear()
                    with open(location + '/vocabulary.pkl', 'ab') as pkl:
                        print('Writing sentences to pickle...')
                        pickle.dump(sentences, pkl)
                    with open(location + '/vocabulary_named_entities.pkl', 'ab') as pne:
                        print('Writing named entities to pickle...')
                        pickle.dump(named_entities, pne)

    end_time = dt.datetime.now()
    total_seconds = (end_time-start_time).total_seconds()
    print("Reading {} words took {} minutes to run. ({} words / second)".format(words, total_seconds / 60.0, words/total_seconds))

def pickleLoader(pklFile):
   try:
       while True:
           yield pickle.load(pklFile)
   except EOFError:
       pass

def clean_text(text):
    text = WHITESPACE_FILTER.sub(' ', text.strip()) # Replace different kind of whitespace with "normal" whitespace.
    text = DASH_FILTER.sub('-', text) # Replace different kind of dashes with a noram ("-") dash.
    text = QUOTE_FILTER.sub("'", text) # Replace different kind of quotes, with single quote.
    text = LEFT_BRACKET_FILTER.sub("(", text) # Replace all kinds of brackets with parentacis
    text = RIGHT_BRACKET_FILTER.sub(")", text) # Replace all kinds of brackets with parentacis
    text = CLEANER.sub('', text) # Basic cleanup
    return text.lower()

def preprocess_sentences(location = DATA_LOCATION):
    codes = ['<PAD>','<EOS>','<GO>']
    vocab_to_int = {}
    int_to_vocab = {}
    int_sentences = []
    count = 0
    sentence_list = []
    lengths = []
    max_length = 92
    min_length = 10
    good_sentences = []
    print(location + "/" + "vocabulary.pkl")
    with open(location + "/" + "vocabulary.pkl", "rb", 20000) as f:
        dump_num = 0
        for event in pickleLoader(f):
            dump_num += 1
            print("Loading dump {0}...".format(dump_num))
            for line in event.splitlines():
                line = clean_text(line)
                sentence_list.append(line)
            if dump_num >= 2:
                break
    print("Example sentence: " + sentence_list[0][:500])

    counter = Counter()
    print("Reading characters:")
    for line in sentence_list:
        counter.update(line)
    top_chars = {key for key, _value
    in counter.most_common(56)}


    print("Converting Vocabulary to integers...")
    iterator = iter(sentence_list)
    for sentence in iterator:
        charIterator = iter(sentence)
        for character in charIterator:
            if character not in vocab_to_int:
                vocab_to_int[character] = count
                count += 1

    print("Adding codes/tokens...")
    for code in codes:
        vocab_to_int[code] = count
        count += 1

    vocab_size = len(vocab_to_int)
    print("The vocabulary contains {} characters.".format(vocab_size))
    print("Vocabulary contains: " + "".join(sorted(vocab_to_int)))
    for character, value in vocab_to_int.items():
        int_to_vocab[value] = character

    print("Creating lookup table from int to char...")
    iterator = iter(sentence_list)
    num_sent = 0
    log_every = 1000000
    for sentence in iterator:
        int_sentence = []
        chariterator = iter(sentence)
        for character in chariterator:
            int_sentence.append(vocab_to_int[character])
        int_sentences.append(int_sentence)
        num_sent += 1
        if log_every == num_sent:
            log_every += 1000000
            print("Transforming characters to int. {0}% of 100%".format(round((num_sent / len(sentence_list)) * 100, 3)))

    new_sentence_list = []
    for sent in sentence_list:
        if not bool(set(sent) - set(sorted(vocab_to_int))):
            new_sentence_list.append(sent)
    sentence_list[:] = []

    with open(location + '/sentence_list.pkl', 'wb') as pkl:
        print('Writing sentence list to pickle...')
        pickle.dump(new_sentence_list, pkl)
    new_sentence_list[:] = []

    print("Converting sentences to integers...")
    for sentence in int_sentences:
        lengths.append(len(sentence))
    lengths = pd.DataFrame(lengths, columns=["counts"])

    for sentence in int_sentences:
        if len(sentence) <= max_length and len(sentence) >= min_length:
            good_sentences.append(sentence)

    print("We will use {} sentences to train and test our model.".format(len(good_sentences)))
    with open(location + '/vocab_to_int.pkl', 'wb') as pkl:
        print('Writing vocab_to_int to pickle...')
        pickle.dump(vocab_to_int, pkl)
    with open(location + '/int_to_vocab.pkl', 'wb') as pkl:
        print('Writing int_to_voac to pickle...')
        pickle.dump(int_to_vocab, pkl)
    with open(location + '/good_sentences.pkl', 'wb') as pkl:
        print('Writing good sentences list to pickle...')
        pickle.dump(good_sentences, pkl)
    with open(location + '/int_sentences.pkl', 'wb') as pkl:
        print('Writing int_sentence to pickle...')
        pickle.dump(int_sentences, pkl)




if __name__ == "__main__":
    #old_parse_xml(DATA_LOCATION)
    #parse_to_plaintext(DATA_LOCATION)
    #parse_xml(DATA_LOCATION)
    preprocess_sentences(DATA_LOCATION)
