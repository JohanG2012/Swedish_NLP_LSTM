# Imports

import urllib.request
import tarfile
from xml.dom import minidom
from xml import sax
import re
import xml.etree.cElementTree as ET
import pandas as pd
import random
from noise_maker import noise_maker




data = ''

def parse_to_plaintext(location):
    start_time = dt.datetime.now()
    for subdir, dirs, files in os.walk(location):
        for file in files:
            if file.endswith('.xml'):
                with open(subdir + '/' + file, buffering=2000) as xml_file:
                    def numrepl(matchobj):
                        numbers = {'0': 'noll', '1':'ett', '2':'två', '3':'tre','4':'fyra','5':'fem','6':'sex','7':'sju','8':'åtta','9':'nio'}
                        return numbers[matchobj.group(0)] + ' '

                    r = re.compile('<[^>]*>')
                    n = re.compile('\n')
                    s = re.compile('[^A-Za-z0-9ÅÄÖÉåäöé ]')
                    p = re.compile(' +')
                    num = re.compile('\d')
                    print('Removing w tags...')
                    data = r.sub('',xml_file.read())
                    print('Converting newline to spaces...')
                    data = n.sub(' ',data)
                    print('Stripping special characters...')
                    data = s.sub('',data)
                    print('Converting numbers to words...')
                    data = num.sub(numrepl, data)
                    print('Converting multiple spaces to single spaces...')
                    data = p.sub(' ',data)
                    print('Converting to lowercase...')
                    data = data.lower()
                with open(location + "vocabulary_plain.pkl", "ab") as pkl:
                    print("Writing data to pickle...")
                    pickle.dump(data, pkl)
                end_time = dt.datetime.now()
                os.remove(subdir + '/' + file)
                print("Reading words took {} minutes to run.".format((end_time-start_time).total_seconds() / 60.0))

def old_parse_xml(location):
    print("Parsing XML to vocabulary pickle...")
    print("Parsing one billion words, this might take some time...")
    data = list()
    log_every = 100000
    start_time = dt.datetime.now()
    data_length = 0

    for subdir, dirs, files in os.walk(location):
        for file in files:
            if file.endswith(".xml"):
                with open(subdir + "/" + file, buffering=200000) as xml_file:
                    print("Processing: " + subdir + "/" + file)
                    for line in xml_file:
                        line = line.rstrip()

                        # Check if not a row with only a start or end tag i.e sentences
                        if "</" in line and not line[1:2] == "/":
                            node = minidom.parseString(line)
                            data_length += 1
                            data.append(node.getElementsByTagName('w')[0].firstChild.nodeValue)
                            if data_length == log_every:
                                print("{0} words has been added to the vocabulary. {1}% of 100%".format(data_length, round((data_length / 1000000000) * 100, 3)))
                                log_every += 100000
                with open(location + "vocalbulary.pkl", "ab") as pkl:
                    print("Writing data to pickle...")
                    pickle.dump(data, pkl)
                print("Removing " + subdir + "/" + file + "...")
                os.remove(subdir + "/" + file)
                # Empty list
                data[:] = []
    end_time = dt.datetime.now()
    print("Reading data took {} minutes to run.".format((end_time-start_time).total_seconds() / 60.0))

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
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[{}@_*<>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    text = re.sub(' +',' ', text)
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
           if dump_num <= 1:
               print("Loading dump {0}...".format(dump_num))
               for line in event.splitlines():
                   line = clean_text(line)
                   sentence_list.append(line)
    print("Example sentence: " + sentence_list[0][:500])

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

    with open(location + '/sentence_list.pkl', 'wb') as pkl:
        print('Writing sentence list to pickle...')
        pickle.dump(sentence_list, pkl)
    sentence_list[:] = []

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





#old_parse_xml(DATA_LOCATION)
#parse_to_plaintext(DATA_LOCATION)
#parse_xml(DATA_LOCATION)
#preprocess_sentences(DATA_LOCATION)
