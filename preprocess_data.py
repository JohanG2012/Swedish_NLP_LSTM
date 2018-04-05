# Imports
import os
import urllib.request
import requests
import tarfile
import bz2
from xml.dom import minidom
from xml import sax
import re
import pickle
import datetime as dt
import xml.etree.cElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# Contants
DATA_LOCATION = "./data"
URL = "http://spraakbanken.gu.se/lb/resurser/meningsmangder/"
DATA_1950 = "gigaword-1950-59.tar"
DATA_1960 = "gigaword-1960-69.tar"
DATA_1970 = "gigaword-1970-79.tar"
DATA_1980 = "gigaword-1980-89.tar"
DATA_1990 = "gigaword-1990-99.tar"
DATA_2000 = "gigaword-2000-09.tar"
DATA_2010 = "gigaword-2010-15.tar"
GP_1994 = "gp1994.xml.bz2"
GP_2001 = "gp2001.xml.bz2"
GP_2002 = "gp2002.xml.bz2"
GP_2003 = "gp2003.xml.bz2"
GP_2004 = "gp2004.xml.bz2"
GP_2005 = "gp2005.xml.bz2"
GP_2006 = "gp2006.xml.bz2"
GP_2007 = "gp2007.xml.bz2"
GP_2008 = "gp2008.xml.bz2"
GP_2009 = "gp2009.xml.bz2"
GP_2010 = "gp2010.xml.bz2"
GP_2011 = "gp2011.xml.bz2"
GP_2012 = "gp2012.xml.bz2"
GP_2013 = "gp2013.xml.bz2"
GP_2D = "gp2d.xml.bz2"
WN_2001 = "webbnyheter2001.xml.bz2"
WN_2002 = "webbnyheter2002.xml.bz2"
WN_2003 = "webbnyheter2003.xml.bz2"
WN_2004 = "webbnyheter2004.xml.bz2"
WN_2005 = "webbnyheter2005.xml.bz2"
WN_2006 = "webbnyheter2006.xml.bz2"
WN_2007 = "webbnyheter2007.xml.bz2"
WN_2008 = "webbnyheter2008.xml.bz2"
WN_2009 = "webbnyheter2009.xml.bz2"
WN_2010 = "webbnyheter2010.xml.bz2"
WN_2011 = "webbnyheter2011.xml.bz2"
WN_2012 = "webbnyheter2012.xml.bz2"
WN_2013 = "webbnyheter2013.xml.bz2"
FILES = [DATA_1950, DATA_1960, DATA_1970, DATA_1980, DATA_1990, DATA_2000, DATA_2010]
WN_FILES = [WN_2001, WN_2002, WN_2003, WN_2004, WN_2005, WN_2006, WN_2007, WN_2008, WN_2009, WN_2010, WN_2011, WN_2012, WN_2013]
GP_FILES = [GP_1994, GP_2001, GP_2002, GP_2003, GP_2004, GP_2005, GP_2006, GP_2007, GP_2008, GP_2009, GP_2010, GP_2011, GP_2012, GP_2013, GP_2D]
DOWNLOAD_FILES = GP_FILES + WN_FILES

def download(file, url, location):

    # Get file if it does not exist
    if not os.path.exists(location + "/" + file):
        print("Downloading " + file + "...")
        start_time = dt.datetime.now()
        #file, _ = urllib.request.urlretrieve(url + file, location + "/" + file)
        file = download_file(url + file, location + '/' + file)
        end_time = dt.datetime.now()
        print("Downloading {0} took {1} minutes to run.".format(file, (end_time-start_time).total_seconds() / 60.0))

    return file

def download_file(url, local_filename):
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return local_filename

def extract_files(location, file):
    for file in os.listdir(location):
        if file.endswith(".tar"):
            print("Extracting " + file + "...")
            tar = tarfile.open(location + file)
            tar.extractall(location)
            tar.close
            print("Removing " + file + "...")
            os.remove(location + file)

    for subdir, dirs, files in os.walk(location):
        for file in files:
            if file.endswith(".bz2") and not os.path.exists(location + file[:-4]):
                start_time = dt.datetime.now()
                print("Extracting " + subdir + "/" + file + "...")
                bz2file = bz2.BZ2File(os.path.join(subdir, file), 'rb')
                data = bz2file.read()
                new_file = os.path.join(subdir, file)[:-4]
                open(new_file, 'wb').write(data)
                end_time = dt.datetime.now()
                print("Extracting {0} took {1} minutes to run.".format(file, (end_time-start_time).total_seconds() / 60.0))
                print("Removing " + subdir + "/" + file + "...")
                os.remove(subdir + "/" + file)

def download_files(files, url, location):
    print("Downloading and extracting one billion words. This might take some time...")
    for file in files:
        download(file, url, location)
        print("Extracting files...")
        extract_files(location, file)

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

def parse_xml(location):
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

def preprocess_sentences(location):
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

def noise_maker(sentence, threshold):
    vocab_to_int = pickle.load(open("./data/vocab_to_int.pkl", "rb"))
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p','q','r','s','t','u','v','w','x','y','z','å', 'ä', 'ö']
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            # ~33% chance characters will swap locations
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_sentence.append(sentence[i+1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            # ~33% chance a character will not be typed
            else:
                pass
        i += 1
    return noisy_sentence

def create_trainingsets(location):
    training_sorted = []
    testing_sorted = []
    int_to_vocab = pickle.load( open( "./data/int_to_vocab.pkl", "rb" ) )
    max_length = 92
    min_length = 50
    good_sentences = pickle.load(open(location + "/good_sentences.pkl", "rb"))
    training, testing = train_test_split(good_sentences, test_size = 0.15, random_state = 2)

    for i in range(min_length, max_length+1):
        for sentence in training:
            if len(sentence) == i:
                training_sorted.append(sentence)
        for sentence in testing:
            if len(sentence) == i:
                testing_sorted.append(sentence)

    for i in range(5):
        print(training_sorted[i], len(training_sorted[i]))

    threshold = 0.9
    for sentence in training_sorted[:5]:
        print("Sentence: ")
        print("".join([int_to_vocab[i] for i in sentence]))
        print("With Noise: ")
        print("".join([int_to_vocab[i] for i in noise_maker(sentence, threshold)]))
        print()
    with open(location + '/training_sorted.pkl', 'wb') as pkl:
        print('Writing training_sorted to pickle...')
        pickle.dump(training_sorted, pkl)
    with open(location + '/testing_sorted.pkl', 'wb') as pkl:
        print('Writing testing_sorted to pickle...')
        pickle.dump(testing_sorted, pkl)

    print("Trainingset: {0} sentences".format(len(training_sorted)))
    print("Testingset: {0} sentences".format(len(testing_sorted)))

#download_files(DOWNLOAD_FILES, URL, DATA_LOCATION)
#old_parse_xml(DATA_LOCATION)
#parse_to_plaintext(DATA_LOCATION)
#parse_xml(DATA_LOCATION)
#preprocess_sentences(DATA_LOCATION)
create_trainingsets(DATA_LOCATION)
