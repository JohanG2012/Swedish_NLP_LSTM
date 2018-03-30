# Imports
import os
import urllib.request
import tarfile
import bz2
from xml.dom import minidom
from xml import sax
import re
import pickle
import datetime as dt
from bs4 import BeautifulSoup

# Contants
DATA_LOCATION = "./data/"
URL = "http://spraakbanken.gu.se/lb/resurser/meningsmangder/"
DATA_1950 = "gigaword-1950-59.tar"
DATA_1960 = "gigaword-1960-69.tar"
DATA_1970 = "gigaword-1970-79.tar"
DATA_1980 = "gigaword-1980-89.tar"
DATA_1990 = "gigaword-1990-99.tar"
DATA_2000 = "gigaword-2000-09.tar"
DATA_2010 = "gigaword-2010-15.tar"
FILES = [DATA_1950, DATA_1960, DATA_1970, DATA_1980, DATA_1990, DATA_2000, DATA_2010]
#FILES = [DATA_1950]

def download(file, url, location):

    # Get file if it does not exist
    if not os.path.exists(location + file):
        print("Downloading " + file + "...")
        start_time = dt.datetime.now()
        file, _ = urllib.request.urlretrieve(url + file, location + file)
        end_time = dt.datetime.now()
        print("Downloading {0} took {1} minutes to run.".format(file, (end_time-start_time).total_seconds() / 60.0))

    return file

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
                with open(subdir + '/' + file, buffering=200000) as xml_file:
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
                with open(location + "vocabulary.pkl", "ab") as pkl:
                    print("Writing data to pickle...")
                    pickle.dump(data, pkl)
                end_time = dt.datetime.now()
                print("Reading words took {} minutes to run.".format((end_time-start_time).total_seconds() / 60.0))

def parse_xml(location):
    print("Parsing XML to vocabulary pickle...")
    print("Parsing one billion words, this might take some time...")
    data = list()
    log_every = 100000
    line_num = 0
    start_time = dt.datetime.now()
    data_length = 0

    for subdir, dirs, files in os.walk(location):
        for file in files:
            if file.endswith(".xml"):
                with open(subdir + "/" + file, buffering=200000) as xml_file:
                    print("Processing: " + subdir + "/" + file)
                    soup = BeautifulSoup(xml_file, 'xml.parser')
                    for line in xml_file:
                        line = line.rstrip()

                        # Check if not a row with only a start or end tag i.e sentences
                        if "</" in line and not line[1:2] == "/":
                            line_num += 1
                            node = minidom.parseString(line)
                            data_length += 1
                            data.append(node.getElementsByTagName('w')[0].firstChild.nodeValue)
                            if line_num == log_every:
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

download_files(FILES, URL, DATA_LOCATION)
#parse_xml(DATA_LOCATION)
parse_to_plaintext(DATA_LOCATION)
parse_soup(DATA_LOCATION)
