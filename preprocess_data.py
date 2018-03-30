# Imports
import os
import urllib.request
import tarfile
import bz2
from xml.dom import minidom
import pickle
import datetime as dt

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
        extract_files(location, file)

def parse_xml(location):
    print("Parsing XML to vocabulary pickle...")
    print("Parsing one billion words, this might take some time...")
    data = list()
    log_every = 100000
    line_num = 0
    start_time = dt.datetime.now()

    for subdir, dirs, files in os.walk(location):
        for file in files:
            if file.endswith(".xml"):
                with open(subdir + "/" + file, buffering=200000) as xml_file:
                    print("Processing: " + subdir + "/" + file)
                    for line in xml_file:
                        line = line.rstrip()

                        # Check if not a row with only a start or end tag i.e sentences
                        if "</" in line and not line[1:2] == "/":
                            line_num += 1
                            node = minidom.parseString(line)
                            data.append(node.getElementsByTagName('w')[0].firstChild.nodeValue)
                            if line_num == log_every:
                                print("{0} words has been added to the vocabulary. {1}% of 100%".format(len(data), round((len(data) / 1000000000) * 100, 3)))
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
parse_xml(DATA_LOCATION)
