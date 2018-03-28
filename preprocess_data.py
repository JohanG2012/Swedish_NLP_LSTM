# Imports
import os
import urllib.request
import tarfile
import bz2
from xml.dom import minidom

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
        file, _ = urllib.request.urlretrieve(url + file, location + file)

    return file

def extract_files(location, file):
    for file in os.listdir(location):
        if file.endswith(".tar"):
            tar = tarfile.open(location + file)
            tar.extractall(location)
            tar.close
            os.remove(location + file)

    for subdir, dirs, files in os.walk(location):
        for file in files:
            if file.endswith(".bz2") and not os.path.exists(location + file[:-4]):
                bz2file = bz2.BZ2File(os.path.join(subdir, file), 'rb')
                data = bz2file.read()
                new_file = os.path.join(subdir, file)[:-4]
                open(new_file, 'wb').write(data)
                os.remove(subdir + "/" + file)

def download_files(files, url, location):
    for file in files:
        download(file, url, location)
        extract_files(location, file)

def parse_xml(location):
    for subdir, dirs, files in os.walk(location):
        if file.endswith(".xml"):
            xml_file = minidom.parse(location + file)
            sentence_node = xml_file.getElementByTagName('sentence')
            print(sentence_node.textContent)

download_files(FILES, URL, DATA_LOCATION)
parse_xml(DATA_LOCATION)
