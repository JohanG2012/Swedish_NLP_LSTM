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

def parse_xml(location):
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

def hacky_hack(location):
    for subdir, dirs, files in os.walk(location):
        for file in files:
            if file.endswith(".xml"):
                with open(subdir + "/" + file, buffering=200000) as xml_file:
                    sentences = ''
                    named_entities = []
                    words = 0
                    start_time = dt.datetime.now()
                    print("Processing: " + subdir + "/" + file)
                    #tree = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml_file) + "</root>")
                    context = ET.iterparse(xml_file, events=('start','end'))
                    context = iter(context)
                    event, root = next(context)
                    for event, element in context:
                        if event == 'end' and element.tag == 'ne' and element.attrib['ex'] == 'ENAMEX':
                            named_entities.append(element.attrib['name'])
                        elif event == 'end' and element.tag == 'w':
                            sentences += str(element.text + ' ')
                            element.clear()
                            root.clear()
                            words += 1
                        elif event == 'end' and element.tag == 'sentence':
                            sentences += '\n'
                            element.clear()
                            root.clear()
                    end_time = dt.datetime.now()
                    total_seconds = (end_time-start_time).total_seconds()
                    print("Reading {} words took {} minutes to run. ({} words / second)".format(words, total_seconds / 60.0, words/total_seconds))
                    with open(location + '/vocabulary.pkl', 'ab') as pkl:
                        print('Writing sentences to pickle...')
                        pickle.dump(sentences, pkl)
                    with open(location + '/vocabulary_named_entities.pkl', 'ab') as pne:
                        print('Writing named entities to pickle...')
                        pickle.dump(named_entities, pne)

#download_files(DOWNLOAD_FILES, URL, DATA_LOCATION)
#parse_xml(DATA_LOCATION)
#parse_to_plaintext(DATA_LOCATION)
hacky_hack(DATA_LOCATION)
