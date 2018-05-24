import os
import datetime as dt
import requests
import bz2

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
                print(file, files, os.path.join(subdir, file))
                start_time = dt.datetime.now()
                print("Extracting " + subdir + "/" + file + "...")
                bz2file = bz2.BZ2File(os.path.join(subdir, file), 'rb')
                #data = bz2file.read()
                new_file = os.path.join(subdir, file)[:-4]
                CHUNK = 128 * 1024 * 1024
                with open(new_file, 'wb') as f:
                    while True:
                        chunk = bz2file.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                end_time = dt.datetime.now()
                print("Extracting {0} took {1} minutes to run.".format(file, (end_time-start_time).total_seconds() / 60.0))
                print("Removing " + subdir + "/" + file + "...")
                os.remove(subdir + "/" + file)

def download_files(files = DOWNLOAD_FILES, url = URL, location = DATA_LOCATION):
    print("Downloading and extracting one billion words. This might take some time...")
    for file in files:
        download(file, url, location)
        print("Extracting files...")
        extract_files(location, file)

if __name__ == "__main__":
    download_files(DOWNLOAD_FILES, URL, DATA_LOCATION)
