#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Stian Rödven Eide

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import re
import os
import bz2
import argparse
import contract


def extract_bw(sentence):
    '''This function takes an XML sentence as input and returns either a list
    of the words themselves (if plain mode is used) or a list of either the
    lemma, saldo or lex attributes, depending on mode. If any of the latter,
    only the first attribute is used for any given word, and if missing, it
    is substituted by the word.'''
    if mode == 'lemma':
        sep = ' '
    else:
        sep = '_'
    sentlist = []
    for w in sentence:
        if mode == 'plain':
            if w.text:
                sentlist.append(w.text)
            else:
                sentlist.append('noword')
        else:
            lexes = [l for l in w.attrib[mode].split('|')
                     if (l and not sep in l)]
            if lexes:
                sentlist.append('|'.join(lexes))
            elif w.text:
                sentlist.append(w.text)
            else:
                sentlist.append('noword')
    return sentlist


def process_dir(directory):
    '''This function traverses a directory and processes any bzipped
    xml files it finds. If the MWE option is used, it calls the check_mwe
    function in contract.py for each sentence element. If the MWE option is
    not used, the extract_bw function above is called instead.

    For each sentence, a string is written to outfile, formatted depending on
    whether plain, lemma, saldo or lex mode is used.'''
    global mode
    global mwe
    global genre
    global outfile
    global first
    for root, dirs, files in os.walk(directory):
        os.chdir(root)
        folder_content = os.listdir(root)
        xmldocs = [f for f in folder_content if f.endswith('.xml.bz2')]
        if xmldocs:
            for doc in xmldocs:
                noparse = False
                print("Processing {dir}/{file}".format(dir=root,file=doc))
                infile = bz2.BZ2File(doc, 'rb')
                xmldata = ET.iterparse(infile, events=['start', 'end'])
                _, xroot = next(xmldata)
                for event, element in xmldata:
                    if genre != 'all':
                        if element.tag == 'text' and event == 'end':
                            noparse = False
                        if element.tag == 'text' and event == 'start':
                            if 'genre' in element.attrib:
                                if element.attrib['genre'] != genre:
                                    noparse = True
                                    continue
                    if noparse == True:
                        continue
                    if element.tag == 'sentence' and event == 'end':
                        if mwe:
                            sent = contract.check_mwe(element, mode)
                        else:
                            sent = extract_bw(element)
                        if sent == None:
                            continue
                        if first:
                            sent = [s.split('|')[0] for s in sent if s]
                        else:
                            sent = [s for s in sent if s]
                        outstring = ' '.join(sent)
                        with open(outfile, "a+") as f:
                            f.write(outstring + '\n')
                    xroot.clear()


if __name__ == '__main__':
    directory = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['plain', 'lemma', 'saldo', 'lex'],
                        default="plain")
    parser.add_argument('--mwe', action="store_true", default=False)
    parser.add_argument('--first-only', action="store_true", default=False)
    parser.add_argument('--genre', choices=['fiction', 'government', 'news',
                        'science', 'socialmedia', 'all'], default="all")
    parser.add_argument('outfile')
    args = parser.parse_args()
    mode = args.mode
    mwe = args.mwe
    genre = args.genre
    first = args.first_only
    outfile = directory + '/' + args.outfile
    if mode == 'plain' and mwe == True:
        print('Multi-Word Expressions are not available for plain mode\n')
        parser.print_help()
    elif mode == 'plain' and first == True:
        print('First-only is not available for plain mode\n')
        parser.print_help()
    else:
        process_dir(directory)