import requests
import re
from bs4 import BeautifulSoup
import pickle

url = 'https://sv.wikipedia.org/wiki/Wikipedia:Lista_%C3%B6ver_vanliga_spr%C3%A5kfel'

result = requests.get(url)

soup = BeautifulSoup(result.content, 'html.parser')

or_pattern = re.compile(r'(el\.|eller|\.\.\.|;|/)', re.I)
parentheses_pattern = re.compile(r'(\(.*\)|\[.*\])')
prefix_pattern = re.compile(r'\s-.*')
suffix_pattern = re.compile(r'-.*\s')

mistakes = {}

def clean_wordlist(wordlist):
  clean_words = []
  for word in wordlist:
    word = re.sub(parentheses_pattern, '', word)
    word = re.sub(prefix_pattern, '', word)
    word = re.sub(suffix_pattern, '', word)
    word = re.split(or_pattern, word)[0].strip()
    if word:
      clean_words.append(word)
  return clean_words

for row in soup.find_all('tr'):
  cells = row.find_all('td')
  if len(cells) > 1:
    corrections = cells[1].text.split(',')
    clean_corrections = clean_wordlist(corrections)
    mistakes_list = cells[0].text.split(',')
    clean_mistakes = clean_wordlist(mistakes_list)
    for index, mistake in enumerate(clean_mistakes):
      # Mistake on wikipedia page :o
      if mistake == 'utryck' or mistake == 'nden':
        continue
      if len(clean_corrections) == 1:
        print('{} -> {}'.format(mistake, clean_corrections[0]))
        mistakes[mistake] = clean_corrections[0]
      elif len(clean_corrections) == len(clean_mistakes):
        #print(clean_corrections)
        print('{} -> {}'.format(mistake, clean_corrections[index]))
        mistakes[mistake] = clean_corrections[index]

with open('common_mistakes.pkl', 'ab') as p:
  print('Writing to pickle...')
  pickle.dump(mistakes, p)