#function to extract wikipedia all revision timestamps for each page, and keep the first timestamp
import requests
from collections import OrderedDict

def get_revision_timestamps(title):
    try:
        # Set the API URL
        url = 'https://en.wikipedia.org/w/api.php'

        # Set the parameters for the API query
        params = {
            'action': 'query',
            'prop': 'revisions',
            'titles': title,
            'rvprop': 'timestamp|ids',
            'rvlimit': 'max',
            'rvdir': 'newer',
            'format': 'json',
            'formatversion': '2'
        }

        # Send the API request and retrieve the revision data
        rev_dict = OrderedDict()
        rvcontinue = None
        while True:
            if rvcontinue is not None:
                params['rvcontinue'] = rvcontinue
            response = requests.get(url, params=params)
            
            data = response.json()
        
            pages = data['query']['pages']
            
            for page in pages:
                revisions = page['revisions']
                for rev in revisions:
                    rev_dict[rev['timestamp']] = rev['revid']

            if 'continue' not in data:
                break
            rvcontinue = data['continue']['rvcontinue']

        return rev_dict
    
    except:
        # Handle exceptions here
        pass


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import ConnectionError

import re
from bs4 import BeautifulSoup
def clean_wiki_article(url):
    try:
        # Get the raw HTML content of the Wikipedia article
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        response =  session.get(url)
        html = response.content

        # Use BeautifulSoup to parse the HTML and extract only the main body of text
        soup = BeautifulSoup(html, 'html.parser')
        text = ""
        for p in soup.find_all('p'):
            text += p.text
        
        # Remove any citation numbers and references from the text
        text = re.sub(r'\[[0-9]*\]', '', text)
        text = re.sub(r'\[(.*?)\]', '', text)

        # Remove any newlines and extra spaces from the text
        text = text.replace('\n', ' ')
        text = re.sub(' +', ' ', text)
        
        # Return the cleaned text
        return text.strip()
    
    except ConnectionError:
        pass


def add_space_after_full_stop(text):
    new_text = ''
    for i, char in enumerate(text):
        if char == '.':
            if i+1 < len(text) and text[i+1] != ' ':
                new_text += '. '
            else:
                new_text += char
        else:
            new_text += char
    return new_text



def remove_old_revision(text):
    start = "This is an old revision of this page,"
    end = "which may differ significantly from the current revision."
    while start in text and end in text:
        start_idx = text.index(start)
        end_idx = text.index(end, start_idx) + len(end)
        text = text[:start_idx] + text[end_idx:]

    return text

def remove_current_revision(text):
    start = "This is the current revision of this page,"
    end = "The present address (URL) is a permanent link to this version."
    while start in text and end in text:
        start_idx = text.index(start)
        end_idx = text.index(end, start_idx) + len(end)
        text = text[:start_idx] + text[end_idx:]

    return text