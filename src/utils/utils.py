import requests
from bs4 import BeautifulSoup


def get_url_paragraphs(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all <p> tags and extract their text
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    return " ".join(paragraphs)
