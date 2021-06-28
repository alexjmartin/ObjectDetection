import numpy as np
import os
import PIL
import PIL.Image
import pathlib

import urllib3
import requests
from bs4 import BeautifulSoup
import json

# Input your own access key to unsplash
ACCESS_KEY = '{ACCESS_KEY}'

# 10 images are returned per page by default, specify which page to start extracting images from 
start_page = 1

# Specify which page to stop extracting images from - all images between start page and end page will be downloaded
end_page = 50

#Specify what you'd like to search unsplah for - in this case it's F1 car pictures
query_string = 'F1 car'

# Specify where to download images to
download_loc = '{Set Path to Download images}'

img_dic = {}

for p in range(start_page,end_page+1):
    r = requests.get(f'https://api.unsplash.com/search/photos?page={p}&query={query_string}&client_id={ACCESS_KEY}')
    soup = BeautifulSoup(r.text)
    string = soup.find("p").contents[0].string
    json_response = json.loads(string)
    print(f'Extracting images for page {p}')
    
    for i in json_response['results']:
        img_dic[i['id']] = {'created':i['created_at'], 'desc':i['description'], 'url':i['urls']['regular'], 'tags':i['tags']}
    
    for key, m in img_dic.items():
        img = requests.get(m['url'])
        file = open(f"{download_loc}{key}.jpeg", "wb")
        file.write(img.content)
        file.close()
        
print(f'{len(img_dic)} images extracted to {download_loc}')
