import json
import os
import time
import requests
import urllib
import magic
import progressbar
from urllib.parse import quote
import re
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

def __init__():
    pass

def val_dir(directory_name):
    # Replace invalid characters with an underscore
    invalid_chars = r'[\\/:*?"<>| ]'
    sanitized_name = re.sub(invalid_chars, '_', directory_name)
    return sanitized_name

def _create_directories(_directory, _name):
    """
    Create directory to save images
    :param _directory:
    :param _name:
    :return:
    """
    name = _name.replace(" ", "_")
    try:
        if not os.path.exists(_directory):
            os.makedirs(_directory)
            time.sleep(0.2)
            path = name
            sub_directory = os.path.join(_directory, path)
            if not os.path.exists(sub_directory):
                os.makedirs(sub_directory)
        else:
            path = name
            sub_directory = os.path.join(_directory, path)
            if not os.path.exists(sub_directory):
                os.makedirs(sub_directory)

    except OSError as e:
        if e.errno != 17:
            raise
        pass
    return

def _download_page(url):
    try:
        headers={}
        headers['User-Agent']="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req)
        respData = str(resp.read())
        return respData
    except Exception as e:
        print(e)
        exit(0)

def get_image_captions(url):
    bs4 = BeautifulSoup(requests.get(url).text, 'html.parser')
    captions = []
    for caption in bs4.find_all('div', {'class': 'rg_meta'}):
        metadata = json.loads(caption.text)
        captions.append(metadata.get('pt', ''))
    return captions

def download(query, sites="", limit=10, directory='', license_type="", directory_name='', extensions={'.jpg', '.png', '.ico', '.gif', '.jpeg'}, min_width=0, min_height=0):
    """
    Download images from Google Images based on given keywords, with optional site restrictions and licensing filters.
    This function searches for images on Google Images using specified keywords and site restrictions. It can also apply
    licensing filters to the search. The downloaded images are saved in a specified directory, and metadata (URL, caption,
    and licensing information) is saved in a JSON file.

    - param query: query.
    - param sites: Comma-separated list of sites to restrict the search to (e.g., "instagram.com, flickr.com").
    - param limit: The maximum number of images to download per keyword.
    - param directory: The main directory where images and metadata will be saved. If empty, defaults to "images/".
    - param extensions: Set of acceptable image file extensions to download (default: {'.jpg', '.png', '.ico', '.gif', '.jpeg'}).
    - param license_type: Optional licensing filter (e.g., "reuse", "commercial_reuse", "noncommercial_reuse", "reuse_with_modification", "publicdomain").
    - param min_width: Minimum width of the images to be downloaded.
    - param min_height: Minimum height of the images to be downloaded.
    - return: None
    """
    keyword_to_search = [query]
    sites_to_search = [str(item).strip() for item in sites.split(',')]
    if directory != '':
        main_directory = directory
    else:
        main_directory = "images/"

    i = 0

    things = len(keyword_to_search) * limit

    bar = progressbar.ProgressBar(maxval=things, \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()

    sites_to_search = sites_to_search
    if sites_to_search == ['']:
        site_restriction = ''
    else:
        site_restriction = " OR ".join([f"site:{site}" for site in sites_to_search])

    license_filters = {
        "reuse": "il:cl",
        "commercial_reuse": "il:cl&as_rights=cc_commercial|cc_publicdomain",
        "noncommercial_reuse": "il:cl&as_rights=cc_noncommercial",
        "reuse_with_modification": "il:cl&as_rights=cc_modifiable",
        "publicdomain": "il:cl&as_rights=cc_publicdomain"
    }

    tbs_param = license_filters.get(license_type, "")

    directory_name = directory_name

    while i < len(keyword_to_search):
        _create_directories(main_directory, val_dir(directory_name))
        full_query = f"{keyword_to_search[i]} {site_restriction}"
    
        encoded_query = quote(full_query.encode('utf-8'))
        
        url = f'https://www.google.com/search?q={encoded_query}&biw=1536&bih=674&tbm=isch&{tbs_param}&source=lnms&sa=X&ved=0ahUKEwioj8jwiMLnAhW9AhAIHbXTBMMQ_AUI3QUoAQ'
        print(url)
        raw_html = _download_page(url)
        captions = get_image_captions(url)
        print(captions)
        end_object = -1
        google_image_seen = False
        j = 0
        while j < limit:
            while True:
                try:
                    new_line = raw_html.find('"https://', end_object + 1)
                    end_object = raw_html.find('"', new_line + 1)

                    buffor = raw_html.find('\\', new_line + 1, end_object)
                    if buffor != -1:
                        object_raw = (raw_html[new_line + 1:buffor])
                    else:
                        object_raw = (raw_html[new_line + 1:end_object])

                    if any(extension in object_raw for extension in extensions):
                        break

                except Exception as e:
                    break
            path = os.path.join(main_directory, val_dir(directory_name))
            try:
                if 'gstatic' in object_raw:
                    raise ValueError()
                r = requests.get(object_raw, allow_redirects=True, timeout=1)
                if ('html' not in str(r.content)):
                    mime = magic.Magic(mime=True)
                    file_type = mime.from_buffer(r.content)
                    file_extension = f'.{file_type.split("/")[1]}'
                    if file_extension not in extensions:
                        raise ValueError()
                    if file_extension == '.png' and not google_image_seen:
                        google_image_seen = True
                        raise ValueError()

                    # Check image resolution
                    img = Image.open(BytesIO(r.content))
                    width, height = img.size
                    if width < min_width or height < min_height:
                        raise ValueError(f"Image does not meet resolution requirements: {width}x{height}")

                    file_name = str(keyword_to_search[i]) + "_" + str(j + 1) + file_extension
                    image_name = os.path.basename(object_raw)
                    image_name = re.sub(r'[^a-zA-Z0-9]', '_', image_name)
                    image_name = os.path.splitext(image_name)[0] + file_extension
                    info = {
                        'query': query,
                        'url': object_raw,
                        'file_name': image_name,
                        'path': path,
                    }
                    json.dump(info, open(f'{path}/{image_name}.json', 'w'))
                    with open(os.path.join(path, image_name), 'wb') as file:
                        file.write(r.content)
                    bar.update(bar.currval + 1)
                else:
                    j -= 1
            except Exception as e:
                # print(e)
                j -= 1
            j += 1

        i += 1
    bar.finish()


'''
Sample usage
download(
    query='Sri lankan sinhala Social customs, etiquette, traditional greetings, and cultural norms.',
    sites='', 
    license_type="reuse", 
    limit=10, 
    directory='images/',
    directory_name='sri lankan sinhala Social customs, etiquette, traditional greetings, and cultural norms.',
    min_height=480,
    min_width=640,
)
'''