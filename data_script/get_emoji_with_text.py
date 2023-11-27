import os
import requests
import bs4
from tqdm import tqdm

platform_list = ["huawei","classic","bubble","microsoftteams","animation",
                 "google","apple","microsoft","facebook","twitter","whatsapp",
                 "mozilla","emojione","emojitwo","openmoji","emojidex",
                 "blobmoji","samsung","htc","lg","docomo","softbank",
                 "aukddi","symbola","telegram","emojipedia","sample"]

for platform in platform_list:
    response = requests.get('https://www.emojiall.com/en/platform-'+platform)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    url_list=list(iter(soup.find_all('div', class_='emoji_card')))

    print("Install from "+platform+"...")
    for s in tqdm(url_list):
        name = s.find_all("a")[1].text

        path = "https://www.emojiall.com"+s.find('img')['data-src']

        if not os.path.exists('./emoji/'+platform):
            os.mkdir('./original_emoji/'+platform)

        with open('./original_emoji/'+platform+'/'+name+'.png', 'wb') as f:
            f.write(requests.get(path).content)

