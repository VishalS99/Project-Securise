from os import name
from selenium import webdriver
import time
import requests
from PIL import Image
import io
import os


def fetch_img_urls(query, thresh, wd,  sleep_between_interactions=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < thresh:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(
            f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= thresh:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls),
                  "image links, looking for more ...")
            time.sleep(30)
            return

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls


# fetch_img_urls("car number plate images", 1000, wd, 1)

def image_downloader(dir, url, name):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')

        folder_path = os.path.join(os.getcwd(), dir)
        print(folder_path)

        if os.path.exists(folder_path) == True:
            file_path = os.path.join(folder_path, name + '.jpg')
        else:
            os.mkdir(folder_path)
            file_path = os.path.join(folder_path, name + '.jpg')

        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


DRIVER_PATH = "./chromedriver"
wd = webdriver.Chrome(executable_path=DRIVER_PATH)
wd.get("https://google.com")

query = "number plates cars"
wd.get('https://google.com')
search_box = wd.find_element_by_css_selector('input.gLFyf')
search_box.send_keys(query)
links = fetch_img_urls(query, 100, wd)


for i in links:
    image_downloader(images_path, i)
wd.quit()
