from selenium import webdriver
import time
import requests
import shutil
from PIL import Image
import os, sys
import pandas as pd

driver_path = '/home/rey/Downloads/chromedriver'
driver = webdriver.Chrome(executable_path=driver_path)


iterate = 1


def resize(file):
    if os.path.isfile(file):
        print('yooo')
        im = Image.open(file)
        f, e = os.path.splitext(file)
        imResize = im.resize((400,400), Image.ANTIALIAS)
        imResize.save(f + '.jpg', 'JPEG', quality=90)

def save_img(symbol, img, i):
    directory = '/home/rey/coursework/ML/ML_clouds/get_data/images/'
    save_dir = os.path.join(directory, symbol)
    try:
        filename = symbol+str(i)+'.jpg'
        filename = filename.replace(' ', '_')
        try:
            os.mkdir(save_dir)
        except:
            pass
        image_path = os.path.join(save_dir, filename)
        response = requests.get(img,stream=True)
        with open(image_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        print('image saved!')
    except Exception as e:
        print(e)
        pass

def find_urls(inp, url, driver, iterate):
    driver.get(url)
    time.sleep(3)
    for j in range (1,iterate+1):
        imgurl = driver.find_element_by_xpath('//div//div//div//div//div//div//div//div//div//div['+str(j)+']//a[1]//div[1]//img[1]')
        imgurl.click()
        time.sleep(5)
        img = driver.find_element_by_xpath('//body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img')
        img = img.get_attribute("src")
        save_img(inp, img, j)



def resize():
    path = '/home/rey/coursework/ML/ML_clouds/get_data/images/'
    dirs = os.listdir( path )
    for subdir in dirs:
        for item in subdir:
            cloud_path = os.path.join(path,subdir)
            if os.path.isfile(cloud_path+item):
                im = Image.open(cloud_path+item)
                f, e = os.path.splitext(cloud_path+item)
                imResize = im.resize((400,400), Image.ANTIALIAS)
                imResize.save(f + '.jpg', 'JPEG', quality=90)


def get_name():
    df = pd.read_csv('../descriptions.csv', sep='; ', engine='python')
    for index, row in df.iterrows():
        symbol, name = row['Symbol'], row['Full_Name']
        google_inp = name + 'clouds'
        url = 'https://www.google.com/search?q='+str(google_inp)+'&source=lnms&tbm=isch&sa=X&ved=2ahUKEwie44_AnqLpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947'
        find_urls(symbol, url, driver, iterate)


get_name()
resize()


