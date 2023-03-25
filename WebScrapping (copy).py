import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

import time,pandas

start = 0
end = 6000

url='https://www.backmarket.es/es-es/l/relojes-conectadoss/0894adca-7735-40d3-a34b-5a77358e3937?page='
driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
data = {}
deviceId = 0
for i in range(1,28):
    response = requests.get(url+str(i))
    soup = BeautifulSoup(response.text, 'html.parser')
    devices = []
    for link in soup.find_all('a'):
        if link.get('class') and link.get('class')[0]=='focus:outline-none':
            devices.append(link.get('href'))
    print(data)
    df = pandas.DataFrame.from_dict(data)
    df.to_excel('backmarketWatch.xlsx',index=False)
    print(f"MAS DATOS + {str(i)}")
    for device in devices:
        deviceSpecifications = {}
        urlDevice = 'https://www.backmarket.es'+device
        print(urlDevice)
        driver.get(urlDevice)
        try:
            driver.find_element("xpath", '/html/body/div[1]/div/div/div/div[1]/div[2]/section/div/div/div[2]/button[3]/div').click()
        except:
            pass
        time.sleep(5)
        try:
            try:
                driver.find_elements("xpath",'/html/body/div[1]/div/div/div/div/div[2]/div/div[2]/div/div/div[2]/div/div/div[6]/ul/li[1]/button')[0].click()
            except:
                try:
                    driver.find_elements("xpath",'/html/body/div[1]/div/div/div/div/div[2]/div/div[2]/div/div/div[2]/div/div/div[7]/ul/li[1]/button')[0].click()
                except:
                    continue
            #driver.find_element("xpath", '/html/body/div[1]/div/div/div/div/div[2]/div/div[2]/div/div/div[2]/div/div/div[5]/ul/li[1]/button').click()
            #WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div[8]/div[2]/section/div")))
            time.sleep(1)
            i = 1
            while 1:
                try:
                    name = driver.find_element("xpath", f'/html/body/div[2]/div[10]/div[2]/section/div/div/div[2]/ul/li[{i}]/div/div[1]').text
                    value = driver.find_element("xpath", f'/html/body/div[2]/div[10]/div[2]/section/div/div/div[2]/ul/li[{i}]/div/div[2]').text
                    deviceSpecifications[name] = value
                    i = i+1
                except:
                    try:
                        name = driver.find_element("xpath", f'/html/body/div[2]/div[8]/div[2]/section/div/div/div[2]/ul/li[{i}]/div/div[1]').text
                        value = driver.find_element("xpath", f'/html/body/div[2]/div[8]/div[2]/section/div/div/div[2]/ul/li[{i}]/div/div[2]').text
                        deviceSpecifications[name] = value
                        i = i+1
                    except:
                        try:
                            name = driver.find_element("xpath", f'/html/body/div[2]/div[9]/div[2]/section/div/div/div[2]/ul/li[{i}]/div/div[1]').text
                            value = driver.find_element("xpath", f'/html/body/div[2]/div[9]/div[2]/section/div/div/div[2]/ul/li[{i}]/div/div[2]').text
                            deviceSpecifications[name] = value
                            i = i+1
                        except:
                            break
            
            try:
                deviceSpecifications['Correcto']= driver.find_element("xpath", '/html/body/div[1]/div/div/div/div/div[2]/div/div[2]/div/div/div[2]/div/div/div[3]/div/div/div/ul/li[1]/a/div/div[2]').text
            except:
                pass
            try:
                deviceSpecifications['Muy bueno']= driver.find_element("xpath", '/html/body/div[1]/div/div/div/div/div[2]/div/div[2]/div/div/div[2]/div/div/div[3]/div/div/div/ul/li[2]/a/div/div[2]').text
            except:
                pass
            try:
                deviceSpecifications['Excelente']= driver.find_element("xpath", '/html/body/div[1]/div/div/div/div/div[2]/div/div[2]/div/div/div[2]/div/div/div[3]/div/div/div/ul/li[3]/a/div/div[2]').text
            except:
                pass
            
            for key,value in data.items():
                if key in deviceSpecifications:
                    data[key].append(deviceSpecifications[key])
                else:
                    data[key].append('nan')
            
            for key, value in deviceSpecifications.items():
                if not key in data:
                    data[key] = ['nan']*deviceId
                    data[key].append(value)
            deviceId+=1
        except:
            continue

df = pandas.DataFrame.from_dict(data)
df.to_excel('backmarket.xlsx',index=False)

quit()

devices = []

workbook = Workbook()
sheet = workbook.active

for link in soup.find_all('a'):
    if link.get('class') and link.get('class')[0]=='link':
        devices.append(link.get('href'))
     
row = 2
first = True

toselect = ['capacidad:','sistema operativo:','marca:','modelo:','pulgadas:','megapixels:','memoria ram:','color:']
for device in devices:
    try:
        response = requests.get("https://www.cashconverters.es"+device)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('li'):
            if link.get('class') and link.get('class')[0]=='attribute-values':
                if link.find_all('span')[0].get_text() in toselect:
                    column=chr(ord('A')+toselect.index(link.find_all('span')[0].get_text()))
                    if first:
                        sheet[column+str(1)]= link.find_all('span')[0].get_text()[0:len(link.find_all('span')[0].get_text())-1]
                    try:
                        sheet[column+str(row)] = link.find_all('a')[0].get_text()
                    except:
                        sheet[column+str(row)] = link.find_all('span')[1].get_text()
        if first:
            sheet[chr(ord('A')+len(toselect))+str(1)]='price'
            sheet[chr(ord('A')+len(toselect)+1)+str(1)]='condition'
            first = False
        for link in soup.find_all('span'):
            if link.get('class') and link.get('class')[0]=='principal':
                price = float(link.get_text()[0:len(link.get_text())-2].replace('.','').replace(',','.'))
                sheet[chr(ord('A')+len(toselect))+str(row)] = price
                break
        for link in soup.find_all('p'):
            if link.get('class') and link.get('class')[0]=='state-title':
                condition = link.find_all('span')[0].get_text()
                sheet[chr(ord('A')+len(toselect)+1)+str(row)] = condition
                break
        row=row+1
        print(row-1)
        workbook.save("6000Entries.xlsx")
    except:
        pass

