#!/usr/bin/env python
# coding: utf-8

# In[3]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import time


# In[4]:


#pip install wget


# In[5]:


import os
import wget


# In[6]:


#specify the path to chromedriver.exe (download and save on your computer)
driver = webdriver.Chrome('C:/Users/Berk/Desktop/chromedriver_win32/chromedriver.exe')

#open the webpage
driver.get("http://www.instagram.com/")

username=WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.CSS_SELECTOR,"input[name='username']")))
password=WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.CSS_SELECTOR,"input[name='password']")))
username.clear()
password.clear()
username.send_keys("instagram kullanıcı adını giriniz.")
password.send_keys("instagram şifrenizi giriniz.")


# In[7]:


log_in=WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.CSS_SELECTOR,"button[type='submit']"))).click()


# In[8]:


simdi_degil=WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.XPATH,"//button[contains(text(),'Şimdi Değil')]"))).click()


# In[9]:


simdi_degil2=WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.XPATH,"//button[contains(text(),'Şimdi Değil')]"))).click()


# In[10]:


searchbox=WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Ara']")))
searchbox.clear()
keyword='#börek'
searchbox.send_keys(keyword)


# In[17]:


searchbox.send_keys(Keys.ENTER)


# In[26]:


driver.execute_script("window.scrollTo(0,4000);")
driver.execute_script("window.scrollTo(0,4000);")
images=driver.find_elements_by_tag_name('img')
images=[image.get_attribute('src') for image in images]
images


# In[21]:


path=os.getcwd()
path=os.path.join(path,keyword[1:]+ "s")
os.mkdir(path)



# In[22]:


path


# In[ ]:


import re


# In[27]:


counter=0
for image in images:    
    save_as=os.path.join(path,keyword[1:]+str(counter)+'.jpg')
    wget.download(image,save_as)
    counter+=1


# In[ ]:




