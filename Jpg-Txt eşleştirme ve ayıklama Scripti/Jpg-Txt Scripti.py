#!/usr/bin/env python
# coding: utf-8

import os
os.listdir()

get_ipython().run_line_magic('cd', 'xy')

pwd

os.chdir('desktop')
os.chdir('datay')
yemek_liste = [i for i in os.listdir()]
yemek_liste
os.chdir('../temiz_xy')
for i in yemek_liste:
    os.mkdir(i)

os.chdir('..')

os.listdir()


get_ipython().run_line_magic('cd', 'data_temizy')


r'"C:\Users\Berk\Desktop\datay\\'+ '\\'

for i in yemek_liste:
    
    data_path = 'C:\\Users\\Berk\\Desktop\\xy\\'+ i+ '\\'
    print(data_path)
    
    data_path_2 = 'C:\\Users\\Berk\\Desktop\\temiz_xy\\'+i+ '\\'
    print(data_path_2)
    txt_list = [i for i in os.listdir(data_path) if i.endswith('.txt')]
    
    for k in txt_list:
        
        txt = k
        
        jpg = k.replace('txt','jpeg')
        
        
    
        command = 'copy "'+ data_path+txt+ '"'+' "'+data_path_2+txt+ '"'
        command_2 = 'copy "'+ data_path+jpg +'"'+' "'+data_path_2+jpg+ '"'
        print(command)
        print(command_2)
    
    
        os.system(command)

        os.system(command_2)






