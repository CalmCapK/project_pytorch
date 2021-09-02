import cv2
import numpy as np

import os
import shutil

import csv
def read_csv():
    csvFile = open("./result/test_ans_model.csv", "r")
    reader = csv.reader(csvFile)
    result = {}
    path = "/PublicData/kzy_data/ok/tg3/"
    out = '/PublicData/kzy_data/ok/result_tg3/'
    folder_list = ['normal', 'abnormal', 'poster']
    #folder_list = ['01_report', '02_post','03_unknown','04_place','05_map','06_joke','07_poster','08_chat','09_comics', '10_other', '11_other_related'    ,'12_form','13_word','14_arms','15_person','16_epidemic','17_procession']
    for item in reader:
        # 忽略第一行
        #if reader.line_num == 1:
        #    continue
        filename = item[0].split('/')[-1]
        print(filename)
        print(item[2])
        fold = folder_list[int(item[2])]
        if not os.path.exists(out+fold):
            os.makedirs(out+fold)
        shutil.copyfile(path+filename, out+fold+'/'+filename)
        #break


def remove_repeat():
    path = "./origin2/" #文件夹目录
    files = os.listdir(path) #得到文件夹下的所有文件名称
    pk = "./test1/"
    pks = os.listdir(pk)
    ans_path = './ans2/'
    err_path = './err2/'
    repeat_path = './repeat2/'

    n = len(files)
    vis=[0 for i in range(n)]
    print(n)
    m = len(pks)
    print(m)
    for i in range(0, m):
        print(i)
        for j in range(0, n):
            if(vis[j] != 0):
                continue
            image1 = cv2.imread(pk+pks[i])
            if image1 is None:
                continue
            image2 = cv2.imread(path+files[j])
            if image2 is None: 
                vis[j] = 2
                shutil.copyfile(path+files[j], err_path+files[j])
                continue
            if(image1.shape != image2.shape):
                continue
            difference = cv2.subtract(image1, image2)
            result = not np.any(difference) 
            if result is True:
                vis[j] = 1
                shutil.copyfile(path+files[j], repeat_path+files[j])
    
    for j in range(0, n):
        #if vis[j] == 2:
        #    shutil.copyfile(path+files[j], err_path+files[j])
        #elif vis[j] == 1:
        #    shutil.copyfile(path+files[j], repeat_path+files[j])
        #else:
        if vis[j] == 0:
            shutil.copyfile(path+files[j], ans_path+files[j])

    print('done')
    
def check():
    path = "/PublicData/center_data/1327536648/1001G327536648/"
    err_path = "//PublicData/center_data/1327536648/err/"
    success_path = "/PublicData/center_data/1327536648/ok/"
    files = os.listdir(path)
    n = len(files)
    print(n)
    for i in range(0, n):
        print(i)
        image = cv2.imread(path+files[i])
        if image is None:
            shutil.copyfile(path+files[i], err_path+files[i])
        else:
            shutil.copyfile(path+files[i], success_path+files[i])
    

def pk():
    #path = "./bbbb/" #文件夹目录
    path = "./origin2/" #文件夹目录
    files = os.listdir(path) #得到文件夹下的所有文件名称
    ans = []
    err = []
    ans_path = './ans3/'
    err_path = './err3/'
    n = len(files)
    vis=[0 for i in range(n)]
    print(n)
    #print(files[0])
    for i in range(0, n):
        print(i)
        if(vis[i] == 1):
            continue
        vis[i] = 1
        ans.append(files[i])
        image1 = cv2.imread(path+files[i])
        if image1 is None:
            continue
        shutil.copyfile(path+files[i], ans_path+files[i])
        for j in range(i,n):
            if(vis[j] == 1):
                continue
            image2 = cv2.imread(path+files[j])
            if image2 is None: 
                vis[j] = 1
                err.append(files[j])
                shutil.copyfile(path+files[j], err_path+files[j])
                continue
            #print('----------------')
            #print(files[i])
            #print(files[j])
            #print(image1.shape)
            #print(image2.shape)
        
            if(image1.shape != image2.shape):
                continue
            difference = cv2.subtract(image1, image2)
            result = not np.any(difference) 
            #print(i)
            #print(result)
            if result is True:
                vis[j] = 1
                err.append(files[j])
                shutil.copyfile(path+files[j], err_path+files[j])
    print('----len----')
    print(len(ans))
    print(len(err))
    print('finish')

def move():
    path = './ans/'
    total = "./aaa/"
    files = os.listdir(path) #得到文件夹下的所有文件名称
    for f in files:
        os.remove(total+f)

import hashlib

def getmd5(filename):
    file_txt = open(filename,'rb').read()
    m = hashlib.md5(file_txt)
    return m.hexdigest()

def rrr(path):
    all_size = {}
    total_file = 0
    total_delete = 0
    for root,dirs,files in os.walk(path):
        for file in files:
            real_path = os.path.join(root,file)
            total_file += 1
            image = cv2.imread(real_path)
            if image is None:
                os.remove(real_path)
                print('删除', file)
                continue
            if os.path.isfile(real_path):
                size = os.stat(real_path).st_size
                name_and_md5 = [real_path, '']
                if size in all_size.keys():
                    new_md5 = getmd5(real_path)
                    if all_size[size][1] == '':
                        all_size[size][1] = getmd5(all_size[size][0])
                    if new_md5 in all_size[size]:
                        os.remove(real_path)
                        #shutil.copyfile(real_path, "./chongfu/"+file)
                        print('删除', file)
                        total_delete += 1
                    else:
                        all_size[size].append(new_md5)
                else:
                    all_size[size] = name_and_md5
    print ('文件个数：', total_file)
    print ('删除个数：', total_delete)

if __name__ == '__main__':
    #check()
    #move()
    #pk()
    #remove_repeat()
    read_csv()
    #rrr('/PublicData/center_data/1327536648/ok/')
    #pk()
    

    
