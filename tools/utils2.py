import json
import numpy as np
import os
from shutil import copyfile


def rewrite_dataset(old_list, new_list):
    lines = [x for x in map(str.strip, open(old_list).readlines())]
    with open(new_list,"w", encoding="utf8") as f:
        cnt_ok = 0
        cnt_no = 0
        cnt_new = 0
        print(len(lines))
        print(len(set(lines)))
        for line in lines:
            sample_path, label, tmp1 , tmp2 = line.split()
            #sample_path_split = sample_path.split('/')
            #for i in range(len(sample_path_split)):
            #          fsize += get_FileSize(sample_path)  
            fname = '.'.join(os.path.basename(sample_path).split('.')[:-1])
            pre = os.path.dirname(sample_path)
            new_pre = '/home/kezhiying'
            if os.path.exists(sample_path):
                cnt_ok += 1
            else:
                cnt_no += 1
                if os.path.exists(new_pre+sample_path):
                    cnt_new += 1
                    f.writelines([new_pre+sample_path, " ", str(label)," ", str(tmp1)," ", str(tmp2),"\n"])
                else:
                    print(sample_path)       
        print("ok: ", cnt_ok)
        print("no: ", cnt_no)
        print("new: ", cnt_new)

def check_dataset_exist(list_path):
    lines = map(str.strip, open(list_path).readlines())
    ok = 0
    no = 0
    json_ok = 0
    json_npy_ok = 0
    npy_ok = 0
    cnt  = 0
    for line in lines:
        print(cnt)
        cnt+=1
        sample_path, label, _, _ = line.split()
        fname = os.path.basename(sample_path)
        pre = os.path.dirname(sample_path)
        if os.path.exists(sample_path):
            ok += 1
            #if not os.path.exists('/home/kezhiying/ffac_data'+pre):
            #   os.makedirs('/home/kezhiying/ffac_data'+pre)
            #copyfile(sample_path, '/home/kezhiying/ffac_data'+sample_path)
            if os.path.exists(sample_path[:-4]+'.json'):
                json_ok += 10
                with open(sample_path[:-4]+'.json',  "r") as f:
                    a = json.load(f)
                #print(a)
                #os.remove('/home/kezhiying'+sample_path[:-4]+'.json')
                #copyfile('/home/kezhiying/ffac_data2'+sample_path[:-4]+'.json', '/home/kezhiying'+sample_path[:-4]+'.json')
                if os.path.exists(sample_path[:-4]+'_68.npy'):
                    json_npy_ok += 1
            if os.path.exists(sample_path[:-4]+'_68.npy'):
                npy_ok += 1
                a = np.load(sample_path[:-4]+'_68.npy')
                #print(a)
                #os.remove('/home/kezhiying'+sample_path[:-4]+'_68.npy')
                #copyfile(sample_path, '/home/kezhiying/ffac_data'+sample_path[:-4]+'_68.npy')
        else:
            no += 1
    print("total:", ok+no)
    print("ok:", ok)
    print("no:", no)
    print("json_ok:", json_ok)
    print("json_68_ok:", json_npy_ok)
    print("68_ok:", npy_ok)


if __name__ == '__main__':
    list_path = '/home/kezhiying/df_project/df_datasets_list/ffac_24/new_0.txt'
    check_dataset_exist(list_path)
