import argparse
import os
import random
from shutil import copyfile
import sys
import yaml

sys.path.append(".")
from tools.utils import remove_makedir, printProgressBar

ALLOWED_EXTENSIONS = set(['.png', '.jpg', '.JPG', '.PNG',
                         '.jpeg', '.JPEG', '.gif', '.GIF', '.bmp', '.BMP'])


def merge_dataset_list(list_paths, final_list):
    datas = []
    for list_path in list_paths:
        lines = map(str.strip, open(list_path).readlines())
        for line in lines:
            datas.append(line)
    with open(final_list, "w") as f:
        for data in datas:
            f.writelines([data, "\n"])

def write_dataset_list_without_label(dir_path, list_path):
    objs = os.listdir(dir_path)
    files = []
    for obj in objs:
        if not os.path.isdir(obj):
            ext = os.path.splitext(obj)[-1]
            if ext in ALLOWED_EXTENSIONS:
                files.append(dir_path + "/" + obj)
            else:
                print("{} is not in allowed_extensions".format(obj))
        else:
            print("{} is a dir".format(obj))
    print(len(files))
    with open(list_path, "w") as f:
        for file in files:
            f.writelines([file, "\n"])


def write_dataset_list_with_label(dir_path, list_path, label):
    objs = os.listdir(dir_path)
    files = []
    for obj in objs:
        if not os.path.isdir(obj):
            ext = os.path.splitext(obj)[-1]
            if ext in ALLOWED_EXTENSIONS:
                files.append(dir_path + "/" + obj)
            else:
                print("{} is not in allowed_extensions".format(obj))
        else:
            print("{} is a dir".format(obj))
    print(len(files))
    with open(list_path, "w") as f:
        for file in files:
            f.writelines([file, " ", str(label), "\n"])


def split_dataset(config):
    remove_makedir(config.train_path)
    remove_makedir(config.valid_path)
    remove_makedir(config.test_path)

    filenames = os.listdir(config.origin_path)
    data_list = []
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext not in ALLOWED_EXTENSIONS:
            print(ext)
        else:
            data_list.append(filename)

    total_num = len(data_list)
    train_num = int((config.train_ratio/(config.train_ratio +
                    config.valid_ratio+config.test_ratio))*total_num)
    valid_num = int((config.valid_ratio/(config.train_ratio +
                    config.valid_ratio+config.test_ratio))*total_num)
    test_num = total_num - train_num - valid_num

    print('Num of total set : ', total_num)
    print('Num of train set : ', train_num)
    print('Num of valid set : ', valid_num)
    print('Num of test  set : ', test_num)

    Arange = list(range(total_num))
    random.shuffle(Arange)

    for i in range(train_num):
        idx = Arange.pop()
        filename = data_list[idx]
        src = os.path.join(config.origin_path, filename)
        dst = os.path.join(config.train_path, filename)
        copyfile(src, dst)
        printProgressBar(
            i + 1, train_num, prefix='Producing train set:', suffix='Complete', length=50)

    for i in range(valid_num):
        idx = Arange.pop()
        filename = data_list[idx]
        src = os.path.join(config.origin_path, filename)
        dst = os.path.join(config.valid_path, filename)
        copyfile(src, dst)
        printProgressBar(
            i + 1, valid_num, prefix='Producing valid set:', suffix='Complete', length=50)

    for i in range(test_num):
        idx = Arange.pop()
        filename = data_list[idx]
        src = os.path.join(config.origin_path, filename)
        dst = os.path.join(config.test_path, filename)
        copyfile(src, dst)
        printProgressBar(
            i + 1, test_num, prefix='Producing test set:', suffix='Complete', length=50)


def split_dataset_list(config):
    lines = map(str.strip, open(config.origin_list_path).readlines())
    data_list = []
    for line in lines:
        data_list.append(line)

    total_num = len(data_list)
    train_num = int((config.train_ratio/(config.train_ratio +
                    config.valid_ratio+config.test_ratio))*total_num)
    valid_num = int((config.valid_ratio/(config.train_ratio +
                    config.valid_ratio+config.test_ratio))*total_num)
    test_num = total_num - train_num - valid_num

    print('Num of total set : ', total_num)
    print('Num of train set : ', train_num)
    print('Num of valid set : ', valid_num)
    print('Num of test  set : ', test_num)

    Arange = list(range(total_num))
    random.shuffle(Arange)

    with open(config.train_list_path, "w") as f:
        for i in range(train_num):
            idx = Arange.pop()
            filename = data_list[idx]
            f.writelines([filename, "\n"])
    with open(config.valid_list_path, "w") as f:
        for i in range(valid_num):
            idx = Arange.pop()
            filename = data_list[idx]
            f.writelines([filename, "\n"])
    with open(config.test_list_path, "w") as f:
        for i in range(test_num):
            idx = Arange.pop()
            filename = data_list[idx]
            f.writelines([filename, "\n"])


def process_list(config):
    origin_list_path = config.origin_list_path
    train_list_path = config.train_list_path
    valid_list_path = config.valid_list_path
    test_list_path = config.test_list_path
    origin_list_paths = []
    train_list_paths = []
    valid_list_paths = []
    test_list_paths = []
    for i in range(len(config.categorys)):
        print('\ndeal with: ', config.categorys[i])
        config.origin_list_path = origin_list_path.split(
            '.')[0]+'_' + config.categorys[i] + '.txt'
        config.train_list_path = train_list_path.split(
            '.')[0]+'_' + config.categorys[i] + '.txt'
        config.valid_list_path = valid_list_path.split(
            '.')[0]+'_' + config.categorys[i] + '.txt'
        config.test_list_path = test_list_path.split(
            '.')[0]+'_' + config.categorys[i] + '.txt'
        dir_path = config.origin_path+'/'+config.categorys[i]
        write_dataset_list_with_label(dir_path, config.origin_list_path, i)
        split_dataset_list(config)
        origin_list_paths.append(config.origin_list_path)
        train_list_paths.append(config.train_list_path)
        valid_list_paths.append(config.valid_list_path)
        test_list_paths.append(config.test_list_path)
    merge_dataset_list(origin_list_paths, origin_list_path)
    merge_dataset_list(train_list_paths, train_list_path)
    merge_dataset_list(valid_list_paths, valid_list_path)
    merge_dataset_list(test_list_paths, test_list_path)


def process_file(config):
    origin_path = config.origin_path
    train_path = config.train_path
    valid_path = config.valid_path
    test_path = config.test_path
    for category in config.categorys:
        print('\ndeal with: ', category)
        config.origin_path = origin_path+"/"+category
        config.train_path = train_path+"/"+category
        config.valid_path = valid_path+"/"+category
        config.test_path = test_path+"/"+category
        split_dataset(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./configs/create_dataset_config.yaml')
    args = parser.parse_args()
    with open(args.config_file) as f:
        config = yaml.load(f)
    # process_file(argparse.Namespace(**config))
    process_list(argparse.Namespace(**config))
