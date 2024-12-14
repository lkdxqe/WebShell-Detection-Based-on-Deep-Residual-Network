import os
import numpy as np
import imageio
import random
import chardet
from tqdm import tqdm

height = 128
width = 128
train_ratio = 0.8
language = "php"
benign_ = "white"
malicious_ = "black"
in_normal_dir = r"./dataset/" + language + "/" + benign_
in_webshell_dir = r"./dataset/" + language + "/" + malicious_
out_dir = r"./codeImage/" + language + "/" + "height" + str(height) + "/"

train_normal_dir = os.path.join(out_dir, 'train', benign_)
test_normal_dir = os.path.join(out_dir, 'test', benign_)
train_webshell_dir = os.path.join(out_dir, 'train', malicious_)
test_webshell_dir = os.path.join(out_dir, 'test', malicious_)


def txt_to_Array(path):
    arr = np.zeros((height, width))
    try:
        with open(path, 'rb') as file:
            data = file.read()
            file_encoding = chardet.detect(data).get('encoding')

        with open(path, 'r', encoding=file_encoding, errors='ignore') as file:
            lines = file.readlines()
            l = min(height, len(lines))
            lines = lines[:l]
            for i, line in enumerate(lines):
                w = min(width, len(line))
                line = line[:w]
                arr[i, :w] = [ord(cc) if ord(cc) != 32 else 0 for cc in line]

            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
    except Exception as e:
        print('[Error]', path, ' Error: ', e)
        return None


def convFileToImage(in_normal_dir, in_webshell_dir):
    os.makedirs(train_normal_dir, exist_ok=True)
    os.makedirs(test_normal_dir, exist_ok=True)
    os.makedirs(train_webshell_dir, exist_ok=True)
    os.makedirs(test_webshell_dir, exist_ok=True)

    webshell_files = [os.path.join(in_webshell_dir, f) for f in os.listdir(in_webshell_dir) if
                      os.path.isfile(os.path.join(in_webshell_dir, f))]
    normal_files = [os.path.join(in_normal_dir, f) for f in os.listdir(in_normal_dir) if
                    os.path.isfile(os.path.join(in_normal_dir, f))]
    webshell_num = len(webshell_files)
    benign_num = len(normal_files)
    if benign_num > 2 * webshell_num:
        normal_files = random.sample(normal_files, 2 * webshell_num)

    webshell_count = 0
    benign_count = 0

    for src_path in tqdm(webshell_files, desc="Processing webshell files", unit="file"):
        # if random.random() > 0.5:
        #     continue
        filename = os.path.splitext(os.path.basename(src_path))[0]
        folder = 'train' if random.random() < train_ratio else 'test'
        des_folder = train_webshell_dir if folder == 'train' else test_webshell_dir
        des_path = os.path.join(des_folder, filename + ".png")
        arr = txt_to_Array(src_path)
        if arr is None:
            continue
        imageio.imsave(des_path, arr)
        webshell_count += 1

    for src_path in tqdm(normal_files, desc="Processing normal files", unit="file"):
        # if random.random() > 0.5:
        #     continue
        filename = os.path.splitext(os.path.basename(src_path))[0]
        folder = 'train' if random.random() < train_ratio else 'test'
        des_folder = train_normal_dir if folder == 'train' else test_normal_dir
        des_path = os.path.join(des_folder, filename + ".png")
        arr = txt_to_Array(src_path)
        if arr is None:
            continue
        imageio.imsave(des_path, arr)
        benign_count += 1

    print(f"webshell_count:{webshell_count}")
    print(f"benign_count:{benign_count}")


convFileToImage(in_normal_dir, in_webshell_dir)
