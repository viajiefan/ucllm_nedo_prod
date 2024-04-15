# -*- coding: utf-8 -*-

import os
import argparse
import json
import time

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import datasets

def __search_idx_file(input_dir, filename_prefix:str):
    # print(os.listdir(input_dir))# debug
    
    for file in os.listdir(input_dir):
        if file.startswith(filename_prefix):
            return file
    print(f"Error not exist filename_prefix:{filename_prefix} in {input_dir}")
    exit()

def convert_parquet_to_json(file_path, out_path, num_proc=None):
    # loding parquet parquet -> pd -> datasets.Dataset
    s_time = time.time()
    df = pd.read_parquet(file_path)
    _dataset = datasets.Dataset.from_pandas(df)
    load_time = time.time() - s_time

    # convert to jsonl
    s_time = time.time()
    _dataset.to_json(out_path, batch_size=4096, num_proc=num_proc)
    convert_time = time.time() - s_time

    print(f"done [ {file_path.split('/')[-1]} ] load_time: {load_time} convert_time: {convert_time}")


def convert_refinedWeb():
    """ Convert parquet to jsonl (1fileごと)"""
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./tmp/output")
    args = parser.parse_args()
    
    start_idx = 0
    end_idx = 0
    num_proc = 16

    for idx in range(start_idx, end_idx+1):
        filename_prefix = f"train-{str(idx).zfill(5)}-of-05534-"
        parquet_filename = __search_idx_file(args.input_dir, filename_prefix)
        parquet_path = os.path.join(args.input_dir, parquet_filename)
        out_path = os.path.join(args.output_dir, parquet_filename.replace(".parquet", ".jsonl"))
        print(f"dealing ... parquet path: {parquet_path}")
        
        convert_parquet_to_json(parquet_path, out_path, num_proc=num_proc)

# ---------------------- multi process convert parquet2jsonl------------------------------------------
def convert_parquet_to_json_multi(inputs):
    file_path, out_path, num_proc = inputs
    s_time = time.time()
    df = pd.read_parquet(file_path)
    _dataset = datasets.Dataset.from_pandas(df)
    load_time = time.time() - s_time

    # convert to jsonl
    s_time = time.time()
    _dataset.to_json(out_path, batch_size=4096, num_proc=num_proc)
    convert_time = time.time() - s_time

    print(f"done [ {file_path.split('/')[-1]} ] load_time: {load_time} convert_time: {convert_time}")

def convert_1set(start_idx, end_idx, input_dir, output_dir):
    s_time = time.time()
    
    # create task list
    multi_proc_inputs = []
    for idx in range(start_idx, end_idx+1):
        filename_prefix = f"train-{str(idx).zfill(5)}-of-05534-"
        parquet_filename = __search_idx_file(input_dir, filename_prefix)
        parquet_path = os.path.join(input_dir, parquet_filename)
        out_path = os.path.join(output_dir, parquet_filename.replace(".parquet", ".jsonl"))
        multi_proc_inputs.append((parquet_path, out_path, None))    
        
    # multi process
    with  ProcessPoolExecutor(max_workers=len(multi_proc_inputs)) as executor:
        results = executor.map(convert_parquet_to_json_multi, multi_proc_inputs)
    e_time = time.time() - s_time
    print(results)
    print(f"done [ {start_idx} - {end_idx} ] time: {e_time}")
    

def convert_refinedWeb_multi():
    """ Convert parquet to jsonl (Multi file)
        srun pre環境だとこちらのほうが圧倒的に早い. 36fileを51secで処理 (--cpus-per-task=36, multi_proc_num=36)
    """
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./tmp/output")
    args = parser.parse_args()
    
    start_idx = 2034
    end_idx = 5533
    multi_proc_num = 36
    for set_start_idx in range(start_idx, end_idx+1, multi_proc_num):
        set_end_idx = set_start_idx + multi_proc_num - 1
        if set_end_idx > end_idx:
            set_end_idx = end_idx
        print(f'start set {set_start_idx} - {set_end_idx}')
        
        convert_1set(set_start_idx, set_end_idx, args.input_dir, args.output_dir)



# ---------------------- multi process replace key name ------------------------------------------
def add_text_as_main_target(example, use_main_key):
    """ [map function] Datasetのitemとして，メイン入力扱いするkey=`text`とするため， 元Datasetのfeaturesからkey=`text`とするfeatureを選択し，その値を`text`にもcopy
        元のDatasetに`text`が存在していた場合には，指定されたkeyの値で上書きされることに注意
    """
    example['text'] = example[use_main_key]
    return example

def replace_key_for_refinedWeb(inputs):
    """ refinedWeb 記事の中身は key:content になっているが， key:text に変更する"""
    file_path, out_path, num_proc = inputs
    s_time = time.time()
    _dataset = datasets.load_dataset("json", data_files=file_path)['train']
    load_time = time.time() - s_time

    # replace
    s_time = time.time()
    _dataset = _dataset.map(add_text_as_main_target, fn_kwargs={"use_main_key": "content"})
    _dataset = _dataset.remove_columns(["content"])
    # save 
    _dataset.to_json(out_path, batch_size=4096, num_proc=num_proc)
    convert_time = time.time() - s_time

    print(f"done [ {file_path.split('/')[-1]} ] load_time: {load_time} convert_time: {convert_time}")

def replace_1set_key(start_idx, end_idx, input_dir, output_dir):
    s_time = time.time()
    # create task list
    multi_proc_inputs = []
    for idx in range(start_idx, end_idx+1):
        filename_prefix = f"train-{str(idx).zfill(5)}-of-05534-"
        base_json_filename = __search_idx_file(input_dir, filename_prefix)
        base_json_path = os.path.join(input_dir, base_json_filename)
        out_path = os.path.join(output_dir, base_json_filename)
        multi_proc_inputs.append((base_json_path, out_path, None))    
        
    # multi process
    with  ProcessPoolExecutor(max_workers=len(multi_proc_inputs)) as executor:
        results = executor.map(replace_key_for_refinedWeb, multi_proc_inputs)
    e_time = time.time() - s_time
    print(results)
    print(f"done [ {start_idx} - {end_idx} ] time: {e_time}")

def replace_key_refinedWeb_multi():
    """ jsonl の key:content を key:text として置換する
    """
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./tmp/output")
    args = parser.parse_args()
    
    start_idx = 0
    end_idx = 0 #5533
    multi_proc_num = 36
    for set_start_idx in range(start_idx, end_idx+1, multi_proc_num):
        set_end_idx = set_start_idx + multi_proc_num - 1
        if set_end_idx > end_idx:
            set_end_idx = end_idx
        print(f'start set [ {set_start_idx} - {set_end_idx} ]')
        replace_1set_key(set_start_idx, set_end_idx, args.input_dir, args.output_dir)



if __name__ == "__main__":
    # convert_refinedWeb()
    # convert_refinedWeb_multi()    # step1
    replace_key_refinedWeb_multi()  # step2