from datetime import datetime
import json
from hojichar import document_filters, tokenization, Compose, Document
import os

import custom_token_filters, custom_tokenization, custom_document_filters


def mc4_ja_main():
    input_dir = "/persistentshare/storage/team_sannai/corpus/category/1B/raw/ja_mc4/"
    output_dir = "/persistentshare/storage/team_sannai/corpus/category/1B/filtered/ja_mc4/"
 
    start_idx = 0
    end_idx = 8


    jsonl_list = []
    for idx in range(start_idx, end_idx+1):
        jsonl_list.append(f"c4-ja-{str(idx).zfill(3)}.jsonl")
    print(f"dealing jsonl list: {jsonl_list}")
    
if __name__ == "__main__":
    mc4_ja_main()