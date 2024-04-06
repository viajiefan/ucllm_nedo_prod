from datetime import datetime
import json
from hojichar import document_filters, tokenization, Compose, Document
import argparse
import os
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


import custom_token_filters, custom_tokenization, custom_document_filters

def __readlines(input_file: str):
    with open(input_file) as fp:
        return fp.readlines()
    
def process_json_lines(inputs):
    print(f"process json filter inputs: {inputs}")
    json_filename, input_dir, output_dir = inputs
    lines = __readlines(input_dir + json_filename)

    remained_lines = []
    cleaner = Compose([
        document_filters.JSONLoader(),
        document_filters.DocumentNormalizer(),
        document_filters.DiscardBBSComments(),
        document_filters.DiscardAds(),
        document_filters.DiscardDiscriminationContentJa(),
        custom_document_filters.DiscardAdultContentJa(),
        custom_tokenization.NewLineSentenceTokenizer(),
        custom_token_filters.RemoveOneword(),
        custom_tokenization.MergeTokens(delimiter="\n"),
        custom_tokenization.WakatiTokenizer(),
        custom_token_filters.RemoveDate(),
        tokenization.MergeTokens(),
        document_filters.MaskPersonalInformation(),
        document_filters.JSONDumper(dump_reason=True),
    ])

    
    with open(os.path.join(output_dir, f"rejected_{json_filename}"), "w") as rejected:
        with open(os.path.join(output_dir, f"filtered_{json_filename}"), "w") as writer:
            for line in lines:
                result = cleaner.apply(Document(line))
                if result.is_rejected:
                    rejected.write(result.text + "\n")
                else:
                    writer.write(result.text + "\n")
                    # remained_lines.append(result.text)

    with open(os.path.join(output_dir, f"stat_{json_filename}"), "w") as writer:
        writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")


def mc4_ja_main():
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./tmp/output")
    args = parser.parse_args()
    
    start_idx = 0
    end_idx = 1


    jsonl_list = []
    for idx in range(start_idx, end_idx+1):
        jsonl_list.append(f"c4-ja-{str(idx).zfill(3)}.jsonl")
    print(f"dealing jsonl list: {jsonl_list}")
    
    process_num = len(jsonl_list)

    # ThreadPoolExecutorを使って並列化
    with ThreadPoolExecutor(max_workers=process_num) as executor:
        results = executor.map(process_json_lines, [(jsonname, args.input_dir, args.output_dir) for jsonname in jsonl_list])
        
    for result in results:
        print(result)

    # with Pool(process_num) as p: 
        # exit_codes = p.map(process_json_lines, [(jsonname, input_dir, output_dir) for jsonname in jsonl_list])
        # print("Exit codes : {}".format(exit_codes))

if __name__ == "__main__":
    mc4_ja_main()