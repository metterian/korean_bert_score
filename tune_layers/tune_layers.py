import argparse
import os
import re
from collections import defaultdict
import json



import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm.auto import tqdm, trange

import bert_score


def get_wmt16(lang_pair, data_folder="wmt16"):
    with open(
        os.path.join(
            data_folder,
            f"wmt16-metrics-results/seg-level-results/DAseg-newstest2016/DAseg-newstest2016.human.{lang_pair}",
        )
    ) as f:
        gold_scores = list(map(float, f.read().strip().split("\n")))

    with open(
        os.path.join(
            data_folder,
            f"wmt16-metrics-results/seg-level-results/DAseg-newstest2016/DAseg-newstest2016.reference.{lang_pair}",
        )
    ) as f:
        all_refs = f.read().strip().split("\n")

    with open(
        os.path.join(
            data_folder,
            f"wmt16-metrics-results/seg-level-results/DAseg-newstest2016/DAseg-newstest2016.mt-system.{lang_pair}",
        )
    ) as f:
        all_hyps = f.read().strip().split("\n")

    return gold_scores, all_refs, all_hyps


def get_wmt16_seg_to_bert_score(lang_pair, scorer, data_folder="wmt16", batch_size=64):
    # os.makedirs(f"cache_score/{network}", exist_ok=True)
    # path = "cache_score/{}/wmt16_seg_to_{}_{}.pkl".format(network, *lang_pair.split("-"))

    gold_scores, refs, cands = get_wmt16(lang_pair, data_folder=data_folder)
    if scorer.idf:
        scorer.compute_idf(refs)
    scores = scorer.score(cands, refs, verbose=False, batch_size=batch_size)
    scores = list(scores)
    max_length = scorer._tokenizer.max_len_single_sentence

    return scores, gold_scores, max_length


def get_aihub_data(lang_pair, data_folder="../data"):
    """AIHub 번역품질평가 데이터를 읽어오는 함수"""
    gold_scores = []
    all_refs = []
    all_hyps = []
    
    # Training 데이터와 Validation 데이터 모두 처리
    # for data_type in ["Training", "Validation"]:
    for data_type in [ "Validation"]:
        data_path = os.path.join(
            data_folder,
            "008.다국어 번역 품질 평가 데이터/3.개방데이터/1.데이터",
            data_type,
            "02.라벨링데이터/extracted"
        )
        
        # JSON 파일들을 순회
        for json_file in os.listdir(data_path):
            if json_file.startswith(f"{lang_pair.lower()}") and json_file.endswith('.json'):
                with open(os.path.join(data_path, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data['data']:
                        gold_scores.append(item['score_da_final'])
                        all_refs.append(item['ht'])
                        all_hyps.append(item['mt'])
    
    return gold_scores, all_refs, all_hyps


def get_aihub_bert_score(lang_pair, scorer, data_folder="../data", batch_size=64):
    """AIHub 데이터에 대한 BERT Score를 계산하는 함수"""
    gold_scores, refs, cands = get_aihub_data(lang_pair, data_folder=data_folder)
    if scorer.idf:
        scorer.compute_idf(refs)
    scores = scorer.score(cands, refs, verbose=True, batch_size=batch_size)
    scores = list(scores)
    max_length = scorer._tokenizer.max_len_single_sentence

    return scores, gold_scores, max_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="wmt16", help="path to wmt16 data")
    parser.add_argument("-m", "--model", nargs="+", help="models to tune")
    parser.add_argument(
        "-l", "--log_file", default="best_layers_log.txt", help="log file path"
    )
    parser.add_argument("--idf", action="store_true")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument(
        "--lang_pairs",
        nargs="+",
        default=["cs-en", "de-en", "fi-en", "ro-en", "ru-en", "tr-en"],
        help="language pairs used for tuning",
    )
    args = parser.parse_args()

    if args.log_file.endswith(".txt"):
        csv_file = args.log_file.replace(".txt", ".csv")
    else:
        csv_file = args.log_file + ".csv"

    torch.set_grad_enabled(False)

    networks = args.model
    for network in networks:
        model_type = network
        scorer = bert_score.scorer.BERTScorer(
            model_type=model_type, num_layers=100, idf=False, all_layers=True
        )
        results = defaultdict(dict)
        for lang_pair in tqdm(args.lang_pairs):
            print(f"lang_pair: {lang_pair}")
            # AIHub 데이터인 경우
            if lang_pair in ["enko", "jako", "zhko"]:
                scores, gold_scores, max_length = get_aihub_bert_score(
                    lang_pair, scorer, batch_size=args.batch_size
                )
            # WMT16 데이터인 경우
            else:
                scores, gold_scores, max_length = get_wmt16_seg_to_bert_score(
                    lang_pair, scorer, batch_size=args.batch_size
                )
            
            for i, score in enumerate(scores[2]):
                results[lang_pair + " " + str(i)]["%s %s" % (network, "F")] = pearsonr(
                    score, gold_scores
                )[0]

        best_layer, best_corr = 0, 0.0
        for num_layer in range(100):
            temp = []
            if f"{args.lang_pairs[0]} {num_layer}" not in results:
                break
            for lp in args.lang_pairs:
                temp.append(results[f"{lp} {num_layer}"][f"{network} F"])
            corr = np.mean(temp)
            results["avg" + " " + str(num_layer)]["%s %s" % (network, "F")] = corr
            print(network, num_layer, corr)
            if corr > best_corr:
                best_layer, best_corr = num_layer, corr

        if args.idf:
            msg = f"'{network}' (idf): {best_layer}, # {best_corr}"
        else:
            msg = f"'{network}': {best_layer}, # {best_corr}"
        print(msg)
        with open(args.log_file, "a") as f:
            print(msg, file=f)
        csv_msg = f"{network},{best_layer},{best_corr},,{max_length}"
        with open(csv_file, "a") as f:
            print(csv_msg, file=f)

        del scorer


if __name__ == "__main__":
    main()
