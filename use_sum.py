import json
import re
import requests

import argparse
import en_core_web_sm
from flask import jsonify, request
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
from scipy import spatial

TOPK = 3

"""
This is a simple script for invoking summarization server.
It needs a running OpenNMT server with extractive model 0 and abstractive sumamrizator as 1 
"""

class USPServer():

    # parameters are id's of the models in available_models/conf.json
    def __init__(self, extractive_id=0, abstractive_id=1, host="0.0.0.0", port=5000, server_path="translator/translate"):
        self.extractive_id = extractive_id
        self.abstractive_id = abstractive_id
        self.nlp = en_core_web_sm.load()
        self.SEPARATOR = "￨" #￨|
        self.host = host
        self.port = port
        self.server_path = server_path

        self.splitregex = re.compile("(?<!^)\.(?=[A-Z])")
        self.review_clean_regex = re.compile('[\'\-\(\)&]')

        # TO DO move these constants to config
        self.HOST_USE = "0.0.0.0"
        self.PORT_USE = 8501
        self.PATH_USE = "v1/models/universal_encoder:predict"

    #add features for abstarctive sumamrization input
    def tag(self, reviews):
        ret = ""
        for r in reviews:
            self.review_clean_regex.sub(' ', r)
            doc = self.nlp(r)
            word_id = -1
            for word in doc:
                ner = "_"
                if word.ent_type_!="":
                    ner = word.ent_type_
                word_id += 1
                lemma = word.lemma_
                if lemma == "-PRON-":
                    lemma = word.text_with_ws.strip().lower()
                if lemma!="":
                    ret += lemma + self.SEPARATOR + word.pos_ + self.SEPARATOR + word.dep_ + self.SEPARATOR + ner + self.SEPARATOR + word.ent_iob_+" "
            ret += "\n"
        return ret

    def _summarize_single(self, text):
        extractive_tagged, extractive_original, scores, pred_vec = self._extractive_summary(text)
        abstractive = []
        for tagged in extractive_tagged:
            abstractive.append(self._query_server(tagged, self.abstractive_id )[0] )
        return abstractive, extractive_original, scores, pred_vec

    def _query_server(self, text, model_id):
        url = 'http://' + self.host + ":" + str(self.port) + "/" + self.server_path
        data = [{"src": text, "id": model_id}]
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        r = requests.post(url, data=json.dumps(data), headers=headers)
        obj = json.loads(r.text) #[0][0]["tgt"]
        ret = []
        ret_src = []
        ret_scores = []
        if len(obj)>0 and isinstance(obj, list):
            for o in obj[0]:
                ret.append(o["tgt"])
                ret_src.append((o["src"]))
                ret_scores.append((o["pred_score"]))
        else:
            print("PROBLEM "+str(obj))
        return ret, ret_src, ret_scores

    def get_USE_vector(self, texts):
        url = 'http://' + self.HOST_USE + ":" + str(self.PORT_USE) + "/" + self.PATH_USE
        data = {"instances": [texts]}
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        r = requests.post(url, data=json.dumps(data), headers=headers)
        ret = json.loads(r.text)
        return ret

    def _extractive_summary(self, data):

        distances_pred = []
        predicted_vector, srcs, scores = self._query_server(data, self.extractive_id)
        predicted_vector = np.array(predicted_vector)

        srcs = np.array(srcs[0][0])
        all_sets = data
        reversed = False
        ret_tagged = []
        ret_untagged = []
        ret_best_scores = []

        for target_id in range(predicted_vector.shape[1]):
            distances_pred.append([])
            sent_id = 0.0
            for vec in srcs:
                v = np.array(vec)
                dist = spatial.distance.cosine(v , predicted_vector[0,target_id])
                distances_pred[target_id].append(dist ) # + sent_id/100
                sent_id += 1.0

            sortd = sorted(zip(distances_pred[target_id], all_sets, range(len(all_sets)) ), reverse=reversed)

            index = 0
            best_sents = []
            best_scores= []

            while len(best_sents) < TOPK:
                if index >= len(sortd):
                    break
                curr = sortd[index]
                best_sents.append(curr[1])
                best_scores.append(curr[0])
                index += 1

            ret_best_scores.append(best_scores)
            local_review_string = ""
            untagged = best_sents[0:TOPK]

            for best in untagged[0:TOPK]:
                local_review_string += self.tag([best])
            ret_tagged.append(local_review_string)
            ret_untagged += untagged

        return ret_tagged, ret_untagged, ret_best_scores, predicted_vector

    def select_best_summary(self, in_summaries, extractive_texts, pred_vec=None):

        target_vec = np.array(pred_vec)
        dists = []
        best_dist = 10.3
        best_str = ""

        for s in in_summaries:
            vec = np.array(self.get_USE_vector(s)["predictions"])
            dist = spatial.distance.cosine(target_vec, vec)
            # print(s+"  "+str(dist))
            dists.append((s, dist))
            if dist < best_dist:
                best_str = s
                best_dist = dist
            # dists.append((dist))
        # if best_str=="":
        #     print(dists)
        return best_str

    def clean_inputs(self, inputs):
        ret = []
        for inp in inputs:
            r = self.splitregex.split(inp)
            for r2 in r:
                ret.append(r2)
        return ret

    def summarize(self, inputs):
        out_summaries = []
        out_extracted = []
        for input in inputs:
            input = sent_tokenize(input)
            summary, extracted,scores, pred_vec = self._summarize_single(input)
            summary = self.select_best_summary(summary[0][0], extracted, pred_vec)
            out_summaries.append(summary)
            out_extracted.append(extracted)
        return out_summaries, out_extracted

    def extractive(self, inputs):
        out_extracted = []
        out_scores = []
        for input in inputs:
            input = sent_tokenize(input)
            extracted_tagged, extracted, scores, pred_vec = self._extractive_summary(input)
            out_extracted.append(extracted)
            out_scores.append(scores)
        return out_extracted

def read_sources(in_path):
    content=[]
    with open(in_path) as f:
        content = f.readlines()
    return content

def save_text(summaries, extracted, file_name):
    if summaries is not None:
        df = pd.DataFrame({"Summaries": summaries, "Extracted": extracted})
    else:
        df = pd.DataFrame({"Extracted": extracted})
    df.to_csv(file_name, sep=';', quotechar='"', encoding='utf-8')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="USE summarization inference")
    parser.add_argument("--server_ip", "-ip", type=str, default="0.0.0.0", help="Summarization server ip address")
    parser.add_argument("--port", type=int, default=5000, help="Summarization server port")
    parser.add_argument("--path", type=str, default="translator/translate", help="Summarization server path")

    parser.add_argument("--input_file", "-i", type=str, default="data/sample/src-test-txt.txt", help="File containing input for summarization")
    parser.add_argument("--output_file", "-o", type=str, default="data/sample/out.csv", help="File where the output is stored")
    parser.add_argument("--type", "-t", type=str, default="e2e", choices=['extractive','e2e'], help='type of summarization. Available options are extractive and e2e . e2e stands for end-to-end summarization.')
    args = parser.parse_args()

    summarization_input = read_sources(args.input_file)
    usp = USPServer(extractive_id=0, abstractive_id=1, host=args.server_ip, port=args.port, server_path=args.path)
    if args.type=="extractive":
        extracted = usp.extractive(summarization_input)
        summaries = None
    else:
        summaries, extracted = usp.summarize(summarization_input)
    save_text(summaries=summaries, extracted=extracted,  file_name=args.output_file)
