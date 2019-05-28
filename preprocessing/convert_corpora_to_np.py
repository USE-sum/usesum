import os
import re

import argparse
from nltk.tokenize import sent_tokenize
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

def vectorize_texts(texts):
    output = []
    with tf.Session() as session:

        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        input_sentences = tf.placeholder(tf.string, shape=[None])
        embeddings = embed(input_sentences)
        for t in texts:
            sentences = sent_tokenize(t)
            embeds = embeddings.eval(feed_dict={input_sentences: sentences })
            matrix = np.array(embeds)
            output.append(matrix)

    return output

def get_lines(file):
    ret = []
    ff = open(file, 'r')
    content = ff.readlines()
    ff.close()
    ret += content
    ret = [re.sub(r"\.\s*\.", "." ,re.sub(r"^.{0,30}-lrb-[\w\W\s]{1,15}-rrb-", "" ,x.rstrip().replace("</t> <t>",".").rstrip().replace("<t>","").replace("</t>","") )) for x in ret ]
    return ret

def save_paralel_np(corpora, file_name):
    f = file_name + ".npy"
    np.save(f, np.asarray(corpora))

if __name__ == "__main__":
    a = np.version.version
    aparser = argparse.ArgumentParser(description='Convert text corpora to sentence embeddings as numpy vectors.')
    aparser.add_argument('-c', '-corpora_path', required=True, type=str, help="Path to folder containing corpora  in textual form.")
    args = aparser.parse_args()
    path = args.c
    for file_name in os.listdir(path):
        if file_name.endswith(".txt"):
            lines = get_lines(path + file_name)
            vectorized = vectorize_texts(lines)
            save_paralel_np(vectorized, path+file_name.replace(".txt", "") )


