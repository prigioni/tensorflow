#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019-9-8 16:34 
# @Author : lauqasim
# @File : inferenceDataOnline.py 
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import pickle
import traceback

display_step = 300

epochs = 13
batch_size = 128

rnn_size = 128
num_layers = 3

encoding_embedding_size = 200
decoding_embedding_size = 200

learning_rate = 0.001
keep_probability = 0.5

def load_preprocess():
    with open('data/preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)
import numpy as np

(source_text_to_int, target_text_to_int), (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = load_preprocess()


def sentence_to_seq(sentence, vocab_to_int):
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])

    return results


# --- 线上服务 ---
from flask import Flask, request, render_template, jsonify
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def online_predict():
    print('test')
    if request.method == "POST":
        # try:
        x = request.form['eng']
        translate_sentence = sentence_to_seq(x, source_vocab_to_int)
        load_path = r'data\dev'
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(load_path + '.meta')
            loader.restore(sess, load_path)

            input_data = loaded_graph.get_tensor_by_name('input:0')
            logits = loaded_graph.get_tensor_by_name('predictions:0')
            target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
            keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

            translate_logits = sess.run(logits, {input_data: [translate_sentence] * batch_size,
                                                 target_sequence_length: [len(translate_sentence) * 2] * batch_size,
                                                 keep_prob: 1.0})[0]
        translations = " ".join([target_int_to_vocab[i] for i in translate_logits[:-1]])
        return render_template('index.html', ENG=x, RESULT=translations)
        # except Exception:
        #     print(traceback.print_exc())
    return render_template('index.html')

# 线上测试
app.run('localhost', port=8899)
# serve(app, host='0.0.0.0', port=8080)
# http_server =WSGIServer(('0.0.0.0', 8899), app)
# http_server.serve_forever()