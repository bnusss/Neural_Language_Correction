#!/bin/python
# -*- coding: utf-8 -*-

import json
from util import *
from decode import *
from flask import Flask
from flask import request


app = Flask(__name__)

@app.route("/revise", methods=['GET', 'POST'])
def revise():
    if request.method == 'POST':
        original_res = request.get_json()

        original_sent = original_res['voice_text']
        print({'original': original_sent})
        output_sent = decode(original_sent)
        evals = evaluate(original_sent, output_sent)
        result = {'revised': output_sent,
                  'evaluate': evals}

        print(result)
        return json.dumps(result)
    else:
        return '<h1>这是revise连通性测试</h1>'


@app.route("/submit", methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        submit_res = request.get_json()

        original_sent = submit_res['original_sent']
        user_revise_sent = submit_res['user_revise_sent']
        result = {'original_sent':original_sent,
                  'user_revise_sent':user_revise_sent}
        
        print(result)
        return json.dumps(result)
    else:
        return '<h1>这是submit连通性测试</h1>'


load_vocab()
load_model()
app.run()
