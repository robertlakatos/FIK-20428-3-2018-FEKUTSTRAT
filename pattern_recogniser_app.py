#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy
import click
import pandas
import pickle
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def write_date(output_file_name, data):
    with pandas.ExcelWriter(output_file_name, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False)
    print("WRITE DATA: COMPLETE")


def main(): 
    model = models.load_model("models/model_pattern_recogniser.h5",
                              compile=False)
    model.summary()

    with open("sources/tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("LOAD TOKENS: COMPLETE")

    files = os.listdir("sources")
    for file in files:
        # print(file)
        file_name = "sources/" + file 

        if os.path.isfile(file_name) == False:
            continue

        if(file_name.split(".")[-1] != "xlsx"):
            continue

        data = pandas.read_excel(file_name)
        print("READ EXCEL: COMPLETED")
        
        templates = json.load(open("config/config_pattern_templates.json",
                           "r",
                           encoding="utf-8"))
        lent = len(templates)
        lenv = len(data["obsval"])                
        maxlen = max([len(item["sample"].split(" ")) for item in templates])

        samples = []
        envs = []
        results = []
        for k in range(0, lent):
            results.append([])
            for n in range(0, lenv):
                results[k].append("")

        with click.progressbar(length=lenv, label="DATA COLLECTING: ", fill_char=click.style('=', fg='white')) as bar:
            for i in range(0, lenv):
                try:
                    tmp_split = data["obsval"][i].lower().split(" ")

                    len_tmp = len(tmp_split)
                    for index in range(0, len_tmp):

                        for template in templates:
                            tmp_next_tokent_count = index + 1 + template["next_tokens_count"]
                            tmp_prev_tokens_count = index - template["prev_tokens_count"]

                            if tmp_prev_tokens_count >= 0 and tmp_next_tokent_count < len_tmp and len(tmp_split[index]) > 0:
                                env = tmp_split[tmp_prev_tokens_count:index] + \
                                    tmp_split[(index+1):tmp_next_tokent_count]
                                    
                                envs.append(" ".join(env))
                                tmp_sample = {
                                    "word" : tmp_split[index],
                                    "word_index" : index,
                                    "row" : i                                    
                                }
                                samples.append(tmp_sample) 
                         
                    bar.update(1)
                except:
                    bar.update(1)

        len_envs = len(envs)
        if(len_envs > 0):
            text = tokenizer.texts_to_sequences(envs)
            text = pad_sequences(text,
                                 padding='post',
                                 maxlen=maxlen) 

            print("TOKENS CREATED")
            preds = model.predict(text)
            print("PREDICTIONS CREATED")  

        start = 0
        prev = samples[0]["row"]
        threshold = 0.9        
        

        with click.progressbar(length=len_envs, label="PREDICTIONS COLLECTING: ", fill_char=click.style('=', fg='white')) as bar:
            for i in range(0, len_envs):
                if prev != samples[i]["row"]:                    
                    tmp_preds = preds[start:i]

                    max_index = numpy.argmax(tmp_preds, axis=0)
                    max_index = max_index + start
                    start = i                    

                    finds = { }
                    len_max_index = len(max_index)
                    for j in range(0, len_max_index):
                        if preds[max_index[j]][j] < threshold:
                            continue
                        
                        tmp_word_index = samples[max_index[j]]["word_index"] 
                        if tmp_word_index not in finds or finds[tmp_word_index]["pred"] < preds[max_index[j]][j]:
                            finds[tmp_word_index] = {
                                "row" : prev,
                                "word" : samples[max_index[j]]["word"],
                                "envs" : envs[max_index[j]],
                                "pred" : preds[max_index[j]][j],
                                "index" : j
                            } 

                    for key in finds:     
                        a = finds[key]["index"]
                        b = finds[key]["row"]
                        c = finds[key]["word"]                                            
                        results[a][b] = c

                    prev = samples[i]["row"] 

                bar.update(1)

        lenr = len(results)
        for i in range(0, lenr):
            data["SAMPLE_" + str(i)] = results[i]

        write_date("sources/pred_" + file, data)


if __name__ == "__main__":
    main()
