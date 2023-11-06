# -*- coding:utf-8 -*-
import json
def get_word_index(word_count):
    words=[ word for [word,count] in word_count.items() if count>10 and len(word)>0]
    word_index=dict(zip(words,range(2,len(words)+2)))
    word_index["__token__"]=0
    word_index["__stopword__"]=1
    return word_index
def get_index(lines):
    label_index={}
    word_count={}
    count=0
    for line in lines:
        if count%1000==0:
            print(count,len(lines))
        count+=1
        label=line[0]
        words=line[1:]
        if label not in label_index:
            label_index[label]=len(label_index)
        for word in words:
            word_count[word]=word_count.get(word,0)+1
    word_index=get_word_index(word_count)
    return word_index,label_index
def parser_data(lines,word_index,label_index):
    results=[]
    count=0
    for line in lines:
        if count%1000==0:
            print(count,len(lines))
        count+=1
        label=line[0]
        words=line[1:]
        y=label_index[label]
        x=[ word_index.get(word,1) for word in words]
        results.append(str([x,y]))
    return results


with open("train_data") as f:
    lines=f.readlines()
lines=[line.strip().split(" ") for line in lines]
word_index,label_index=get_index(lines)
with open("word_data/word_index","w") as f:
    json.dump(word_index,f,ensure_ascii=False)
with open("word_data/label_index_word","w") as f:
    json.dump(label_index,f,ensure_ascii=False)
results=parser_data(lines,word_index,label_index)
with open("word_data/train_data_index_word","w") as f:
    f.writelines("\n".join(results))
