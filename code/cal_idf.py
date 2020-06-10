import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json
import math
with open("forward_index") as f:
	lines=f.readlines()
lines=[json.loads(line.strip())['article_title_words'] for line in lines]
lines=[set(words) for words in lines]
num=len(lines)
word_count={}
for words in lines:
	for word in words:
		word_count[word]=word_count.get(word,0.0)+1.0
idf=dict([ [word,math.log(num/count)] for [word,count] in word_count.items()])
with open("idf","w") as f:
	json.dump(idf,f,ensure_ascii=False) 
