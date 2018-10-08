#coding:utf-8
import sys
import json
import random
reload(sys)
sys.setdefaultencoding('utf8')

docs = json.load(open('news_sohusite.json'))
label_maps = json.load(open('url_to_catagory_to_label.json'))
num_of_docs_of_label = [0]*15
docs_labeled = {}

num = 0
num_labeled = 0
for doc in docs:
    num += 1
    if num%10000 == 0:
        print num
    content = doc['title']+' '+doc['content']
    url = doc['url']
    label = None
    for label_map in label_maps:
        if label_map['url'] in url:
            label = label_map['label']
            break
    if label is None:
        continue
    num_labeled += 1
    if str(label) not in docs_labeled:
        docs_labeled[str(label)] = []
        docs_labeled[str(label)].append(content)
    else:
        docs_labeled[str(label)].append(content)

print 'all:'+str(len(docs))
print 'labeled:'+str(num_labeled)

for i in range(1,16):
    if str(i) in docs_labeled:
        print 'label '+str(i)+':'+str(len(docs_labeled[str(i)]))

fout = open('news_sohusite_labeled.json','w')
fout.write(json.dumps(docs_labeled,ensure_ascii=False,indent=4))
fout.close()