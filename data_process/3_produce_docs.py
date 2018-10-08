#coding:utf-8
import sys
import json
import random
import matplotlib.pyplot as plt
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')

docs = json.load(open('news_sohusite_labeled.json'))

min_size = 10
max_size = 100

def length_stac(docs):
    freq = [0]*2100
    X = []
    Y = []
    for label in docs:
        for doc in docs[label]:
            freq[len(doc)] += 1
    for x,y in enumerate(freq):
        if y > 0:
            X.append(x)
            Y.append(y)
    plt.plot(np.asarray(X[:100]),np.asarray(Y[:100]))
    plt.show()
    return

def produce_docs(docs):
    train_docs = []
    test_docs = []
    label_num = 0
    for i in range(1,16):
        train_num = 0
        test_num = 0
        nums = 0
        if str(i) in docs:
            d = docs[str(i)]
            if len(d) > 1900:
                label_num += 1
                for doc in d:
                    content = doc
                    if len(doc) > max_size:
                        content = doc[:max_size]
                    if nums < 1600:
                        train_docs.append({'label':str(label_num), 'content': content})
                        train_num += 1
                    else:
                        test_docs.append({'label':str(label_num), 'content': content})
                        test_num += 1
                    nums += 1
                    if nums == 2000:
                        break
                print 'label '+str(label_num)+': train '+str(train_num)+', test '+str(test_num)
    print 'train docs:'+str(len(train_docs))
    print 'test docs:'+str(len(test_docs))
    
    random.shuffle(train_docs)
    random.shuffle(test_docs)
    train_contents = []
    train_labels = []
    test_contents = []
    test_labels = []
    for doc in train_docs:
        train_contents.append(doc['content'])
        train_labels.append(doc['label'])
    for doc in test_docs:
        test_contents.append(doc['content'])
        test_labels.append(doc['label'])
    f1 = open('train_contents_unsplit.txt','w')
    f2 = open('train_labels.txt','w')
    f3 = open('test_contents_unsplit.txt','w')
    f4 = open('test_labels.txt','w')
    f1.write('\n'.join(train_contents))
    f2.write('\n'.join(train_labels))
    f3.write('\n'.join(test_contents))
    f4.write('\n'.join(test_labels))
    f1.close()
    f2.close()
    f3.close()
    f4.close()
   
produce_docs(docs)