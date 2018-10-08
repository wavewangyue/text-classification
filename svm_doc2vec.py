#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

VECTOR_DIR = 'vectors.bin'

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
TEST_SPLIT = 0.2

train_docs = open('train_contents.txt').read().split('\n')
train_labels = open('train_labels.txt').read().split('\n')
test_docs = open('test_contents.txt').read().split('\n')
test_labels = open('test_labels.txt').read().split('\n')

def train_d2v_model():
    all_docs = train_docs + test_docs
    fout = open('all_contents.txt','w')
    fout.write('\n'.join(all_docs))
    fout.close()
    import gensim
    sentences = gensim.models.doc2vec.TaggedLineDocument('all_contents.txt')
    model = gensim.models.Doc2Vec(sentences, size=200, window=5, min_count=5)
    model.save('doc2vec.model')
    print 'num of docs: ' + str(len(model.docvecs))
        

if __name__ == '__main__':
    print '(1) training doc2vec model...'
    # train_d2v_model()
    print '(2) load doc2vec model...'
    import gensim
    model = gensim.models.Doc2Vec.load('doc2vec.model')
    x_train = []
    x_test = []
    y_train = train_labels
    y_test = test_labels
    for idx, docvec in enumerate(model.docvecs):
        if idx < 17600:
            x_train.append(docvec)
        else:
            x_test.append(docvec)
    print 'train doc shape: '+str(len(x_train))+' , '+str(len(x_train[0]))
    print 'test doc shape: '+str(len(x_test))+' , '+str(len(x_test[0]))

    print '(3) SVM...'
    from sklearn.svm import SVC   
    svclf = SVC(kernel = 'rbf') 
    svclf.fit(x_train,y_train)  
    preds = svclf.predict(x_test);  
    num = 0
    preds = preds.tolist()
    for i,pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print 'precision_score:' + str(float(num) / len(preds))






        




