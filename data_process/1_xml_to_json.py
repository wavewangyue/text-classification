#coding:utf-8
import xml.dom.minidom
import sys
import json
reload(sys)
sys.setdefaultencoding('utf8')

xml = open("news_sohusite_xml.txt").read()
docs_xml = xml.split('<doc>\n')
print len(docs_xml)
del docs_xml[0]
docs = []

for doc_xml in docs_xml:
    if (len(doc_xml.split('<url>')) != 2) or (len(doc_xml.split('<docno>')) != 2) or (len(doc_xml.split('<contenttitle>')) != 2) or (len(doc_xml.split('<content>')) != 2):
        continue
    url = doc_xml.split('<url>')[1].split('</url>')[0]
    docno = doc_xml.split('<docno>')[1].split('</docno>')[0]
    title = doc_xml.split('<contenttitle>')[1].split('</contenttitle>')[0]
    content = doc_xml.split('<content>')[1].split('</content>')[0]
    doc = {'url':url, 'docno':docno, 'title':title, 'content':content}
    docs.append(doc)

print len(docs)
fout = open("news_sohusite.json",'w')
fout.write(json.dumps(docs,ensure_ascii=False))
fout.close()
