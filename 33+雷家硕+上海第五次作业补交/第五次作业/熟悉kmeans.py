#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from numpy.linalg import norm

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):#指定训练词向量的模型
    model = Word2Vec.load(path)
    return model

def load_sentence(path):#把文章的每个句子切好
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))#不是lcut的话返回的就不是list，
                                                        # 不按''.join的话就add的是迭代器
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []#收集的是每句话的向量
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]#将每个词转化为词向量
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
        '''
        每个元素是几维呢？—— 一条
        '''
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
                                #虽然是用向量聚类计算的，但还是会 句子，向量，类别 一一对应
                                # 聚类没有标签，就是按照kmeans算法自己的规定直到最终迭代停止，所以没有计算loss的过程

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起，同标签的是一个矩阵，一个键对应的是一个列表，
                                                            #列表里面的元素都是同类的句子
    for label, sentences in sentence_label_dict.items():#这个for这么命名就代表每一类对应的那个句子列表叫sentences了
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

def cosdis(vec1, vec2):
    x1 = vec1/norm(vec1, axis = -1, keepdims=True)
    x2 = vec2/norm(vec2, axis = -1, keepdims=True)
    return np.dot(x1, x2.T)

# def main():#重现
#     sentences = load_sentence('titles.txt')
#     model = load_word2vec_model('model.w2v')
#     vectors = sentences_to_vectors(sentences, model)
#
#     n_clusters = int(math.sqrt(len(sentences)))
#     kmeans = KMeans(n_clusters)
#     kmeans.fit(vectors)
#     #现在要拿出来的是每个分类及其包含的所有句子，句向量只是中间的一个计算过程
#     sentence_dict = defaultdict(list)
#     for sentence, label in zip(sentences, kmeans.labels_):
#         sentence_dict[label].append(sentence)               #拿到了每一类及其包含的句子，剩下的就是打印了
#                                                             #算类内距离，余弦距离,取类内所有两两向量距离的平均
#
#     #计算类内距离
#     for label, sentencelist in sentence_dict.items():
#         #首先将这一类文本转化为句向量
#         vectorlist = []
#         for sentence in sentencelist:
#             vector = sentences_to_vectors(sentence, model)
#             vectorlist.append(vector)
#         #计算这个类内的所有句向量的余弦距离
#         culnum = 0#总的两两向量计算次数
#         iandelse = []
#         for i in range(len(vectorlist)-1):
#             whilenum = 1#当前向量与其他向量的计算次数，序号用到
#             while int(len(vectorlist)-i-1):
#                 culnum += 1
#                 iandanotherdis = cosdis(vectorlist[i], vectorlist[i+whilenum])
#                 iandelse.append(iandanotherdis)
#                 whilenum += 1
#         alldis = np.array(iandelse)
#         labeldis = alldis/culnum    #类内距离
#



if __name__ == "__main__":
    main()