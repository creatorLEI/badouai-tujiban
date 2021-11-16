'''
计算一句话A属于某个类别的概率，可能有多个类别，属于哪个类别都有可能，概率从低到高排

通过计算这句话中包含的词B，每个词属于某一类别的概率相乘实现

也就是计算P(A|B),包含这些词的情况下这句话属于A类的概率

bayes_pro: P(A|B) = P(A) * P(B|A) / P(B)
label_pro: P(A)：整个样本中有多少属于A类的文本，直接通过统计计算
label_word_pro: P(B|A)：A类样本中包含词B的概率，从训练数据中通过词频计算？
word_pro: P(B)：这些词出现的概率，不知道，用全概率公式求
P(B) = P(B|A1) * P(A1) + P(B|A2) * P(A2) +...+ P(B|An) * P(An)

需要的数据：dict，这句话及这句话所属的类别

#bayes所谓的训练是用训练集得到每个单词P(A|Wx)和和每个类别P(A)的概率，预测的时候不用再计算了直接用
因为测试集可能比较小，所以直接用测试集算bayes概率只会得到当前数据集的准确结果，没有普适性
所以类别和单词都要训练
'''
from collections import defaultdict
import numpy as np
import json
import jieba


def load_file(path):            #按照tag将每句话聚合；再把每句话分词
    tag_word_dict = defaultdict(list)
    all_words = set()
    all_class = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            line = json.loads(line)
            tag = line['tag']
            title = line['title']
            words = jieba.lcut(title)
            all_words = all_words.union(set(words))     #得到了不包含重复词的词表
            all_class[tag] = len(all_class)     #看有几类
            tag_word_dict[tag].append(words)
    return tag_word_dict, all_words, all_class
    #tag_word_dict得到的是{各tag:[[a,b,c],[c,d,f],...,[q.e.d]]}
    #all_word得到词表
    #all_class得到的是所有的类别，一共有18类



#计算各类别的概率，注意看的是每类句子占总句子数的比，要得到一个总表
def tag_pro(tag_word_dict):
    tag_pro_dict = {}
    total_sentence_num = sum([len(sentences) for sentences in tag_word_dict.values()])
    for tag, sentence_list in tag_word_dict.items():
        tag_sentence_num = len(sentence_list)
        p_tag = tag_sentence_num/total_sentence_num
        tag_pro_dict[tag] = p_tag
    return tag_pro_dict
    #得到的是每个类别的概率{每个tag:概率}

#计算所有词出现在各类中的概率，即P(Wx|Ax)
#bayes的训练要得到每个词出现在各种类别中的概率，要计算出一个完整的表，预测的时候直接调用
def tag_word_pro(tag_word_dict, all_words):
    word_alltag_dict = {}
    for word in all_words:
        tag_word_pro_dict = {}
        for tag, sentence_list in tag_word_dict.items():
            word_num = sum([sentence.count(word) for sentence in sentence_list])
            total_num = sum([len(sentence) for sentence in sentence_list])
            pro = word_num/total_num
            tag_word_pro_dict[tag] = pro
        word_alltag_dict[word] = tag_word_pro_dict
    return word_alltag_dict
    #得到的是{每个词:{各类别：概率}}

def word_pro(tag_word_dict, all_words):
    total_dict = {}
    word_alltag_dict = tag_word_pro(tag_word_dict, all_words)
    for word, alltag_dict in word_alltag_dict.items():
        p_word = 0
        for tag, pro in alltag_dict.items():
            p_tag_dict = tag_pro(tag_word_dict)
            p_multi = p_tag_dict[tag] * pro
            p_word += p_multi
        total_dict[word] = p_word
    return total_dict
    #得到的是{每个词：全概率}

#预测使用的公式，P(Ax|B) = (P(Ax|W1)+P(Ax|W2)+...+P(Ax|Wn)) / n
#所以训练集的最后一步需要得到此表中所有词对所有类别的概率P(Ax|Wx)
def bayes(tag_word_dict, all_words, all_class):
    word_alltag_dict = tag_word_pro(tag_word_dict, all_words)
        #{每个词:{各类别：概率}} P(Wx|Ax)
    tag_dict = tag_pro(tag_word_dict)
        #{每个tag:概率} P(Ax)
    word_dict = word_pro(tag_word_dict, all_words)
        #{每个词：全概率} P(Wx)
    allword_alltag_bayes_dict = {}
    for word in all_words:
        #Wx为 word
        word_alltag_bayes_dict = {}
        for tag in all_class.keys():
            # Ax为 tag 类
            p_tag = tag_dict[tag]
            p_tag_word = word_alltag_dict[word][tag]
            p_word = word_dict[word]
            p_bayes = p_tag * p_tag_word / p_word
            word_alltag_bayes_dict[tag] = p_bayes
        allword_alltag_bayes_dict[word] = word_alltag_bayes_dict
    return allword_alltag_bayes_dict
    #得到的是 这个词属于各类的概率

def predict(text, allword_alltag_bayes_dict, all_class, top = 5):
    words = jieba.lcut(text)
    p_text_tag = []
    for tag in all_class:
        p = 0
        for word in words:
            p_bayes = allword_alltag_bayes_dict[word][tag]
            p += p_bayes
        p_text_tag.append([tag, p/len(words)])
    for tag, p in sorted(p_text_tag, reverse = True, key=lambda x:x[1])[:top]:
        print('文本属于类别%s的概率为%f'%(tag, p))

if __name__ == '__main__':
    path = 'train_tag_news.json'
    tag_word_dict, all_words, all_class = load_file(path)
    allword_alltag_bayes_dict = bayes(tag_word_dict, all_words, all_class)
    text = '菲律宾向越南示好归还所扣7艘越方渔船'
    predict(text, allword_alltag_bayes_dict, all_class)







