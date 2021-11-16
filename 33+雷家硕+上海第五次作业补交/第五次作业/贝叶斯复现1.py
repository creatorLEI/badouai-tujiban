import numpy as np
from collections import defaultdict
import jieba
import json

jieba.initialize()

def load_data(path):
    tag_word_dict = defaultdict(list)
    all_words = set()
    all_class = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            line = json.loads(line) #读取每句话
            tag = line['tag']   #取每句话的tag
            title = line['title']   #取每句话的标题，作为分类对象
            words = jieba.lcut(title)   #切每句话的标题
            tag_word_dict[tag].append(words)
            all_words = all_words.union(set(words))
            all_class[tag] = len(all_class)
        return tag_word_dict, all_words, all_class
    #得到 {每个类：所有属于这个类的切好词的话}

#按句统计每个类别出现的概率
def tag_pro(tag_word_dict):
    total_sentence_num = sum(len(sentence_list) for sentence_list in tag_word_dict.values())
    tag_pro_dict = {}
    for tag, sentences_list in tag_word_dict.items():
        tag_pro_dict[tag] = len(sentences_list)/total_sentence_num
    return tag_pro_dict

#计算某个词在某类当中出现的概率，按句子算，出现在不同句子当中才算多次，一个句子中出现多次只算一次
def word_in_tag_pro(tag_word_dict):
    word_in_tag_pro_dict = {}
    for tag, sentence_list in tag_word_dict.items():
        word_appearance_dict = defaultdict(set)
        for index, sentence in enumerate(sentence_list):
            for word in sentence:
                word_appearance_dict[word].add(index)
                #defaultdict + set是常用的分类方法
                #记录这个词在这个类别中所有句子中出现的次数，每句话出现只记录一次，用set实现
        for word, sentence_appearence in word_appearance_dict.items():
            key = word + '_' + tag
            #这种命名方法可以用get调用的，不需要字典套字典了
            word_in_tag_pro_dict[key] = len(sentence_appearence)/len(sentence_list)
    return word_in_tag_pro_dict

#某个词的全概率p(wx) = p(wx|ax)*p(ax) + p(wx|as)*p(as) +...
def word_pro(tag_word_dict,all_words):
    word_in_tag_pro_dict = word_in_tag_pro(tag_word_dict)
    tag_pro_dict = tag_pro(tag_word_dict)
    word_pro_dict = {}
    for word in all_words:
        p_word = 0
        for tag, p_tag in tag_pro_dict.items():
            key = word + '_' + tag
            # p_word_in_tag = word_in_tag_pro_dict[key]
            p_word_in_tag = word_in_tag_pro_dict.get(key, 0)
            '''
            
            '''
            p_word += p_word_in_tag * p_tag
        word_pro_dict[word] = p_word
    return word_pro_dict

def bayes(tag_word_dict, all_words, all_class):
    tag_pro_dict = tag_pro(tag_word_dict)
    word_in_tag_pro_dict = word_in_tag_pro(tag_word_dict)
    word_pro_dict = word_pro(tag_word_dict, all_words)
    all_bayes_dict = {}
    for word in all_words:
        for tag in all_class:
            p_tag = tag_pro_dict[tag]
            key = word +'_'+ tag
            p_word_in_tag = word_in_tag_pro_dict.get(key, 0)
            p_word = word_pro_dict[word]
            p_bayes = p_tag * p_word_in_tag / p_word
            all_bayes_dict[key] = p_bayes
    return all_bayes_dict

def predict(text, all_bayes_dict, all_class, top = 5):
    words = jieba.lcut(text)
    record = [] #得到属于每个类别的概率
    for tag in all_class:   #既然判断的是这句话的每个词属于哪一类，就先用这个做for
        p_tag = 0
        for word in words:
            key = word + '_' + tag
            p_bayes = all_bayes_dict[key]
            p_tag += p_bayes
        record.append([tag, p_tag/len(words)])
    for tag, p_text_tag in sorted(record, reverse=True,
                                  key=lambda x:x[1])[:top]:
        print('属于[%s]类的概率为%f'%(tag, p_text_tag))

if __name__ == '__main__':
    path = 'train_tag_news.json'
    tag_word_dict, all_words, all_class = load_data(path)
    all_bayes_dict = bayes(tag_word_dict, all_words, all_class)
    text = '菲律宾向越南示好归还所扣7艘越方渔船'
    predict(text, all_bayes_dict, all_class)

