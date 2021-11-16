# import jieba
# sentences = set()
# with open('test.txt', encoding="utf8") as f:
#     for line in f:
#         sentence = line.strip()
#         sentences.add(" ".join(jieba.cut(sentence)))
# print('这是sentences', sentences)
import numpy as np
# x = np.array([2,3,4,5,6])
# print(x.shape)
# x = {1:[1,2,3], 2:[2,3,4,5], 3:[4,5,6]}
# num = np.sum([len(x[i]) for i in x.keys()])
# num = x.keys()
# print(num)
# s = []
# x = [1,2,3,4,5]
# q = [6,7,8,5]
# for i in x:
#     s.append(i)
# for j in q:
#     s.append(j)
# num = s.count(5)
# print(num)
x = {('a', 'b'):{'x':0.1,'w':0.2},2:{'q':0.5,'r':0.6}}
print(x[('a', 'b')]['x'])


