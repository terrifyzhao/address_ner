import json

texts = []
tags = []
tag_dic = {}
with open('data/train.conll')as file:
    str_list = []
    tag_list = []
    for line in file.readlines():
        line = line.strip()
        if len(line) > 0:
            line = line.split(' ')

            str_list.append(line[0])
            tag_list.append(line[1])
            tag_dic[line[1]] = tag_dic.get(line[1], 0) + 1
        else:
            text = ''.join(str_list)
            texts.append(text)
            tags.append(tag_list)
            str_list = []
            tag_list = []


with open('data/dev.conll')as file:
    str_list = []
    tag_list = []
    for line in file.readlines():
        line = line.strip()
        if len(line) > 0:
            line = line.split(' ')

            str_list.append(line[0])
            tag_list.append(line[1])
            tag_dic[line[1]] = tag_dic.get(line[1], 0) + 1
        else:
            text = ''.join(str_list)
            texts.append(text)
            tags.append(tag_list)
            str_list = []
            tag_list = []

dic = {}
revers_dic = {}
for k, v in tag_dic.items():
    dic[k] = len(dic)
revers_dic = {int(v): k for k, v in dic.items()}

res = {'tag': dic, 'revers_tag': revers_dic}
with open('data/tags.json', 'w')as file:
    json.dump(res, file)
print(123)
