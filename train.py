import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from model import BertForTokenClassification
import os
from utils import fix_seed
from trainer import TransformersTrainer
import pandas as pd
from sklearn.utils import shuffle
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


tag_data = json.load(open('data/tags.json'))
tag_dic = tag_data['tag']
revers_tag_dic = tag_data['revers_tag']


class MyTrainer(TransformersTrainer):

    def read_data(self, file_name):
        texts = []
        labels = []

        with open(file_name)as file:
            str_list = []
            tag_list = []
            for line in file.readlines():
                line = line.strip()
                if len(line) > 0:
                    line = line.split(' ')
                    str_list.append(line[0])
                    tag_list.append(line[1])
                else:
                    texts.append(str_list)
                    tag_list_index = [-100]
                    for t in tag_list:
                        tag_list_index.append(tag_dic[t])
                    while len(tag_list_index) < self.max_length:
                        tag_list_index.append(-100)

                    labels.append(tag_list_index)
                    str_list = []
                    tag_list = []

        encoding = self.tokenizer_(text=texts,
                                   is_split_into_words=True)
        return encoding, labels

    def get_train_data(self):

        encoding, labels = self.read_data('data/train.conll')

        return DataLoader(Dataset(encoding, labels), self.batch_size)

    def get_dev_data(self):
        encoding, labels = self.read_data('data/dev.conll')

        return DataLoader(Dataset(encoding, labels), self.batch_size)

    def get_test_data(self):
        texts = []
        with open('data/final_test.txt')as file:
            for line in file.readlines():
                line = line.strip()
                texts.append(list(line.split('\x01')[1]))

        encoding = self.tokenizer_(text=texts,
                                   is_split_into_words=True)
        return DataLoader(Dataset(encoding), batch_size=1)

    def configure_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.lr)

    def train_step(self, data, mode):
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        labels = data['labels'].to(self.device).long()

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        output = outputs.logits.argmax(dim=2).cpu().numpy()
        return outputs.loss, output, labels.cpu().numpy()

    def predict_step(self, data):
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        output = outputs.logits.argmax(dim=2).cpu()[0]
        length = attention_mask.sum(dim=1) - 2
        out = output[1:length[0] + 1].numpy()
        res = [revers_tag_dic[str(o)] for o in out]
        return ' '.join(res)


def main(mode):
    if mode == 'train':
        do_train = True
        do_dev = True
        do_test = False
        load_model = False
    else:
        do_train = False
        do_dev = False
        do_test = True
        load_model = True

    fix_seed(2021)

    max_length = 78
    batch_size = 4
    lr = 5e-5

    model_name = 'best_model.p'
    model_path = '/data/home/joskazhao/ptm/roberta'
    if not os.path.exists(model_path):
        model_path = '/Users/joezhao/Documents/pretrain model/chinese_roberta_wwm_ext_L-12_H-768_A-12'
    if not os.path.exists(model_path):
        model_path = '/data/joska/ptm/roberta'

    if os.path.exists(model_name) and load_model:
        print('************load model************')
        # model = torch.load(model_name, map_location='cpu')
        model = torch.load(model_name)
    else:
        model = BertForTokenClassification.from_pretrained(model_path, num_labels=57)

    trainer = MyTrainer(model, batch_size=batch_size, lr=lr, max_length=max_length, model_path=model_path,
                        do_train=do_train, do_dev=do_dev, do_test=do_test, test_with_label=False,
                        save_model_name=model_name, attack=False, monitor='f1')
    trainer.configure_metrics(do_acc=False, do_f1=True, do_recall=True, do_precision=True, print_report=False)
    y_pred = trainer.run()
    if do_test:
        with open('result.txt', 'w')as res_file:
            with open('data/final_test.txt')as file:
                for line, pred in zip(file.readlines(), y_pred):
                    line = line.strip() + '\x01' + pred
                    res_file.write(line)
                    res_file.write('\n')


if __name__ == '__main__':
    import sys

    main(sys.argv[1])
