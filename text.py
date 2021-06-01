from transformers import BertTokenizerFast

model_path = '/Users/joezhao/Documents/pretrain model/chinese_roberta_wwm_ext_L-12_H-768_A-12'
tokenizer = BertTokenizerFast.from_pretrained(model_path)
print(tokenizer.decode(tokenizer('朝阳区嘉翔大厦A座0000室')['input_ids']))