import json, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer


def load_json(json_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)
    return json_data

class ECGDataset(Dataset):
    def __init__(self, split, args=None):
        
        self.args = args
        self.split = split
        
        data_file = '{}/{}_{}.json'.format(args.data_dir, split, args.task_type)
        self.data = load_json(data_file)
        print("\nLoaded {} {} data from {} !\n".format(len(self.data), split, data_file))
        
        self.v_id_map = eval(str(np.load(args.video_id_mapping_file, allow_pickle=True))) # dia1utt1: 1
        
        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone


        self.tokenizer = T5Tokenizer.from_pretrained(args.backbone,
            max_length=args.input_max_length)
        
        self.add_tokens = ['<utt>', '<emo>', '<dia>', '<caption>']
        self.tokenizer.unique_no_split_tokens = self.tokenizer.unique_no_split_tokens + self.add_tokens
        self.tokenizer.add_tokens(self.add_tokens)
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        
    def collate_fn(self, batch):
        batch_data = {'input_text': [], 'target_text': [], 'input_length': [], 'target_length': [], 'av_emb_ids': []}
        for i, tmp_data in enumerate(batch):
            dia_id, input_text, target_text = tmp_data["emo_utt_id"].split('utt')[0], tmp_data["input"], tmp_data["output"]
            if self.args.prefix:
                input_text = '{}: {}'.format(self.args.prefix, input_text)
            batch_data['input_text'].append(input_text)
            batch_data['target_text'].append(target_text)
            input_token_list = self.tokenizer.tokenize(input_text)
            batch_data['input_length'].append(len(input_token_list))
            batch_data['target_length'].append(len(self.tokenizer.tokenize(target_text)))
            
            av_ids = [0]*(self.args.input_max_length+1)

            
            for ii, x in enumerate(input_token_list):
                if (ii < self.args.input_max_length) and ('<extra_id' in x):
                    tmp_utt_id = int(x.split('_')[-1].replace('>',''))
                    if tmp_utt_id>50:
                        tmp_utt_id -= 50
                    dia_utt_id = '{}utt{}'.format(dia_id, tmp_utt_id)
                    av_ids[ii] = self.v_id_map[dia_utt_id]

            batch_data['av_emb_ids'].append(av_ids)
        
        batch_data['av_emb_ids'] = torch.tensor(batch_data['av_emb_ids'])
        
        tokenized_input = self.tokenizer(batch_data['input_text'], max_length=self.args.input_max_length, padding='max_length', truncation=True, return_tensors="pt") 
        batch_data['input_ids'] = tokenized_input['input_ids']
        batch_data['attention_mask'] = tokenized_input['attention_mask']

        tokenized_target = self.tokenizer(batch_data['target_text'], max_length=self.args.gen_max_length, padding='max_length', truncation=True, return_tensors="pt")
        target_ids = tokenized_target['input_ids']
        target_ids[~(tokenized_target['attention_mask'].bool())] = -100

        batch_data['target_ids'] = target_ids
        return batch_data
        
def get_dataloader(args, spilt, mode='train', batch_size=32, workers=4):
    dataset = ECGDataset(spilt, args)
    
    if mode == 'train':
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=workers, 
            pin_memory=True,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, 
            pin_memory=True,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False)
        
    return loader, dataset

