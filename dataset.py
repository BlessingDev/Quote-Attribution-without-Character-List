from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch

import pathlib
import pandas as pd
import json
import os


class PDNCDataset(Dataset):
    def __init__(
        self,
        novel_dir_path: str,
        tokenizer
    ):
        novels_dir = pathlib.Path(novel_dir_path)
        
        self.novel_title_list = []
        self.novel_dir_list = []
        for dir in novels_dir.iterdir():
            self.novel_title_list.append(dir.name)
            self.novel_dir_list.append(dir)
        
        self.cur_book = None
        self.book_num = len(self.novel_dir_list)
        self.tokenizer = tokenizer
        self._book_len = 0
        self._cur_book_index = 0
        
    def set_book(self, book_index: int):
        cur_book_dir = self.novel_dir_list[book_index]
        self._cur_book_index = book_index
        
        cur_book_file = pathlib.Path(os.path.join(cur_book_dir, "paragraph_list.json"))
        with cur_book_file.open() as f:
            self.cur_book = json.loads(f.read())
        
        self._book_len = len(self.cur_book)

    def __len__(self):
        return self._book_len
    
    def __getitem__(self, index):
        """파이토치 데이터셋의 주요 진입 메서드
        
        매개변수:
            index (int): 데이터 포인트에 대한 인덱스 
        반환값:
            데이터 포인트(paragraph, speaker)를 담고 있는 딕셔너리
        """
        data = self.cur_book[index]
        
        encode_result = self.tokenizer.encode(data["paragraph"])

        
        return {
            "x": encode_result,
            "speaker": data["speaker"],
            "type": data["type"],
            "x_length": len(encode_result)
        }
    
    def get_num_batches(self, batch_size):
        """배치 크기가 주어지면 데이터셋으로 만들 수 있는 배치 개수를 반환합니다
        
        매개변수:
            batch_size (int)
        반환값:
            배치 개수
        """
        return len(self) // batch_size
    
    def get_cur_title(self):
        return self.novel_title_list[self._cur_book_index]

        
def generate_pdnc_batches(dataset, max_seq_length, shuffle=True, drop_last=True, device=torch.device("cpu")):
    dataloader = DataLoader(dataset=dataset, batch_size=1,
                            shuffle=shuffle, drop_last=drop_last)
    
    cur_seq_length = 0
    unified_dict = {
        "x": [],
        "speaker": [],
        "type": [],
        "cls_index": [],
        "x_length": [],
    }
    cls_token = dataset.tokenizer.cls_token_id
    sep_token = dataset.tokenizer.sep_token_id
    

    for data_dict in dataloader:
        if cur_seq_length + data_dict['x_length'] > max_seq_length:
            if len(unified_dict["x"]) > 0:
                unified_dict["x"] = torch.Tensor([unified_dict["x"]]).int().to(device)
                yield unified_dict
                unified_dict = {
                    "x": data_dict["x"],
                    "speaker": data_dict["speaker"],
                    "type": data_dict["type"],
                    "cls_index": [0],
                    "x_length": [int(data_dict["x_length"])],
                }
                cur_seq_length = data_dict['x_length']
            else:
                unified_dict = {
                    "x": data_dict["x"],
                    "speaker": data_dict["speaker"],
                    "type": data_dict["type"],
                    "cls_index": [0],
                    "x_length": [int(data_dict["x_length"])],
                }
                unified_dict["x"] = torch.Tensor([unified_dict["x"]]).int().to(device)
                yield unified_dict
                
                unified_dict = {
                    "x": [],
                    "speaker": [],
                    "type": [],
                    "cls_index": [],
                    "x_length": [],
                }
        else:
            unified_dict["x"].extend(data_dict["x"])
            unified_dict["speaker"].extend(data_dict["speaker"])
            unified_dict["type"].extend(data_dict["type"])
            unified_dict["x_length"].append(int(data_dict["x_length"]))
            unified_dict["cls_index"].append(int(cur_seq_length))
            
            cur_seq_length += data_dict['x_length']
    
    # 시퀀스 길이가 다 안 차서 못 나간 샘플들 마지막에 모두 방출
    unified_dict["x"] = torch.Tensor([unified_dict["x"]]).int().to(device)
    yield unified_dict
    