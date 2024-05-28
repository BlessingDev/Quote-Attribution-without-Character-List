import pathlib
import pandas as pd
import pickle as pkl
import csv
import json


def pdnc_file_to_paragraph_dict(qdf, ntext):
    
    def get_q_start_end(qdf, idx):
        if idx < len(qdf):
            cur_q_span = json.loads(qdf.iloc[idx]["qSpan"])
            return [cur_q_span[0][0], cur_q_span[-1][1]]
        else:
            return [0, 0]
        
        
    paragraph_list = []

    paragraphs = ntext.split('\n\n')
    paragraphs = [p for p in paragraphs if len(p) > 0]
    
    utt_count = 0
    acu_len = 0
    cur_q = 0
    cur_q_start_end = get_q_start_end(qdf, cur_q)
    text_length = len(ntext)
    for p in paragraphs:
        p_dict = {
            "paragraph": "",
            "speaker": None,
            "type": ""
        }
        p_start = acu_len
        p_end = acu_len + len(p)
        while p_end < text_length and ntext[p_end] == '\n':
            p_end += 1
        
        p_dict["paragraph"] = p.replace('\n', ' ')
        
        if p_start <= cur_q_start_end[0] and \
            p_end >= cur_q_start_end[1]:
            # 문단 안에 정확하게 발화문 전체가 등장할 때
            p_dict["speaker"] = qdf.iloc[cur_q]["speaker"]
            p_dict["type"] = "utterance"
            
            cur_q += 1
            cur_q_start_end = get_q_start_end(qdf, cur_q)
            
            paragraph_list.append(p_dict)
            utt_count += 1
            
            # q_index를 한 칸 전진시켰음에도 같은 문단 안에서 구분되는 발화문 발견시
            # 즉 한 문단 안에서 복수 인원이 발화하는 경우
            while p_start <= cur_q_start_end[0] and \
                p_end >= cur_q_start_end[1]:
                # 이전 발화와 해당 발화의 화자가 다른지를 검사
                if paragraph_list[-1]["speaker"] != qdf.iloc[cur_q]["speaker"]:
                    # 우선 이전 문단에서 현재 발화문까지의 구간을 분리
                    paragraph_list[-1]["paragraph"] = ntext[p_start:cur_q_start_end[0] - 1].replace('\n', ' ')
                    
                    # 문단을 분할하여 새로운 문단 구성
                    p_dict = {
                        "paragraph": "",
                        "speaker": None,
                        "type": ""
                    }
                    
                    p_dict["paragraph"] = ntext[cur_q_start_end[0] - 1:p_end].replace('\n', ' ')
                    p_dict["speaker"] = qdf.iloc[cur_q]["speaker"]
                    p_dict["type"] = "utterance"
                    
                    cur_q += 1
                    cur_q_start_end = get_q_start_end(qdf, cur_q)
                    
                    paragraph_list.append(p_dict)
                    utt_count += 1
                # 화자가 다르지 않다면 같은 문단으로 두어도 무방함
                else:
                    cur_q += 1
                    cur_q_start_end = get_q_start_end(qdf, cur_q)
        elif p_start <= cur_q_start_end[0] and \
            p_end >= cur_q_start_end[0]:
            # 여러 문단에 걸쳐서 한 등장인물의 발화가 등장할 때
            # 문단: [---------]
            # 발화:     [----------]
            p_dict["speaker"] = qdf.iloc[cur_q]["speaker"]
            p_dict["type"] = "utterance"
            
            paragraph_list.append(p_dict)
            utt_count += 1
        elif p_start < cur_q_start_end[1] and \
            p_end >= cur_q_start_end[1]:
            # 여러 문단에 걸쳐서 한 등장인물의 발화가 등장할 때
            # 발화가 종료됨
            # 문단:      [---------]
            # 발화: [----------]
            p_dict["speaker"] = qdf.iloc[cur_q]["speaker"]
            p_dict["type"] = "utterance"
            
            cur_q += 1
            cur_q_start_end = get_q_start_end(qdf, cur_q)
            
            paragraph_list.append(p_dict)
            utt_count += 1
            
            # q_index를 한 칸 전진시켰음에도 같은 문단 안에서 구분되는 발화문 발견시
            # 즉 한 문단 안에서 복수 인원이 발화하는 경우
            while p_start <= cur_q_start_end[0] and \
                p_end >= cur_q_start_end[1]:
                # 이전 발화와 해당 발화의 화자가 다른지를 검사
                if paragraph_list[-1]["speaker"] != qdf.iloc[cur_q]["speaker"]:
                    # 우선 이전 문단에서 현재 발화문까지의 구간을 분리
                    paragraph_list[-1]["paragraph"] = ntext[p_start:cur_q_start_end[0] - 1].replace('\n', ' ')
                    
                    # 문단을 분할하여 새로운 문단 구성
                    p_dict = {
                        "paragraph": "",
                        "speaker": None,
                        "type": ""
                    }
                    
                    p_dict["paragraph"] = ntext[cur_q_start_end[0] - 1:p_end].replace('\n', ' ')
                    p_dict["speaker"] = qdf.iloc[cur_q]["speaker"]
                    p_dict["type"] = "utterance"
                    
                    cur_q += 1
                    cur_q_start_end = get_q_start_end(qdf, cur_q)
                    
                    paragraph_list.append(p_dict)
                    utt_count += 1
                # 화자가 다르지 않다면 같은 문단으로 두어도 무방함
                else:
                    cur_q += 1
                    cur_q_start_end = get_q_start_end(qdf, cur_q)
        else:
            p_dict["speaker"] = ""
            p_dict["type"] = "narrative"
            
            paragraph_list.append(p_dict)
        
        acu_len += len(p)
        while acu_len < text_length and ntext[acu_len] == '\n':
            acu_len += 1
    
    print("paragraph length: {0}".format(len(paragraphs)))
    print("number of utterances: {0}".format(len(qdf)))
    print("detected utterance count: {0}".format(utt_count))
    print("")
    
    return paragraph_list

def process_whole_novel():
    novels = []
    with open('data/pdnc/ListOfNovels.txt', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            novels.append(row[-1])

    for novel in novels:
        print(novel)

        qdf = pd.read_csv('data/pdnc/novels/'+novel+'/quotations.csv', index_col=0)
        charDict = pkl.load(open('data/pdnc/novels/'+novel+'/charDict.pkl', 'rb'))
            
        print("Number of characters: {}".format(len(charDict['id2names'])))
                                    
        with open('data/pdnc/novels/'+novel+'/text.txt', 'r') as f:
            ntext = f.read().strip()

        p_list = pdnc_file_to_paragraph_dict(qdf, ntext)

        with open('data/pdnc/novels/'+novel+'/paragraph_list.json', "w+") as f:
            f.write(json.dumps(p_list))

def process_a_novel(novel_name: str):
    novel = novel_name
    print(novel)

    qdf = pd.read_csv('data/pdnc/novels/'+novel+'/quotations.csv', index_col=0)
    charDict = pkl.load(open('data/pdnc/novels/'+novel+'/charDict.pkl', 'rb'))
            
    print("Number of characters: {}".format(len(charDict['id2names'])))
                                    
    with open('data/pdnc/novels/'+novel+'/text.txt', 'r') as f:
        ntext = f.read().strip()

    p_list = pdnc_file_to_paragraph_dict(qdf, ntext)

    with open('data/pdnc/novels/'+novel+'/paragraph_list.json', "w+") as f:
        f.write(json.dumps(p_list))

#process_a_novel("AnneOfGreenGables")
process_whole_novel()