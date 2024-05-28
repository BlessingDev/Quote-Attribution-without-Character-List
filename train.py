from argparse import Namespace, ArgumentParser
from dataset import *
from model.speaker_clustering import SpeakerClusterModel
from cluster_supervised_contrastive_loss import ProtoSupConLoss
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import platform
import json
import datetime
import time
import tqdm.cli as tqdm
from transformers import (
    BertConfig, 
    BertTokenizer
)
import random
import metric
import gc

# gpu 번호 지정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def set_seed_everywhere(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, Namespace) else v
        for k, v in vars(namespace).items()
    }

def update_train_state(args, model, train_state):
    """후련 상태 업데이트합니다.
    
    콤포넌트:
     - 조기 종료: 과대 적합 방지
     - 모델 체크포인트: 더 나은 모델을 저장합니다

    :param args: 메인 매개변수
    :param model: 훈련할 모델
    :param train_state: 훈련 상태를 담은 딕셔너리
    :returns:
        새로운 훈련 상태
    """

    # 적어도 한 번 모델을 저장합니다
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # 성능이 향상되면 모델을 저장합니다
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]
        loss_tolerance = 0.001
         
        # 손실이 나빠지면
        if loss_t >= loss_tm1 - loss_tolerance:
            # 조기 종료 단계 업데이트
            train_state['early_stopping_step'] += 1
            print()
            print("early stopping step: {0}".format(train_state['early_stopping_step']))
            print()
        # 손실이 감소하면
        else:
            # 조기 종료 단계 재설정
            train_state['early_stopping_step'] = 0
        
        # 최상의 모델 저장
        if loss_t < train_state['early_stopping_best_val']:
            torch.save(model.state_dict(), train_state['model_filename'])
            train_state['early_stopping_best_val'] = loss_t


        # 조기 종료 여부 확인
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def init_model_and_dataset(args:Namespace) -> Tuple[PDNCDataset, BertTokenizer, SpeakerClusterModel]:
    data_set = None

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data_set = PDNCDataset(args.dataset_path, tokenizer)


    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    bert_config.decoder_intermediate_size = args.decoder_hidden
    bert_config.feature_dimension = args.latent_dimension
    
    model = SpeakerClusterModel(bert_config)

    if args.reload_from_files and os.path.exists(args.model_state_file):
        model.load_state_dict(torch.load(args.model_state_file))
        print("로드한 모델")
    else:
        print("새로운 모델")
    
    return data_set, tokenizer, model

def update_anchor(
    anchor_info: dict,
    anchor_num: int, 
    batch_dict: dict,
    y_pred: torch.Tensor
) -> dict:
    # 추후 앵커를 잊고, 갱신하는 구조가 필요할지도 모르겠다.
    # 현재는 계속 추가하는 구조만 있고, 설명 문단은 첫 배치에 바로 채워져버릴 것
    
    # 우선 현재 batch의 구성을 살펴보도록 하자
    for i, speaker in enumerate(batch_dict["speaker"]):
        # 아직 한 번도 등록 안 된 앵커인지
        # 혹은 args로 준 앵커 저장 횟수만큼 저장이 덜 되었는지 확인
        cur_anchor_list = anchor_info.get(speaker, [])
        cur_anchor_count = len(cur_anchor_list)
        if cur_anchor_count == 0 or cur_anchor_count < anchor_num:
            # 새 특성 벡터 등록
            # batch를 1로 둔 특수한 경우의 코드
            cur_anchor_list.append(y_pred[0][batch_dict["cls_index"][i]].view(1, -1))
            anchor_info[speaker] = cur_anchor_list
            
        elif cur_anchor_count == anchor_num:
            # 충분한 숫자가 앵커로 들어가 있을 때는 queue 구조로 먼저 들어온 것을 밀어내고 새 것을 넣는다.
            cur_anchor_list.pop(0)
            cur_anchor_list.append(y_pred[0][batch_dict["cls_index"][i]].view(1, -1))
            anchor_info[speaker] = cur_anchor_list
    
    return anchor_info

def concat_saved_anchors(
    anchor_info: dict,
    batch_dict: dict,
    y_pred: torch.Tensor
) -> Tuple[torch.Tensor, list]:
    # 차원을 맞추고 합친다.
    # 모델 파라미터와 연결된 gradient가 유지되는가가 관건
    
    device = y_pred.device
    if len(anchor_info.keys()) > 0:
        anchor_features = []
        anchor_labels = []
        for k in anchor_info.keys():
            anchor_features.extend(anchor_info[k])
            anchor_labels.extend([k] * len(anchor_info[k]))
        
        saved_anchors = torch.cat(anchor_features, dim=0).to(device)
        # 전체 예측에서 CLS 토큰 특징만 분리
        pred_features = torch.cat([y_pred[0, cls_idx, :].view(1, -1) for cls_idx in batch_dict["cls_index"]], dim=0)
        
        features_con = torch.cat((pred_features, saved_anchors), dim=0)
        
        speaker_con = list(batch_dict["speaker"])
        speaker_con.extend(anchor_labels)
        
        return features_con.reshape(1, features_con.shape[0], -1), speaker_con
    else:
        # 전체 예측에서 CLS 토큰 특징만 분리
        pred_features = torch.cat([y_pred[0, cls_idx, :].view(1, -1) for cls_idx in batch_dict["cls_index"]], dim=0)
        
        return pred_features.view(1, pred_features.shape[0], -1), batch_dict["speaker"]

def calc_cluster_phi(
        features, 
        labels,
        return_fig=False, 
        alpha=5
    ):
    # 레이블 별 feature상 centroid 구하기
    centroids = dict()
    phi = dict()
    n_labels = dict()
    label_set = set(labels)
    label_to_idx = dict()

    num_label_list = []
    for l_idx, l in enumerate(label_set):
        indices = [i for i, j in enumerate(labels) if j == l]
        n_labels[l] = len(indices)
        label_to_idx[l] = l_idx

        num_label_list.append((l, n_labels[l]))
    
    num_label_list.sort(key=lambda x: x[1], reverse=True)
    # centroid를 구할 때는 k-mean을 사용하여 가장 많이 군집된 레이블의 centroind를 할당
    km = KMeans(n_clusters=len(label_set)).fit(features)

    is_cluster_assigned = [False] * len(label_set)
    for l, l_num in num_label_list:
        indices = [i for i, j in enumerate(labels) if j == l]
        cluster_result = km.labels_[indices]
        cluster_count = Counter(cluster_result)
        
        cluster_counter_list = [(k, v) for k, v in cluster_count.items()]
        cluster_counter_list.sort(key=lambda x: x[1], reverse=True)

        assigned_cluster = -1
        idx = 0
        while assigned_cluster == -1 and idx < len(cluster_counter_list):
            c_idx, cluster_count = cluster_counter_list[idx]
            if not is_cluster_assigned[c_idx]:
                assigned_cluster = c_idx
                is_cluster_assigned[assigned_cluster] = True
            
            idx += 1

        # 못 찾았을 때
        # 남은 centroid 중 가장 가까운 것을 할당
        if assigned_cluster == -1:
            avg_point = np.average(features[indices].reshape((len(indices), -1)), axis=0, keepdims=True)
            
            # bool 인덱싱
            unassigned_centroids = km.cluster_centers_[[not b for b in is_cluster_assigned]]
            unassigned_centroids_idx = [i for i, b in enumerate(is_cluster_assigned) if b == False]
            norms = np.linalg.norm((avg_point - unassigned_centroids).reshape((len(unassigned_centroids_idx), -1)), axis=1)

            min_idx = np.argmin(norms)
            assigned_cluster = unassigned_centroids_idx[min_idx]
            is_cluster_assigned[assigned_cluster] = True


        centroids[l] = km.cluster_centers_[assigned_cluster]

        norms = np.linalg.norm(features[indices] - centroids[l], axis=1, keepdims=True)
        norm_sum = np.sum(norms)

        low = l_num * np.log(l_num + alpha)

        phi[l] = norm_sum / low
        if phi[l] == 0:
            phi[l] = 0.2

    fig = None
    if return_fig:
        # 구한 centroid와 phi를 저장해두자
        # fig를 그리기에 앞서 PCA를 통해 차원 축소 수행
        norm_x = StandardScaler().fit_transform(features)
        pca = PCA(n_components=3)
        principal_component = pca.fit_transform(norm_x)
        principal_df = pd.DataFrame(data=principal_component, columns = ['component1', 'component2', 'component3'])
        principal_df["labels"] = labels
        explained_ratio = sum(pca.explained_variance_ratio_)

        sns.set_style("darkgrid")
        fig = plt.figure(figsize=(16,9), dpi=300)
        ax = fig.add_subplot(projection='3d')
        cmap = ListedColormap(sns.color_palette("Set2", n_colors=len(label_set)).as_hex())
        label_list = sorted(list(label_set))
        for l in label_list:
            l_df = principal_df.loc[principal_df["labels"] == l]
            
            display_label = l
            if l == '':
                display_label = "narrative"
            
            sc = ax.scatter(
                l_df["component1"],
                l_df["component2"],
                l_df["component3"],
                c=cmap.colors[label_to_idx[l]],
                label=display_label,
                alpha=0.2
            )

        ax.set_title("3D Scatter (PCA Exp. Rate={0})".format(round(explained_ratio, 2)))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)

    #plt.savefig("scatter_hue", bbox_inches='tight')
        
    return phi, centroids, fig

def detach_achors(achor_info:dict):
    for k in achor_info.keys(): 
        for i, feature in enumerate(achor_info[k]):
            achor_info[k][i] = feature.contiguous().detach()

def train_model(args, data_set, model):
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
        maximize=False
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                            mode='min', factor=0.5,
                                            patience=1)

    epoch_bar = tqdm.tqdm(desc='training routine', 
                                total=args.max_epochs,
                                position=0)

    

    is_linux = platform.system() == "Linux"
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    tf_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "exp_{0}".format(time_str)), comment="Infini-BERT-speaker-cluster")

    hparam_dict={
        "laten_dimension": args.latent_dimension,
        "decoder_hidden": args.decoder_hidden,
        "lr": args.learning_rate,
        "weight decay": args.weight_decay,
        "seed": args.seed,
        "saved_anchor": args.saved_anchor_num,
        "detach_mem_stemp": args.detach_mems_step,
        "loss_temperature": args.temperature,
    }


    device = torch.device("cuda" if args.cuda else "cpu")
    train_state = make_train_state(args)
    tokenizer = data_set.tokenizer
    
    
    try:
        for epoch_index in range(args.max_epochs):
            
            start_time = time.time()
            # 훈련 세트에 대한 순회
            train_books = [
                0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16
            ]
            # 
            val_books = [
                3, 8, 12, 17
            ]
            test_books = [
                18, 19, 20
            ]
            book_bar = tqdm.tqdm(desc='books', 
                                total=len(train_books),
                                position=1)
            random.shuffle(train_books)
            for book in train_books:
                # 훈련 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
                data_set.set_book(book)
                train_bar = tqdm.tqdm(desc='book{0} momentum'.format(book),
                                            total=data_set.get_num_batches(1), 
                                            position=2, 
                                            leave=True)
                batch_generator = generate_pdnc_batches(data_set,
                                                    max_seq_length=tokenizer.model_max_length,
                                                    device=device)

                cls_features = []
                speakers = []
                model.init_memories()
                model.eval()

                # momentum을 얻는다.
                # 의사적인 centroid를 얻는다.
                # 군집 밀집도 phi를 얻는다.
                for batch_index, batch_dict in enumerate(batch_generator):
                    should_detach_memories = ((batch_index + 1) % args.detach_mems_step == 0)
                    seq_length = np.sum(batch_dict["x_length"])
                    if seq_length > tokenizer.model_max_length:
                        # 문단 길이가 context window를 초과할 때
                        # 문단에서 발화자가 달라지지 않고 전체가 다 한 사람이 말한 것이거나 설명문이거나
                        # 여러 번 출력을 받아 pooling하는 방식으로 해결
                        process_num = (seq_length // tokenizer.model_max_length) + 1
                        avg_pool = nn.AvgPool2d(kernel_size=(process_num, 1), stride=1)
                        pred_features = []
                        error_accum = 0
                        
                        for i in range(process_num):
                            x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(seq_length, (i + 1) * tokenizer.model_max_length) - error_accum].reshape((1, -1))
                            if x_i[0][0] != tokenizer.cls_token_id:
                                x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(seq_length, (i + 1) * tokenizer.model_max_length - 1) - error_accum].reshape((1, -1))
                                x_i = torch.cat((torch.tensor([[tokenizer.cls_token_id]]).to(device), x_i), dim=1)
                                error_accum += 1
                            
                            pred = model(x_i)
                            pred_features.append(pred[0][0][0])
                        
                        pred_features = torch.cat(pred_features, dim=0).reshape((1, process_num, pred_features[0].shape[0]))
                        y_pred = avg_pool(pred_features)
                        
                        y_pred = (y_pred, None)
                    else:
                        y_pred = model(batch_dict['x'])
                    
                    cls_features.extend([feature.detach().cpu().numpy() for feature in y_pred[0][0][batch_dict["cls_index"]]])
                    speakers.extend(batch_dict["speaker"])

                    
                    # 진행 상태 막대 업데이트
                    train_bar.update(len(batch_dict["speaker"]))
                    if should_detach_memories:
                        model.detach_memories_()
                
                cluster_phi, centroids, fig = calc_cluster_phi(np.array(cls_features), speakers)
                '''tf_writer.add_figure(
                    tag="book{0}/train/scatter".format(book), 
                    figure=fig,
                    global_step=epoch_index
                )'''
                
                homo, comp, v1, fig = metric.calc_v_measure_with_hdb(labels=speakers, features=np.array(cls_features))
                tf_writer.add_scalar(tag="book{0}/train/v1".format(book), scalar_value=v1, global_step=epoch_index)
                tf_writer.add_scalar(tag="book{0}/train/homogeneity".format(book), scalar_value=homo, global_step=epoch_index)
                tf_writer.add_scalar(tag="book{0}/train/completeness".format(book), scalar_value=comp, global_step=epoch_index)
                
                tf_writer.add_figure(tag="book{0}/train/cluster result".format(book), figure=fig, global_step=epoch_index)

                data_set.set_book(book)
                train_bar = tqdm.tqdm(desc='book{0}'.format(book),
                                            total=data_set.get_num_batches(1), 
                                            position=2, 
                                            leave=True)
                batch_generator = generate_pdnc_batches(data_set,
                                                    max_seq_length=tokenizer.model_max_length,
                                                    device=device)
                
                running_loss = 0.0
                model.init_memories()
                model.train()
                
                # 앵커로 저장해둔 특징 벡터와 레이블 리스트
                anchor_info = dict()
                
                gc.collect()
                torch.cuda.empty_cache()
                
                paragraph_num = data_set.get_num_batches(1)
                acu_paragraph_num = 0
                loss = torch.zeros((1, 1)).to(device)
                # 단계 1. 그레이디언트를 0으로 초기화합니다
                optimizer.zero_grad()
                cluster_loss = ProtoSupConLoss(cluster_phi, centroids, args.temperature)
                
                cls_features = []
                speakers = []

                for batch_index, batch_dict in enumerate(batch_generator):
                    
                    # 현재 스텝에서 역전파해야하는지 확인
                    acu_paragraph_num += len(batch_dict["x_length"]) 
                    is_last = (acu_paragraph_num == paragraph_num)
                    should_detach_memories = ((batch_index + 1) % args.detach_mems_step == 0)
                    should_backward = (is_last or should_detach_memories)
                        
                    # 훈련 과정은 5단계로 이루어집니다

                    # --------------------------------------

                    # 단계 2. 출력을 계산합니다
                    seq_length = np.sum(batch_dict["x_length"])
                    if seq_length > tokenizer.model_max_length:
                        # 문단 길이가 context window를 초과할 때
                        # 문단에서 발화자가 달라지지 않고 전체가 다 한 사람이 말한 것이거나 설명문이거나
                        # 여러 번 출력을 받아 pooling하는 방식으로 해결
                        process_num = (seq_length // tokenizer.model_max_length) + 1
                        avg_pool = nn.AvgPool2d(kernel_size=(process_num, 1), stride=1)
                        pred_features = []
                        error_accum = 0
                        
                        for i in range(process_num):
                            x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(seq_length, (i + 1) * tokenizer.model_max_length) - error_accum].reshape((1, -1))
                            if x_i[0][0] != tokenizer.cls_token_id:
                                x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(seq_length, (i + 1) * tokenizer.model_max_length - 1) - error_accum].reshape((1, -1))
                                x_i = torch.cat((torch.tensor([[tokenizer.cls_token_id]]).to(device), x_i), dim=1)
                                error_accum += 1
                            
                            pred = model(x_i)
                            pred_features.append(pred[0][0][0])
                        
                        pred_features = torch.cat(pred_features, dim=0).reshape((1, process_num, pred_features[0].shape[0]))
                        y_pred = avg_pool(pred_features)
                        
                        y_pred = (y_pred, None)
                    else:
                        y_pred = model(batch_dict['x'])
                    
                    # 현재 데이터에 앵커를 추가한 feature, label 리스트를 얻는다.
                    features, labels = concat_saved_anchors(anchor_info, batch_dict, y_pred[0])
                    
                    # 단계 3. 손실을 계산합니다
                    cur_loss = cluster_loss(features, [labels])
                    
                    anchor_info = update_anchor(anchor_info, args.saved_anchor_num, batch_dict, y_pred[0])
                    
                    cls_features.extend([feature.detach().cpu().numpy() for feature in y_pred[0][0][batch_dict["cls_index"]]])
                    speakers.extend(batch_dict["speaker"])
                    
                    loss += cur_loss
                    # 이동 손실과 이동 정확도를 계산합니다
                    running_loss += (cur_loss.item() - running_loss) / (batch_index + 1)
                    tf_writer.add_scalar(tag="book{0}/train/step_loss".format(book), scalar_value=cur_loss, global_step=(epoch_index) * paragraph_num + batch_index)
                    
                    #acc_t = None
                    #running_acc += (acc_t - running_acc) / (batch_index + 1)
                    
                    if should_backward:
                        # 특정 스텝 수마다 backward 수행. 그런 후에 detach하여 메모리 초기화
                        # 단계 4. 손실을 사용해 그레이디언트를 계산합니다
                        nan = torch.isnan(loss)
                        inf = torch.isinf(loss)
                        grad_fn = loss.grad_fn is not None
                        if not nan and not inf and grad_fn:
                            loss.backward()

                            # 단계 5. 옵티마이저로 가중치를 업데이트합니다
                            optimizer.step()
                        else:
                            if nan:
                                tf_writer.add_text(tag="book{0}/train/log".format(book), text_string="NAN occured", global_step=epoch_index)
                            if grad_fn:
                                tf_writer.add_text(tag="book{0}/train/log".format(book), text_string="gradient is empty", global_step=epoch_index)
                        
                        loss = torch.zeros((1, 1)).to(device)
                        
                        
                        # 메모리를 분리하여 계속 사용 가능하도록 함
                        model.detach_memories_()
                        del anchor_info
                        anchor_info = dict()
                        
                        # 단계 1. 그레이디언트를 0으로 초기화합니다
                        optimizer.zero_grad()
                    # -----------------------------------------

                    
                    # 진행 상태 막대 업데이트
                    train_bar.set_postfix(loss=running_loss)
                    train_bar.update(len(batch_dict["speaker"]))


                train_state['train_loss'].append(running_loss)
                #train_state['train_acc'].append(running_acc)
                tf_writer.add_scalar(tag="book{0}/train/loss".format(book), scalar_value=running_loss, global_step=epoch_index)

                book_bar.set_postfix(loss=running_loss, v1=v1)
                book_bar.update()

                tf_writer.flush()
            
            gc.collect()
            torch.cuda.empty_cache()

            book_bar = tqdm.tqdm(desc='books', 
                                total=len(val_books),
                                position=1)
            running_loss = 0.
            val_batch_idx = 0
            running_v1 = 0.
            # 검증 세트에 대한 순회
            for book in val_books:
                # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
                data_set.set_book(book)
                val_bar = tqdm.tqdm(desc='book{0} momentum'.format(book),
                                total=data_set.get_num_batches(1),
                                position=2, 
                                leave=True)
                batch_generator = generate_pdnc_batches(data_set, 
                                                    max_seq_length=tokenizer.model_max_length,
                                                    device=device)
                
                cls_features = []
                speakers = []
                model.init_memories()
                model.eval()

                # momentum을 얻는다.
                # 의사적인 centroid를 얻는다.
                # 군집 밀집도 phi를 얻는다.
                for batch_index, batch_dict in enumerate(batch_generator):
                    should_detach_memories = ((batch_index + 1) % args.detach_mems_step == 0)
                    seq_length = np.sum(batch_dict["x_length"])
                    if seq_length > tokenizer.model_max_length:
                        # 문단 길이가 context window를 초과할 때
                        # 문단에서 발화자가 달라지지 않고 전체가 다 한 사람이 말한 것이거나 설명문이거나
                        # 여러 번 출력을 받아 pooling하는 방식으로 해결
                        process_num = (seq_length // tokenizer.model_max_length) + 1
                        avg_pool = nn.AvgPool2d(kernel_size=(process_num, 1), stride=1)
                        pred_features = []
                        error_accum = 0
                        
                        for i in range(process_num):
                            x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(seq_length, (i + 1) * tokenizer.model_max_length) - error_accum].reshape((1, -1))
                            if x_i[0][0] != tokenizer.cls_token_id:
                                x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(seq_length, (i + 1) * tokenizer.model_max_length - 1) - error_accum].reshape((1, -1))
                                x_i = torch.cat((torch.tensor([[tokenizer.cls_token_id]]).to(device), x_i), dim=1)
                                error_accum += 1
                            
                            pred = model(x_i)
                            pred_features.append(pred[0][0][0])
                        
                        pred_features = torch.cat(pred_features, dim=0).reshape((1, process_num, pred_features[0].shape[0]))
                        y_pred = avg_pool(pred_features)
                        
                        y_pred = (y_pred, None)
                    else:
                        y_pred = model(batch_dict['x'])
                    
                    cls_features.extend([feature.detach().cpu().numpy() for feature in y_pred[0][0][batch_dict["cls_index"]]])
                    speakers.extend(batch_dict["speaker"])

                    
                    # 진행 상태 막대 업데이트
                    val_bar.update(len(batch_dict["speaker"]))
                    if should_detach_memories:
                        model.detach_memories_()
                
                cluster_phi, centroids, fig = calc_cluster_phi(np.array(cls_features), speakers)
                '''tf_writer.add_figure(
                    tag="book{0}/val/scatter".format(book), 
                    figure=fig,
                    global_step=epoch_index
                )'''

                homo, comp, v1, fig = metric.calc_v_measure_with_hdb(labels=speakers, features=np.array(cls_features))
                tf_writer.add_scalar(tag="book{0}/val/v1".format(book), scalar_value=v1, global_step=epoch_index)
                tf_writer.add_scalar(tag="book{0}/val/homogeneity".format(book), scalar_value=homo, global_step=epoch_index)
                tf_writer.add_scalar(tag="book{0}/val/completeness".format(book), scalar_value=comp, global_step=epoch_index)

                tf_writer.add_figure(tag="book{0}/val/cluster result".format(book), figure=fig, global_step=epoch_index)
                
                # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
                data_set.set_book(book)
                val_bar = tqdm.tqdm(desc='book{0}'.format(book),
                                total=data_set.get_num_batches(1),
                                position=2, 
                                leave=True)
                batch_generator = generate_pdnc_batches(data_set, 
                                                    max_seq_length=tokenizer.model_max_length,
                                                    device=device)
                book_loss = 0.
                model.eval()
                model.init_memories()
                
                # 앵커로 저장해둔 특징 벡터와 레이블 리스트
                anchor_info = dict()
                
                gc.collect()
                torch.cuda.empty_cache()
                
                paragraph_num = data_set.get_num_batches(1)
                acu_paragraph_num = 0
                
                cls_features = []
                speakers = []

                cluster_loss = ProtoSupConLoss(cluster_phi, centroids, args.temperature)

                for batch_index, batch_dict in enumerate(batch_generator):
                    acu_paragraph_num += len(batch_dict["x_length"]) 
                    is_last = (acu_paragraph_num == paragraph_num)
                    should_detach_memories = ((batch_index + 1) % args.detach_mems_step == 0)
                    should_backward = (is_last or should_detach_memories)
                    
                    # 단계 2. 출력을 계산합니다
                    seq_length = np.sum(batch_dict["x_length"])
                    if seq_length > tokenizer.model_max_length:
                        # 문단 길이가 context window를 초과할 때
                        # 문단에서 발화자가 달라지지 않고 전체가 다 한 사람이 말한 것이거나 설명문이거나
                        # 여러 번 출력을 받아 pooling하는 방식으로 해결
                        process_num = (seq_length // tokenizer.model_max_length) + 1
                        avg_pool = nn.AvgPool2d(kernel_size=(process_num, 1), stride=1)
                        pred_features = []
                        error_accum = 0
                        
                        for i in range(process_num):
                            x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(seq_length, (i + 1) * tokenizer.model_max_length) - error_accum].reshape((1, -1))
                            if x_i[0][0] != tokenizer.cls_token_id:
                                x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(seq_length, (i + 1) * tokenizer.model_max_length - 1) - error_accum].reshape((1, -1))
                                x_i = torch.cat((torch.tensor([[tokenizer.cls_token_id]]).to(device), x_i), dim=1)
                                error_accum += 1
                            
                            pred = model(x_i)
                            pred_features.append(pred[0][0][0])
                        
                        pred_features = torch.cat(pred_features, dim=0).reshape((1, process_num, pred_features[0].shape[0]))
                        y_pred = avg_pool(pred_features)
                        
                        y_pred = (y_pred, None)
                    else:
                        y_pred = model(batch_dict['x'])
                        
                    features, labels = concat_saved_anchors(anchor_info, batch_dict, y_pred[0])
                    
                    # 단계 3. 손실을 계산합니다
                    loss = cluster_loss(features, [labels])
                    
                    anchor_info = update_anchor(anchor_info, args.saved_anchor_num, batch_dict, y_pred[0])

                    cls_features.extend([feature.detach().cpu().numpy() for feature in y_pred[0][0][batch_dict["cls_index"]]])
                    speakers.extend(batch_dict["speaker"])
                    
                    # 단계 3. 이동 손실과 이동 정확도를 계산합니다
                    book_loss += (loss.item() - book_loss) / (batch_index + 1)
                    
                    #acc_t = None
                    #running_acc += (acc_t - running_acc) / (batch_index + 1)
                    
                    # 진행 상태 막대 업데이트
                    val_bar.set_postfix(loss=book_loss)
                    val_bar.update(len(batch_dict["speaker"]))
                    
                    if should_backward:
                        # 메모리를 분리하여 계속 사용 가능하도록 함
                        model.detach_memories_()
                        del anchor_info
                        anchor_info = dict()
                    

                running_loss += (loss.item() - running_loss) / (val_batch_idx + 1)
                running_v1 += (v1 - running_v1) / (val_batch_idx + 1)

                book_bar.set_postfix(loss=running_loss, v1=running_v1)
                book_bar.update()

                tf_writer.add_scalar(tag="book{0}/val/loss".format(book), scalar_value=running_loss, global_step=epoch_index)
                #train_state['val_acc'].append(running_acc)

                tf_writer.flush()
                val_batch_idx += 1
            train_state['val_loss'].append(running_loss)
            train_state["val_acc"].append(running_v1)
            train_state = update_train_state(args=args, model=model, 
                                            train_state=train_state)
            
            end_time = time.time()

            print(f"epoch 실행 시간: {end_time - start_time}")
            gc.collect()
            torch.cuda.empty_cache()
            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break
                
            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.set_postfix(best_val=train_state['early_stopping_best_val'] )
            epoch_bar.update()

            train_state['epoch_index'] += 1
            
    except KeyboardInterrupt:
        print("반복 중지")
        tf_writer.add_hparams(hparam_dict=hparam_dict, metric_dict={"v1": train_state['val_acc'][-1]})
        tf_writer.close()
        return train_state

    except torch.cuda.OutOfMemoryError:
        print("메모리 부족")
        tf_writer.add_hparams(hparam_dict=hparam_dict, metric_dict={"v1": train_state['val_acc'][-1]})
        tf_writer.close()
        return train_state
    
    tf_writer.add_hparams(hparam_dict=hparam_dict, metric_dict={"v1": train_state['val_acc'][-1]})
    tf_writer.close()
    return train_state

def save_train_result(train_state, file_path):
    with open(file_path, "wt", encoding="utf-8") as fp:
        fp.write(json.dumps(train_state))

def mt_args():
    return Namespace(dataset_csv="datas/jeonla_dialect_jamo_integration.csv",
                vectorizer_file="vectorizer.json",
                model_state_file="model.pth",
                train_state_file="train_state.json",
                log_json_file="logs/train_at_{time}.json",
                save_dir="model_storage/stan-JL_jamo",
                reload_from_files=True,
                expand_filepaths_to_save_dir=True,
                cuda=True,
                seed=1337,
                learning_rate=5e-4,
                batch_size=200,
                num_epochs=100,
                early_stopping_criteria=3,         
                source_embedding_size=64, 
                target_embedding_size=64,
                encoding_size=128,
                catch_keyboard_interrupt=True)

def main():
    parser = ArgumentParser()
    
    parser.add_argument(
        "--dataset_path", 
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--model_state_file",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--latent_dimension",
        type=int,
        default=1000
    )
    
    parser.add_argument(
        "--decoder_hidden",
        type=int,
        default=1024
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=10025
    )
    
    parser.add_argument(
        "--learning_rate",
        default=2e-8,
        type=float
    )
    
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=int
    )
    
    parser.add_argument(
        "--warmup_steps",
        default=0,
    )
    
    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float
    )

    parser.add_argument(
        "--saved_anchor_num",
        default=2,
        type=int
    )
    
    parser.add_argument(
        "--detach_mems_step",
        default=2,
        type=int
    )

    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float
    )

    parser.add_argument(
        "--early_stopping_criteria",
        default=5
    )
    
    parser.add_argument(
        "--max_epochs",
        default=10,
        type=int
    )
    
    parser.add_argument(
        "--reload_from_files",
        default=False
    )
    
    parser.add_argument(
        "--expand_filepaths_to_save_dir",
        default=True
    )
    
    parser.add_argument(
        "--log_json_file",
        default="train_at_{time}.json"
    )
    
    parser.add_argument(
        "--catch_keyboard_interrupt",
        default=True
    )
    
    '''args = parser.parse_args([
        "--dataset_path", "data/pdnc/novels",
        "--save_dir", "models_storage/test",
        "--model_state_file", "model.pth",
        "--latent_dimension", "1024",
        "--decoder_hidden", "4096",
        "--saved_anchor_num", "3",
        "--detach_mems_step", "15",
        "--learning_rate", "1e-6",
        "--weight_decay", "0.0",
        "--seed", "201456",
        "--temperature", "0.7",
    ])'''

    args = parser.parse_args()
    # console argument 구성 및 받아오기

    if args.expand_filepaths_to_save_dir:
        args.model_state_file = os.path.join(args.save_dir,
                                            args.model_state_file)
        
        args.log_dir = os.path.join(args.save_dir, "log/")
        
        args.log_json_file = os.path.join(args.log_dir,
                                            args.log_json_file)
        
        
        print("파일 경로: ")
        print("\t{}".format(args.model_state_file))
        print("\t{}".format(args.log_json_file))
    
    args.cuda = True
    # CUDA 체크
    if not torch.cuda.is_available():
        args.cuda = False

    device = torch.device("cuda" if args.cuda else "cpu")
        
    print("CUDA 사용 여부: {}".format(args.cuda))

    # 재현성을 위해 시드 설정
    set_seed_everywhere(args.seed, args.cuda)

    # 디렉토리 처리
    handle_dirs(args.save_dir)
    handle_dirs(args.log_dir)
    handle_dirs('/'.join(args.log_json_file.split('/')[:-1]))

    data_set, _, model = init_model_and_dataset(args)

    model = model.to(device)

    train_state = train_model(args, data_set, model)

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")

    args.log_json_file = args.log_json_file.format(
        time=time_str
    )

    model_log_name = "model_{time}.pth".format(time=time_str)

    model_log_name = os.path.join(args.save_dir,
                                model_log_name)
    os.rename(args.model_state_file, model_log_name)
    args.model_state_file = model_log_name

    args_dict = namespace_to_dict(args)
    args_dict.update(train_state)
    save_train_result(args_dict, args_dict["log_json_file"])
    

if __name__ == "__main__":
    main()