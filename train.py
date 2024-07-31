from argparse import Namespace, ArgumentParser
from dataset import *
from model.speaker_clustering import SpeakerClusterRoBERTa, SpeakerPreClassificationRoBERTa
from loss import ProtoSupConLoss, SpeakerDescLoss
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from collections import Counter

from sklearn.cluster import KMeans
from arguments import parse_arguments

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
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer
)
import random
import metric
import gc

import visualize

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
            'early_stopping_best_val': 0,
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
        train_state['early_stopping_best_val'] = train_state['val_acc'][-1]

    # 성능이 향상되면 모델을 저장합니다
    elif train_state['epoch_index'] >= 1:
        acc_tm1, acc_t = train_state['val_acc'][-2:]
        loss_tolerance = 0
         
        # 손실이 나빠지면
        if acc_t <= acc_tm1 - loss_tolerance:
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
        if acc_t > train_state['early_stopping_best_val']:
            torch.save(model.state_dict(), train_state['model_filename'])
            train_state['early_stopping_best_val'] = acc_t


        # 조기 종료 여부 확인
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def init_model_and_dataset(args:Namespace) -> Tuple[PDNCDataset, RobertaTokenizer, SpeakerClusterRoBERTa]:
    data_set = None

    model_name = "roberta-base"
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    data_set = PDNCDataset(args.dataset_path, tokenizer)


    '''bert_config = BertConfig.from_pretrained("bert-base-cased")
    bert_config.decoder_intermediate_size = args.decoder_hidden
    bert_config.feature_dimension = args.feature_dimension
    bert_config.feature_freedom = args.feature_freedom'''
    
    robert_config = RobertaConfig.from_pretrained(model_name)
    robert_config.decoder_intermediate_size = args.decoder_hidden
    robert_config.feature_dimension = args.feature_dimension
    robert_config.feature_freedom = args.feature_freedom
    
    model = SpeakerPreClassificationRoBERTa(robert_config, model_name)

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
    
    anchor_num = 100 # anchor 제한을 무력화하여 최대한 많이 저장할 수 있도록 함.
    
    # 우선 현재 batch의 구성을 살펴보도록 하자
    for i, speaker in enumerate(batch_dict["speaker"]):
        # narrative는 계산하지 않음
        if speaker != '':
            # 아직 한 번도 등록 안 된 앵커인지
            # 혹은 args로 준 앵커 저장 횟수만큼 저장이 덜 되었는지 확인
            cur_anchor_list = anchor_info.get(speaker, [])
            cur_anchor_count = len(cur_anchor_list)
            if cur_anchor_count == 0 or cur_anchor_count < anchor_num:
                # 새 특성 벡터 등록
                # batch를 1로 둔 특수한 경우의 코드
                cur_anchor_list.append(y_pred[i].view(1, -1))
                anchor_info[speaker] = cur_anchor_list
                
            elif cur_anchor_count == anchor_num:
                # 충분한 숫자가 앵커로 들어가 있을 때는 queue 구조로 먼저 들어온 것을 밀어내고 새 것을 넣는다.
                cur_anchor_list.pop(0)
                cur_anchor_list.append(y_pred[i].view(1, -1))
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
        pred_features = y_pred
        
        features_con = torch.cat((pred_features, saved_anchors), dim=0)
        
        speaker_con = list(batch_dict["speaker"])
        speaker_con.extend(anchor_labels)
        
        return features_con.reshape(1, features_con.shape[0], -1), speaker_con
    else:
        # 전체 예측에서 CLS 토큰 특징만 분리
        pred_features = y_pred
        
        return pred_features.unsqueeze(0), batch_dict["speaker"]

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
        # narrative는 centroid와 phi를 할당하지 않음
        if l != '':
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
            # centroid를 할당하지 않도록 함
            if assigned_cluster != -1:
                centroids[l] = km.cluster_centers_[assigned_cluster]

                norms = np.linalg.norm(features[indices] - centroids[l], axis=1, keepdims=True)
                norm_sum = np.sum(norms)

                #low = l_num * np.log(l_num + alpha)
                #phi[l] = norm_sum / l_num

                if norm_sum == 0:
                    phi[l] = 1.0
                else:
                    phi[l] = norm_sum / l_num
    
    # phi를 정규화
    phi_list = [phi[k] for k in phi.keys()]
    phi_max = max(phi_list)
    phi_min = min(phi_list)
    for k in phi.keys():
        phi[k] = (phi_max - phi[k] + alpha) / (phi_max - phi_min + alpha)

    fig = None
    if return_fig:
        # 구한 centroid와 phi를 저장해두자
        # fig를 그리기에 앞서 PCA를 통해 차원 축소 수행
        fig = visualize.plot_scatter_fig(features, labels)

    #plt.savefig("scatter_hue", bbox_inches='tight')
        
    return phi, centroids, fig

def detach_achors(achor_info:dict):
    for k in achor_info.keys(): 
        for i, feature in enumerate(achor_info[k]):
            achor_info[k][i] = feature.contiguous().detach()

def get_model_prediction(batch_dict, tokenizer, model, device):
    seq_length = np.sum(batch_dict["x_length"])
    if seq_length > tokenizer.model_max_length:
        # 문단 길이가 context window를 초과할 때
        # 문단에서 발화자가 달라지지 않고 전체가 다 한 사람이 말한 것이거나 설명문이거나
        # 여러 번 출력을 받아 pooling하는 방식으로 해결
        process_num = (
            seq_length // tokenizer.model_max_length) + 1
        avg_pool = nn.AvgPool2d(
            kernel_size=(process_num, 1), stride=1)
        latent_features = []
        desc_features = []
        
        error_accum = 0
                        
        for i in range(process_num):
            x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(
                seq_length, (i + 1) * tokenizer.model_max_length) - error_accum].reshape((1, -1))
            if x_i[0][0] != tokenizer.cls_token_id:
                x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(
                    seq_length, (i + 1) * tokenizer.model_max_length - 1) - error_accum].reshape((1, -1))
                x_i = torch.cat(
                    (torch.tensor([[tokenizer.cls_token_id]]).to(device), x_i), dim=1)
                error_accum += 1

            pred = model(input_ids=x_i, cls_idx=[0])
            latent_features.append(pred[0][0])
            desc_features.append(pred[1][0])

        latent_features = torch.cat(latent_features, dim=0).reshape(
            (1, process_num, latent_features[0].shape[0]))
        desc_features = torch.cat(desc_features, dim=0).reshape(
            (1, process_num, desc_features[0].shape[0]))
        latent_feature = avg_pool(latent_features)
        desc_feature = avg_pool(desc_features)

        y_pred = (latent_feature.squeeze(0), desc_feature.squeeze(0))
    else:
        y_pred = model(input_ids=batch_dict['x'], cls_idx=batch_dict["cls_index"])

    return y_pred

def get_model_inference(batch_dict, tokenizer, model, device):
    seq_length = np.sum(batch_dict["x_length"])
    if seq_length > tokenizer.model_max_length:
        # 문단 길이가 context window를 초과할 때
        # 문단에서 발화자가 달라지지 않고 전체가 다 한 사람이 말한 것이거나 설명문이거나
        # 여러 번 출력을 받아 pooling하는 방식으로 해결
        process_num = (
            seq_length // tokenizer.model_max_length) + 1
        avg_pool = nn.AvgPool2d(
            kernel_size=(process_num, 1), stride=1)
        latent_features = []
        
        error_accum = 0
                        
        for i in range(process_num):
            x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(
                seq_length, (i + 1) * tokenizer.model_max_length) - error_accum].reshape((1, -1))
            if x_i[0][0] != tokenizer.cls_token_id:
                x_i = batch_dict['x'][0][i * tokenizer.model_max_length - error_accum: min(
                    seq_length, (i + 1) * tokenizer.model_max_length - 1) - error_accum].reshape((1, -1))
                x_i = torch.cat(
                    (torch.tensor([[tokenizer.cls_token_id]]).to(device), x_i), dim=1)
                error_accum += 1

            pred = model.inference(input_ids=x_i, cls_idx=[0])
            latent_features.append(pred[0][0])

        latent_features = torch.cat(latent_features, dim=0).reshape(
            (1, process_num, latent_features[0].shape[0]))
        latent_feature = avg_pool(latent_features)

        y_pred = (latent_feature.squeeze(0), None)
    else:
        y_pred = model.inference(input_ids=batch_dict['x'], cls_idx=batch_dict["cls_index"])

    return y_pred

    
def train_model(args, data_set, model):
    print(args)
    
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

    max_grad_norm = 3.
    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    epoch_bar = tqdm.tqdm(desc='training routine', 
                                total=args.max_epochs,
                                position=0)
    

    #is_linux = platform.system() == "Linux"
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    tf_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "exp_{0}".format(time_str)), comment="Infini-BERT-speaker-cluster")

    hparam_dict={
        "feature_dimension": args.feature_dimension,
        "decoder_hidden": args.decoder_hidden,
        "lr": args.learning_rate,
        "weight decay": args.weight_decay,
        "seed": args.seed,
        "saved_anchor": args.saved_anchor_num,
        "detach_mem_stemp": args.detach_mems_step,
        "loss_sigma": args.loss_sig
    }


    device = torch.device("cuda" if args.cuda else "cpu")
    train_state = make_train_state(args)
    tokenizer = data_set.tokenizer
    
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    if args.train_books:
        train_books = [int(l) for l in args.train_books.split()]
    else:
        train_books = []
    # 
    if args.val_books:
        val_books = [int(l) for l in args.val_books.split()]
    else:
        val_books = []
    if args.test_books:
        test_books = [int(l) for l in args.test_books.split()]
    else:
        test_books = []
    
    book_bar = tqdm.tqdm(desc='books', 
                        total=len(train_books),
                        position=1)
    train_bar = tqdm.tqdm(desc="",
                        total=10,
                        position=2,
                        leave=True)
    val_bar = tqdm.tqdm(desc="",
                        total=10,
                        position=2, 
                        leave=True)
    try:
        for epoch_index in range(args.max_epochs):
            
            start_time = time.time()
            # 훈련 세트에 대한 순회
            random.shuffle(train_books)
            
            book_bar.total = len(train_books)
            for book in train_books:
                # 훈련 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
                data_set.set_book(book)
                batch_generator = generate_pdnc_batches(data_set,
                                                        max_seq_length=tokenizer.model_max_length,
                                                        device=device)
                train_bar.n = 0
                train_bar.total = data_set.get_num_batches(1)
                train_bar.desc = 'book{0} momentum'.format(book)
                
                cls_features = []
                speakers = []
                model.init_memories()
                model.eval()

                cur_title = data_set.get_cur_title()

                # momentum을 얻는다.
                # 의사적인 centroid를 얻는다.
                # 군집 밀집도 phi를 얻는다.
                for batch_index, batch_dict in enumerate(batch_generator):
                    should_detach_memories = (
                        (batch_index + 1) % args.detach_mems_step == 0)
                    
                    y_pred = get_model_inference(batch_dict, tokenizer, model, device)

                    # model에서 cls_idx의 feature만 나오니 이제 cls 위치를 따로 걸러낼 필요가 없어진다
                    cls_features.extend([feature.detach().cpu().numpy() for feature in y_pred[0]])
                    speakers.extend(batch_dict["speaker"])

                        # 진행 상태 막대 업데이트
                    train_bar.update(len(batch_dict["speaker"]))
                    if should_detach_memories:
                        model.detach_memories_()

                cluster_phi, centroids, _ = calc_cluster_phi(np.array(cls_features), speakers)
                
                if epoch_index == 0:
                    fig = visualize.plot_speaker_pie(speakers, cur_title)
                    tf_writer.add_figure(tag="book{0}/train/speaker_pie".format(book), figure=fig, global_step=epoch_index)
                
                model.train()
                cur_key = "book{0}".format(book)
                model.add_classifier(cur_key, len(set(speakers)), device)
                model.set_key(cur_key)
                
                iter_loss = []
                log_distance = True
                cluster_loss = ProtoSupConLoss(
                    cluster_phi, centroids, speakers, 
                    sig=args.loss_sig, 
                    log_distance=log_distance,
                    epsilon=args.loss_epsilon
                )
                
                desc_loss = SpeakerDescLoss(speakers)
                
                for book_iter_idx in range(args.book_train_iter):
                    data_set.set_book(book)
                    train_bar.n = 0
                    train_bar.desc = 'book{0}-{1}'.format(book, book_iter_idx)
                    train_bar.total = data_set.get_num_batches(1)
                    
                    batch_generator = generate_pdnc_batches(data_set,
                                                        max_seq_length=tokenizer.model_max_length,
                                                        device=device)
                    
                    running_loss = 0.0
                    running_acc = 0.0
                    model.init_memories()
                    
                    # 앵커로 저장해둔 특징 벡터와 레이블 리스트
                    feature_anchor_info = dict()
                    
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    paragraph_num = data_set.get_num_batches(1)
                    acu_paragraph_num = 0
                    loss = torch.zeros((1, 1)).to(device)
                    # 단계 1. 그레이디언트를 0으로 초기화합니다
                    optimizer.zero_grad()
                    pos_dis_dict = {s: [] for s in set(speakers)}
                    neg_dis_dict = {s: [] for s in pos_dis_dict.keys()}
                    
                    for batch_index, batch_dict in enumerate(batch_generator):
                        
                        # 현재 스텝에서 역전파해야하는지 확인
                        acu_paragraph_num += len(batch_dict["x_length"]) 
                        is_last = (acu_paragraph_num == paragraph_num)
                        should_detach_memories = ((batch_index + 1) % args.detach_mems_step == 0)
                        should_backward = (is_last or should_detach_memories)
                            
                        # 훈련 과정은 5단계로 이루어집니다

                        # --------------------------------------

                        # 단계 2. 출력을 계산합니다
                        y_pred = get_model_prediction(batch_dict, tokenizer, model, device)
                        
                        # 현재 데이터에 앵커를 추가한 feature, label 리스트를 얻는다.
                        latent_features, labels = concat_saved_anchors(feature_anchor_info, batch_dict, y_pred[0])
                        
                        # 단계 3. 손실을 계산합니다
                        cur_con_loss, step_pos_dis, step_neg_dis = cluster_loss(latent_features, [labels])
                        
                        desc_features = y_pred[1]
                        cur_desc_loss = desc_loss(desc_features.unsqueeze(0), [batch_dict["speaker"]])
                        acc_t = metric.get_pred_accuracies(desc_features.unsqueeze(0), [batch_dict["speaker"]], desc_loss.speaker_to_idx)
                        
                        # contrastive loss는 narrative 분류가 잘 된 만큼 loss 계산에 통합
                        cur_loss = acc_t * cur_con_loss + cur_desc_loss
                        
                        feature_anchor_info = update_anchor(feature_anchor_info, args.saved_anchor_num, batch_dict, y_pred[0])
                        
                        
                        running_acc += (acc_t - running_acc) / (batch_index + 1)
                        
                        if not torch.isinf(cur_loss).item():
                            loss += cur_loss
                            # 이동 손실과 이동 정확도를 계산합니다
                            running_loss += (cur_loss.item() - running_loss) / (batch_index + 1)
                        else:
                            tf_writer.add_text(tag="book{0}/train/log".format(book), text_string="Inf occured", global_step=epoch_index)
                        
                        tf_writer.add_scalar(tag="book{0}/train/step_loss".format(book), scalar_value=cur_con_loss, global_step=(epoch_index * args.book_train_iter + book_iter_idx) * paragraph_num + batch_index)
                        
                        
                        if log_distance:
                            for s in pos_dis_dict.keys():
                                if len(step_pos_dis.get(s, [])) > 0:
                                    pos_dis_dict[s].append(np.mean(step_pos_dis[s]))
                                if len(step_neg_dis.get(s, [])) > 0:
                                    neg_dis_dict[s].append(np.mean(step_neg_dis[s]))
                        
                        
                        
                        
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
                            del feature_anchor_info
                            feature_anchor_info = dict()
                            
                            # 단계 1. 그레이디언트를 0으로 초기화합니다
                            optimizer.zero_grad()
                        # -----------------------------------------

                        
                        # 진행 상태 막대 업데이트
                        train_bar.set_postfix(loss=running_loss, acc=running_acc)
                        train_bar.update(len(batch_dict["speaker"]))
                    
                    iter_loss.append(running_loss)
                    
                    if log_distance:
                        for s in pos_dis_dict.keys():
                            if len(pos_dis_dict[s]) > 0:
                                tf_writer.add_scalar(tag="book{0}/train/dis/{1}_pos_dis".format(book, s), scalar_value=np.mean(pos_dis_dict[s]), global_step=(epoch_index * args.book_train_iter + book_iter_idx))
                            if len(neg_dis_dict[s]) > 0:
                                tf_writer.add_scalar(tag="book{0}/train/dis/{1}_neg_dis".format(book, s), scalar_value=np.mean(neg_dis_dict[s]), global_step=(epoch_index * args.book_train_iter + book_iter_idx))

                # -----------------------------------------------------
                # book iter 훈련 종료
                
                data_set.set_book(book)
                train_bar.n = 0
                train_bar.total = data_set.get_num_batches(1)
                train_bar.desc = 'book{0} momentum'.format(book)
                batch_generator = generate_pdnc_batches(data_set,
                                                        max_seq_length=tokenizer.model_max_length,
                                                        device=device)

                cls_features = []
                speakers = []
                model.init_memories()
                model.eval()
                

                # 최적화 결과로 얻은 군집을 평가
                for batch_index, batch_dict in enumerate(batch_generator):
                    should_detach_memories = (
                        (batch_index + 1) % args.detach_mems_step == 0)
                    y_pred = get_model_inference(batch_dict, tokenizer, model, device)

                    cls_features.extend([feature.detach().cpu().numpy() for feature in y_pred[0]])
                    speakers.extend(batch_dict["speaker"])

                        # 진행 상태 막대 업데이트
                    train_bar.update(len(batch_dict["speaker"]))
                    if should_detach_memories:
                        model.detach_memories_()
                
                iter_loss = np.mean(iter_loss)

                if not args.debug:
                    homo, _ = metric.calc_adj_homo_with_af(labels=speakers, features=np.array(cls_features), draw_fig=False)
                    tf_writer.add_scalar(tag="book{0}/train/homog score".format(book), scalar_value=homo, global_step=epoch_index)
                
                #train_state['train_loss'].append(iter_loss)
                #train_state['train_acc'].append(running_acc)
                tf_writer.add_scalar(tag="book{0}/train/loss".format(book), scalar_value=iter_loss, global_step=epoch_index)
                tf_writer.add_scalar(tag="book{0}/train/acc".format(book), scalar_value=running_acc, global_step=epoch_index)

                if not args.debug:
                    book_bar.set_postfix(loss=running_loss, homo=homo)
                
                book_bar.update()

                tf_writer.flush()
            
            gc.collect()
            torch.cuda.empty_cache()

            book_bar.n = 0
            book_bar.total = len(val_books)
            
            running_loss = 0.
            running_acc = 0.
            val_batch_idx = 0
            running_score = 0.
            # 검증 세트에 대한 순회
            for book in val_books:
                # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
                data_set.set_book(book)
                cur_title = data_set.get_cur_title()

                val_bar.n = 0
                val_bar.total = data_set.get_num_batches(1)
                val_bar.desc = 'book{0} momentum'.format(book)
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
                    y_pred = get_model_inference(batch_dict, tokenizer, model, device)
                    
                    cls_features.extend([feature.detach().cpu().numpy() for feature in y_pred[0]])
                    speakers.extend(batch_dict["speaker"])

                    
                    # 진행 상태 막대 업데이트
                    val_bar.update(len(batch_dict["speaker"]))
                    if should_detach_memories:
                        model.detach_memories_()
                
                cluster_phi, centroids, _ = calc_cluster_phi(np.array(cls_features), speakers)
                
                if epoch_index == 0:
                    fig = visualize.plot_speaker_pie(speakers, cur_title)
                    tf_writer.add_figure(tag="book{0}/val/speaker_pie".format(book), figure=fig, global_step=epoch_index)

                if not args.debug:
                    homo, fig = metric.calc_adj_homo_with_af(labels=speakers, features=np.array(cls_features), draw_fig=True)
                    tf_writer.add_scalar(tag="book{0}/val/homog score".format(book), scalar_value=homo, global_step=epoch_index)

                    tf_writer.add_figure(tag="book{0}/val/cluster result".format(book), figure=fig, global_step=epoch_index)
                
                # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
                data_set.set_book(book)
                val_bar.n = 0
                val_bar.total = data_set.get_num_batches(1)
                val_bar.desc = 'book{0} momentum'.format(book)
                batch_generator = generate_pdnc_batches(data_set, 
                                                    max_seq_length=tokenizer.model_max_length,
                                                    device=device)
                book_loss = 0.
                model.eval()
                model.init_memories()
                
                cur_key = "book{0}".format(book)
                model.add_classifier(cur_key, len(set(speakers)), device)
                model.set_key(cur_key)
                
                # 앵커로 저장해둔 특징 벡터와 레이블 리스트
                feature_anchor_info = dict()
                
                gc.collect()
                torch.cuda.empty_cache()
                
                paragraph_num = data_set.get_num_batches(1)
                acu_paragraph_num = 0
                

                cluster_loss = ProtoSupConLoss(
                    cluster_phi, centroids, speakers, 
                    sig=args.loss_sig, 
                    log_distance=log_distance,
                    epsilon=args.loss_epsilon
                )
                desc_loss = SpeakerDescLoss(speakers)
                
                pos_dis_dict = {s: [] for s in set(speakers)}
                neg_dis_dict = {s: [] for s in pos_dis_dict.keys()}
                
                cls_features = []
                speakers = []

                for batch_index, batch_dict in enumerate(batch_generator):
                    acu_paragraph_num += len(batch_dict["x_length"]) 
                    is_last = (acu_paragraph_num == paragraph_num)
                    should_detach_memories = ((batch_index + 1) % args.detach_mems_step == 0)
                    should_backward = (is_last or should_detach_memories)
                    
                    # 단계 2. 출력을 계산합니다
                    y_pred = get_model_prediction(batch_dict, tokenizer, model, device)
                        
                    latent_features, labels = concat_saved_anchors(feature_anchor_info, batch_dict, y_pred[0])
                    
                    # 단계 3. 손실을 계산합니다
                    cur_con_loss, step_pos_dis, step_neg_dis = cluster_loss(latent_features, [labels])
                    
                        
                    cur_loss = cur_con_loss
                    
                    
                    for s in pos_dis_dict.keys():
                        if len(step_pos_dis.get(s, [])) > 0:
                            pos_dis_dict[s].append(np.mean(step_pos_dis[s]))
                        if len(step_neg_dis.get(s, [])) > 0:
                            neg_dis_dict[s].append(np.mean(step_neg_dis[s]))
                    
                    feature_anchor_info = update_anchor(feature_anchor_info, args.saved_anchor_num, batch_dict, y_pred[0])
                    
                    # 단계 3. 이동 손실과 이동 정확도를 계산합니다
                    book_loss += (cur_loss.item() - book_loss) / (batch_index + 1)
                    
                    #acc_t = metric.get_pred_accuracies(desc_features, [batch_dict["speaker"]], desc_loss.speaker_to_idx)
                    #running_acc += (acc_t - running_acc) / (batch_index + 1)
                    
                    # 진행 상태 막대 업데이트
                    val_bar.set_postfix(loss=book_loss)
                    val_bar.update(len(batch_dict["speaker"]))
                    
                    if should_backward:
                        # 메모리를 분리하여 계속 사용 가능하도록 함
                        model.detach_memories_()
                        del feature_anchor_info
                        feature_anchor_info = dict()
                    

                running_loss += (book_loss - running_loss) / (val_batch_idx + 1)
                if not args.debug:
                    running_score += (homo - running_score) / (val_batch_idx + 1)

                book_bar.set_postfix(loss=running_loss, homo=running_score)
                book_bar.update()

                tf_writer.add_scalar(tag="book{0}/val/loss".format(book), scalar_value=book_loss, global_step=epoch_index)
                #tf_writer.add_scalar(tag="book{0}/val/acc".format(book), scalar_value=running_acc, global_step=epoch_index)
                for s in pos_dis_dict.keys():
                    if len(pos_dis_dict[s]) > 0:
                        tf_writer.add_scalar(tag="book{0}/val/dis/{1}_pos_dis".format(book, s), scalar_value=np.mean(pos_dis_dict[s]), global_step=(epoch_index * args.book_train_iter + book_iter_idx))
                    if len(neg_dis_dict[s]) > 0:
                        tf_writer.add_scalar(tag="book{0}/val/dis/{1}_neg_dis".format(book, s), scalar_value=np.mean(neg_dis_dict[s]), global_step=(epoch_index * args.book_train_iter + book_iter_idx))
                #train_state['val_acc'].append(running_acc)

                tf_writer.flush()
                val_batch_idx += 1
            train_state['val_loss'].append(running_loss)
            train_state["val_acc"].append(running_score)
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
            book_bar.n = 0
            epoch_bar.set_postfix(best_val=train_state['early_stopping_best_val'] )
            epoch_bar.update()

            train_state['epoch_index'] += 1
            
    except KeyboardInterrupt:
        print("반복 중지")
        tf_writer.add_hparams(hparam_dict=hparam_dict, metric_dict={"adjusted rand index": train_state['early_stopping_best_val']})
        tf_writer.close()
        return train_state

    except torch.cuda.OutOfMemoryError:
        print("메모리 부족")
        tf_writer.add_hparams(hparam_dict=hparam_dict, metric_dict={"adjusted rand index": train_state['early_stopping_best_val']})
        tf_writer.close()
        return train_state
    
    tf_writer.add_hparams(hparam_dict=hparam_dict, metric_dict={"adjusted rand index": train_state['early_stopping_best_val']})
    tf_writer.close()
    return train_state

def save_train_result(train_state, file_path):
    with open(file_path, "wt", encoding="utf-8") as fp:
        fp.write(json.dumps(train_state))

def main():
    parser = parse_arguments()
    
    '''args = parser.parse_args([
        "--dataset_path", "data/pdnc/novels",
        "--train_books", "0",
        "--val_books", "",
        "--save_dir", "models_storage/test",
        "--model_state_file", "model.pth",
        "--feature_dimension", "2048",
        "--decoder_hidden", "1024",
        "--detach_mems_step", "5",
        "--learning_rate", "2e-6",
        "--weight_decay", "0.1",
        "--seed", "201456",
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