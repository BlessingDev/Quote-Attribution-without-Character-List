import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from collections import Counter

class ProtoSupConLoss(nn.Module):
    def __init__(self, phi, centroids, speakers, sig=0.5, epsilon=10, log_distance=False):
        super(ProtoSupConLoss, self).__init__()
        self.phi = phi
        self.centroids = centroids
        self.sig = sig
        self.epsilon = epsilon
        
        if speakers is not None:
            self.speaker_counter = Counter(speakers)
        else:
            self.speaker_counter = dict()
        
        self.log_distance = log_distance
    
    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # features 차원 (b n d)
        # labels 차원 (b n)
        batch_size = features.shape[0]
        sample_size = features.shape[1]

        # 각 배치당 레이블 to index인 dict를 정의
        label_dicts = []

        for b in range(batch_size):
            label_dict = {}

            for idx, label in enumerate(labels[b]):
                # 모든 narrative는 계산에서 제외
                if label != '':
                    c = label
                    l = label_dict.get(c, [])
                    l.append(idx)
                    label_dict[c] = l
        
            label_dicts.append(label_dict)

        # 16, 2
        mu_pos = 1.0
        mu_neg = 1.0
        
        pos_dis_dict = dict()
        neg_dis_dict = dict()
        loss = torch.zeros((1, 1)).to(device)
        계산된_손실_수 = 0
        # for문으로 계산하는 부분을 추후 행렬 곱으로 개선하기를 바람
        for batch_idx, batch_dict in enumerate(label_dicts):
            # 각 레이블마다 positive negative 구하여 로스 계산
            for k in batch_dict.keys():
                if k != '':
                    # narrative가 아닐 때만 positive를 계산
                    # narrative는 negative만 계산된다.
                    # 각 speaker는 narrative에서 멀어지는 쪽으로만 학습. narrative는 서로 가까워지지 않는다.
                    positive_features = features[batch_idx, batch_dict[k]]
                    negative_indices = [idx for idx in range(sample_size) if k != labels[batch_idx][idx]]

                    # 각 레이블에 들어있는 샘플을 순회
                    for i in range(len(batch_dict[k])):
                        cur_feature = positive_features[i].unsqueeze(0)
                        if len(negative_indices) > 0:
                            # cos 유사도 대신 L2 거리를 사용
                            negative_loss = torch.cdist(cur_feature, features[batch_idx, negative_indices])
                            # 수치 안정성을 위해서
                            neg_dis = negative_loss

                            # negative 거리의 영향을 줄이기 위해 평균을 구함
                            # postive 거리의 평균합은 이제 negative 거리의 평균합보다 훨씬 작아져야 함
                            neg_x = torch.mean(neg_dis * mu_neg)
                            neg_sup = neg_x
                            
                            pos_dis = [torch.Tensor([0])]

                            for j in range(len(batch_dict[k])):
                                if i != j:
                                    anchor_contrast = torch.cdist(cur_feature, positive_features[j].unsqueeze(0))
                                    norm_contrast = anchor_contrast


                                    pos_dis.append(norm_contrast)

                            if len(pos_dis) > 1:
                                pos_dis.pop(0)
                            pos_dis = torch.stack(pos_dis)
                            pos_x = torch.mean(pos_dis)
                            norm_pos_sup = mu_pos * pos_x


                            # positive의 anchor가 없을 수 있음을 감안해야 함
                            pos_cent = self.centroids.get(k, None)
                            if pos_cent is not None:
                                pos_cent = torch.Tensor(self.centroids[k]).to(device)
                                pos_phi = torch.Tensor([self.phi[k]]).to(device)
                            
                            # 현재 positive가 아닌 모든 label의 집합
                            neg_label_set = set([l for l in self.centroids.keys() if l != k])

                            neg_cents = torch.stack([torch.Tensor(self.centroids[n_l]).to(device) for n_l in neg_label_set])
                            neg_phis = torch.stack([torch.Tensor([self.phi[n_l]]).to(device) for n_l in neg_label_set])
                            
                            if pos_cent is not None:
                                pos_cent_dis = torch.cdist(cur_feature, pos_cent.unsqueeze(0))
                                cent_contrast = torch.div(pos_cent_dis, pos_phi).view(1, -1)
                                norm_contrast = cent_contrast
                                pos_cent_logit = mu_pos * norm_contrast
                            
                            negative_cent_loss = torch.stack([torch.div(torch.cdist(cur_feature, neg_cents[idx].unsqueeze(0)).squeeze(), neg_phis[idx]) for idx in range(neg_cents.shape[0])])
                            # 수치 안정성을 위해서
                            logits = negative_cent_loss

                            neg_cent_x = torch.mean(logits * mu_neg)
                            negative_cent_logits = neg_cent_x

                            if pos_cent is not None:
                                pos_spcl = self.sig * norm_pos_sup + (1 - self.sig) * pos_cent_logit
                            else:
                                pos_spcl = norm_pos_sup
                            
                            neg_spcl = self.sig * neg_sup + (1 - self.sig) * negative_cent_logits

                            # pos_spcl의 제곱값에 penalty
                            total_prob = pos_spcl - neg_spcl + self.epsilon + torch.exp(pos_spcl) - 1
                            
                            # 화자 비율로 정규화
                            total_prob = total_prob / self.speaker_counter.get(k, 1)

                            loss += total_prob
                            계산된_손실_수 += 1

                            if self.log_distance:
                                pos_dis_list =  pos_dis_dict.get(k, [])
                                log_pos_dis = pos_x.item()
                                if pos_cent is not None:
                                    log_pos_dis += cent_contrast.item()
                                pos_dis_list.append(log_pos_dis)
                                pos_dis_dict[k] = pos_dis_list

                                neg_dis_list =  neg_dis_dict.get(k, [])
                                neg_dis_list.append(torch.mean(neg_dis).item() + torch.mean(negative_cent_loss).item())
                                neg_dis_dict[k] = neg_dis_list

        if 계산된_손실_수 > 0:
            norm_loss = torch.div(loss, 계산된_손실_수)
        else:
            norm_loss = torch.zeros((1, 1)).to(device)
        
        return norm_loss, pos_dis_dict, neg_dis_dict

class NarrativeDescLoss(nn.Module):
    def __init__(self):
        super(NarrativeDescLoss, self).__init__()
    
    
    def forward(self, features, labels):
        device = features.device
        
        batch_size = features.shape[0]
        sample_size = features.shape[1]
        
        total_loss = torch.zeros(1)
        
        bce_loss = nn.BCEWithLogitsLoss()
        num_labels = []
        for b in range(batch_size):
            batch_labels = []
            for s in range(sample_size):
                if labels[b][s] == '':
                    batch_labels.append(1)
                else:
                    batch_labels.append(0)
            
            num_labels.append(batch_labels)
        
        total_loss = bce_loss(features.squeeze(2), torch.Tensor(num_labels).to(device))
        
        return total_loss

class SpeakerDescLoss(nn.Module):
    def __init__(self, speakers):
        super(SpeakerDescLoss, self).__init__()
        
        speakers = sorted(list(set(speakers)))
        self.speaker_to_idx = dict()
        
        for i, speaker in enumerate(speakers):
            self.speaker_to_idx[speaker] = i
        
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, features, labels):
        batch_size = features.shape[0]
        sample_size = features.shape[1]
        
        num_labels = []
        for b in range(batch_size):
            batch_labels = []
            for s in range(sample_size):
                batch_labels.append(self.speaker_to_idx[labels[b][s]])
            
            num_labels.append(batch_labels)
        
        
        labels_tensor = torch.Tensor(num_labels).to(dtype=torch.long).to(features.device)
        
        loss = self.loss(features.permute((0, 2, 1)), labels_tensor)
        
        return loss
        
        

class SupConLoss(nn.Module):
    def __init__(self, phi, centroids, temperature=0.7):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.phi = phi
        self.centroids = centroids
    
    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # features 차원 (b n d)
        # labels 차원 (b n)
        batch_size = features.shape[0]
        sample_size = features.shape[1]
        dim_norm_factor = torch.sqrt(torch.Tensor([features.shape[2]])).to(device)

        # 각 배치당 레이블 to index인 dict를 정의
        label_dicts = []

        for b in range(batch_size):
            label_dict = {}

            for idx, label in enumerate(labels[b]):
                c = label
                l = label_dict.get(c, [])
                l.append(idx)
                label_dict[c] = l
        
            label_dicts.append(label_dict)

        loss = torch.zeros((1, 1)).to(device)
        # for문으로 계산하는 부분을 추후 행렬 곱으로 개선하기를 바람
        for batch_idx, batch_dict in enumerate(label_dicts):
            # 각 레이블마다 positive negative 구하여 로스 계산
            for k in batch_dict.keys():
                positive_features = features[batch_idx, batch_dict[k]]
                negative_indices = [idx for idx in range(sample_size) if idx not in batch_dict[k]]
                negative_features = features[batch_idx, negative_indices]

                positive_num = len(batch_dict[k])

                # 각 레이블에 들어있는 샘플을 순회
                for i in range(len(batch_dict[k])):
                    if negative_features.shape[0] > 0:
                        cur_feature = positive_features[i]
                        
                        # cos 유사도 대신 L2 거리를 사용
                        negative_loss = torch.div(torch.matmul(cur_feature, negative_features.T), self.temperature).view(1, -1)
                        # 수치 안정성을 위해서
                        logits = negative_loss / dim_norm_factor
                        negative_loss = torch.div(torch.matmul(cur_feature, negative_features.T), self.temperature).view(1, -1)
                        logits = logits / negative_features.shape[0]

                        negative_logits = torch.sum(torch.exp(logits))
                        pos_probs = torch.zeros(1).to(device)

                        for j in range(i + 1, len(batch_dict[k])):

                            anchor_contrast = torch.div(torch.matmul(cur_feature, positive_features[j]), self.temperature)
                            norm_contrast = anchor_contrast / dim_norm_factor

                            pos_logit = torch.exp(norm_contrast)

                            prob = torch.div(pos_logit, negative_logits)

                            pos_probs += torch.log(prob)

                        norm_probs = torch.multiply(torch.div(1, positive_num), pos_probs)

                    
                        total_prob = norm_probs
                        

                        total_prob = torch.multiply(-1, total_prob)

                        loss += total_prob
        

        norm_loss = torch.div(loss, sample_size)

        return norm_loss
