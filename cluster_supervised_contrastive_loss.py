import torch
import torch.nn as nn
import torch.functional as F

class ProtoSupConLoss(nn.Module):
    def __init__(self, phi, centroids, temperature=0.7):
        super(ProtoSupConLoss, self).__init__()
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
        mse_loss = nn.MSELoss(reduction="sum")
        # for문으로 계산하는 부분을 추후 행렬 곱으로 개선하기를 바람
        for batch_idx, batch_dict in enumerate(label_dicts):
            # 각 레이블마다 positive negative 구하여 로스 계산
            for k in batch_dict.keys():
                positive_features = features[batch_idx, batch_dict[k]]

                positive_num = len(batch_dict[k])

                # 각 레이블에 들어있는 샘플을 순회
                for i in range(len(batch_dict[k])):
                    negative_indices = [idx for idx in range(sample_size) if idx != i]
                    cur_feature = positive_features[i]
                    if len(negative_indices) > 0:
                        # cos 유사도 대신 L2 거리를 사용
                        #negative_loss = torch.div(torch.matmul(cur_feature, negative_features.T), self.temperature).view(1, -1)
                        negative_loss = torch.stack([torch.div(mse_loss(cur_feature, features[batch_idx, neg_idx]), self.temperature) for neg_idx in negative_indices]).view(1, -1)
                        # 수치 안정성을 위해서
                        logits = negative_loss / dim_norm_factor
                        #logits = logits / negative_features.shape[0]

                        neg_sup = torch.sum(torch.exp(logits))
                        
                        pos_sub = torch.zeros(1).to(device)

                        for j in range(len(batch_dict[k])):
                            if i != j:
                                #anchor_contrast = torch.div(torch.matmul(cur_feature, positive_features[j]), self.temperature)
                                anchor_contrast = torch.div(mse_loss(cur_feature, positive_features[j]), self.temperature)
                                norm_contrast = anchor_contrast / dim_norm_factor

                                pos_logit = torch.exp(norm_contrast)

                                pos_sub += pos_logit

                        #norm_probs = torch.multiply(torch.div(1, positive_num), pos_sub)

                    
                        pos_cent = torch.Tensor(self.centroids[k]).to(device)
                        pos_phi = torch.Tensor([self.phi[k]]).to(device)
                        
                        # 현재 positive가 아닌 모든 label의 집합
                        neg_label_set = set([l for l in self.centroids.keys() if l != k])

                        neg_cents = torch.stack([torch.Tensor(self.centroids[n_l]).to(device) for n_l in neg_label_set])
                        neg_phis = torch.stack([torch.Tensor([self.phi[n_l]]).to(device) for n_l in neg_label_set])
                        
                        #cent_contrast = torch.div(torch.matmul(cur_feature, pos_cent), pos_phi).view(1, -1)
                        cent_contrast = torch.div(mse_loss(cur_feature, pos_cent), pos_phi).view(1, -1)
                        norm_contrast = cent_contrast / dim_norm_factor
                        pos_cent_logit = torch.exp(norm_contrast)
                        
                        #negative_cent_loss = torch.div(torch.matmul(cur_feature, neg_cents.T), neg_phis)
                        negative_cent_loss = torch.stack([torch.div(mse_loss(cur_feature, neg_cents[idx]), neg_phis[idx]) for idx in range(neg_cents.shape[0])])
                        # 수치 안정성을 위해서
                        logits = negative_cent_loss / dim_norm_factor
                        #logits = logits / neg_cents.shape[0]

                        negative_cent_logits = torch.sum(torch.exp(logits))

                        pos_spcl = pos_sub + pos_cent_logit
                        neg_spcl = neg_sup + negative_cent_logits
                        
                        norm_prob = torch.multiply(torch.div(1, positive_num + 1), torch.div(pos_spcl, neg_spcl))
                        norm_prob = torch.log(norm_prob)

                        total_prob = torch.multiply(-1, norm_prob)

                        loss += total_prob
        

        norm_loss = torch.div(loss, sample_size)

        return norm_loss


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
