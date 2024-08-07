import torch
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions
)
from typing import List, Optional, Tuple, Union

from model.custom_bert_model import InfiniBertModel
from model.custom_roberta_model import InfiniRobertaModel

class SpeakerClusterBERT(nn.Module):
    def __init__(self, config):
        super(SpeakerClusterBERT, self).__init__()
        self.infini_bert = InfiniBertModel.from_pretrained("bert-base-cased", config=config, add_pooling_layer=False)
        self.config = self.infini_bert.config
        
        self.sequencial_decoder = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.decoder_intermediate_size),
            nn.ELU(),
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(config.decoder_intermediate_size, config.feature_dimension)
        )
        
        self.feature_freedom = config.feature_freedom
        
        self.rnn_last_hidden = None
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        cls_idx: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        
        encoder_output = self.infini_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        # bert_last_hidden: (b n d_model)
        bert_last_hidden = encoder_output.last_hidden_state
        
        if cls_idx is None:
            cls_idx = [0]
        cls_hidden = bert_last_hidden[0][cls_idx].unsqueeze(0)
        
        # b = 1
        # rnn_out: (b t rnn_hidden)
        # rnn_h: (rnn_layer b rnn_hidden)
        if self.rnn_last_hidden is not None:
            rnn_out, rnn_h = self.sequencial_decoder(cls_hidden, self.rnn_last_hidden)
        else:
            rnn_out, rnn_h = self.sequencial_decoder(cls_hidden)
        
        self.rnn_last_hidden = rnn_h
        seq_feature = rnn_out[-1, :, :]
        
        latent_features = self.decoder(seq_feature)
        #latent_features = latent_features * self.feature_freedom
        
        return (latent_features), encoder_output

    def init_memories(self):
        self.rnn_last_hidden = None
        self.infini_bert.encoder.init_memories()
    
    def detach_memories_(self):
        self.rnn_last_hidden = self.rnn_last_hidden.detach_()
        self.infini_bert.encoder.detach_memories_()


class SpeakerClusterRoBERTa(nn.Module):
    def __init__(self, config, model_name="roberta-base"):
        super(SpeakerClusterRoBERTa, self).__init__()
        self.infini_bert = InfiniRobertaModel.from_pretrained(model_name, config=config, add_pooling_layer=False)
        self.config = self.infini_bert.config
        
        self.sequencial_decoder = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.decoder_intermediate_size),
            nn.ELU(),
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(config.decoder_intermediate_size, config.feature_dimension)
        )
        
        # narrative와 비 narrative를 부분할 선형 층
        self.narrative_classifier = nn.Linear(config.hidden_size, 1)
        
        self.feature_freedom = config.feature_freedom
        
        self.rnn_last_hidden = None
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        cls_idx: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        
        encoder_output = self.infini_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        # bert_last_hidden: (b n d_model)
        bert_last_hidden = encoder_output.last_hidden_state
        
        if cls_idx is None:
            cls_idx = [0]
        cls_hidden = bert_last_hidden[0][cls_idx].unsqueeze(0)
        
        # b = 1
        # rnn_out: (b t rnn_hidden)
        # rnn_h: (rnn_layer b rnn_hidden)
        if self.rnn_last_hidden is not None:
            rnn_out, rnn_h = self.sequencial_decoder(cls_hidden, self.rnn_last_hidden)
        else:
            rnn_out, rnn_h = self.sequencial_decoder(cls_hidden)
        
        self.rnn_last_hidden = rnn_h
        seq_feature = rnn_out[-1, :, :]
        
        latent_features = self.decoder(seq_feature)
        narrative_desc = self.narrative_classifier(seq_feature)
        #latent_features = latent_features * self.feature_freedom
        
        return latent_features, narrative_desc.unsqueeze(0)

    def init_memories(self):
        self.rnn_last_hidden = None
        self.infini_bert.encoder.init_memories()
    
    def detach_memories_(self):
        self.rnn_last_hidden = self.rnn_last_hidden.detach_()
        self.infini_bert.encoder.detach_memories_()
    
class SpeakerPreClassificationRoBERTa(nn.Module):
    def __init__(self, config, model_name="roberta-base"):
        super(SpeakerPreClassificationRoBERTa, self).__init__()
        self.infini_bert = InfiniRobertaModel.from_pretrained(model_name, config=config, add_pooling_layer=False)
        self.config = self.infini_bert.config
        
        self.sequencial_decoder = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, batch_first=True)
        
        
        decoder_lin = nn.Linear(config.hidden_size, config.feature_dimension)
        decoder_lin.requires_grad_(False)
        self.decoder = nn.Sequential(
            decoder_lin,
            nn.ELU()
        )
        
        # 화자 구분하는 선형 층
        self.classifier_dict = nn.ModuleDict()
        
        self.current_classifier = None
        
        self.feature_freedom = config.feature_freedom
        
        self.rnn_last_hidden = None
        
        self.current_key = ""
    
    def add_classifier(self, key, label_num, device):
        if key not in self.classifier_dict.keys():
            self.classifier_dict[key] = nn.Linear(self.config.feature_dimension, label_num)
            self.classifier_dict[key].to(device)
    
    def set_key(self, key):
        self.current_key = key
        self.current_classifier = self.classifier_dict.get_submodule(key)
        
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        cls_idx: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        assert self.current_classifier is not None, "classifier is not set"
        
        encoder_output = self.infini_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        # bert_last_hidden: (b n d_model)
        bert_last_hidden = encoder_output.last_hidden_state
        
        if cls_idx is None:
            cls_idx = [0]
        cls_hidden = bert_last_hidden[0][cls_idx].unsqueeze(0)
        
        # b = 1
        # rnn_out: (b t rnn_hidden)
        # rnn_h: (rnn_layer b rnn_hidden)
        if self.rnn_last_hidden is not None:
            rnn_out, rnn_h = self.sequencial_decoder(cls_hidden, self.rnn_last_hidden)
        else:
            rnn_out, rnn_h = self.sequencial_decoder(cls_hidden)
        
        self.rnn_last_hidden = rnn_h
        seq_feature = rnn_out[-1, :, :]
        
        latent_features = self.decoder(seq_feature)
        
        classification_result = self.current_classifier(latent_features)
        
        return latent_features, classification_result

    def inference(self, 
        input_ids: Optional[torch.Tensor] = None,
        cls_idx: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> torch.Tensor:
        
        encoder_output = self.infini_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        # bert_last_hidden: (b n d_model)
        bert_last_hidden = encoder_output.last_hidden_state
        
        if cls_idx is None:
            cls_idx = [0]
        cls_hidden = bert_last_hidden[0][cls_idx].unsqueeze(0)
        
        # b = 1
        # rnn_out: (b t rnn_hidden)
        # rnn_h: (rnn_layer b rnn_hidden)
        if self.rnn_last_hidden is not None:
            rnn_out, rnn_h = self.sequencial_decoder(cls_hidden, self.rnn_last_hidden)
        else:
            rnn_out, rnn_h = self.sequencial_decoder(cls_hidden)
        
        self.rnn_last_hidden = rnn_h
        seq_feature = rnn_out[-1, :, :]
        
        latent_features = self.decoder(seq_feature)
        
        return (latent_features, None)
        

    def init_memories(self):
        self.rnn_last_hidden = None
        self.infini_bert.encoder.init_memories()
    
    def detach_memories_(self):
        self.rnn_last_hidden = self.rnn_last_hidden.detach_()
        self.infini_bert.encoder.detach_memories_()