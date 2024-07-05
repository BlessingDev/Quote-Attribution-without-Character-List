import torch
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions
)
from typing import List, Optional, Tuple, Union

from model.custom_bert_model import InfiniBertModel

class SpeakerClusterModel(nn.Module):
    def __init__(self, config):
        super(SpeakerClusterModel, self).__init__()
        self.infini_bert = InfiniBertModel.from_pretrained("bert-base-uncased", config=config, add_pooling_layer=False)
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