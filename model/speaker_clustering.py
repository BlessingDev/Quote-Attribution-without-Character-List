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
        
        self.initial_presentation = nn.Sequential(
            nn.Linear(config.hidden_size, config.decoder_intermediate_size),
            nn.ELU()
        )
        
        self.decoder_hidden = nn.Sequential(
            nn.Linear(config.decoder_intermediate_size, config.decoder_intermediate_size),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(config.decoder_intermediate_size, config.decoder_intermediate_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.decoder_intermediate_size, config.feature_dimension),
            nn.Tanh()
        )
        
        self.feature_freedom = config.feature_freedom
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
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
        
        last_hidden = encoder_output.last_hidden_state
        
        initial_hidden = self.initial_presentation(last_hidden)
        latent_hidden = self.decoder_hidden(initial_hidden)
        norm_hidden = nn.LayerNorm(initial_hidden.shape[1:], device=latent_hidden.device)(latent_hidden)
        norm_hidden = norm_hidden + initial_hidden
        
        latent_features = self.decoder(norm_hidden)
        latent_features = latent_features * self.feature_freedom
        
        return (latent_features), encoder_output

    def init_memories(self):
        self.infini_bert.encoder.init_memories()
    
    def detach_memories_(self):
        self.infini_bert.encoder.detach_memories_()