from transformers import AutoModel

import torch



class Encoder(torch.nn.Module):
    def __init__(self, model_name, pre_trained, state_path):
        super(Encoder, self).__init__()
        self.enocder = None
        self.state_path = state_path
        
        if pre_trained: 
            self.encoder = AutoModel.from_pretrained(model_name)

            self.encoder.load_state_dict(self.get_state_without_module())
            self.encoder.eval()

        else:
            self.encoder = AutoModel.from_pretrained(model_name)
            self.encoder.eval()


    def forward(
        self, x: torch.LongTensor, attn_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """passage 또는 query를 bert로 encoding합니다."""
        
        return self.encoder(input_ids = x, attention_mask = attn_mask)
        
    def get_state_without_module(self):
        checkpoint = torch.load(self.state_path, map_location='cpu')
        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith("module."):
                new_key = key[len("module."):]
            else:
                new_key = key
            new_state_dict[new_key] = value
    
        return new_state_dict
