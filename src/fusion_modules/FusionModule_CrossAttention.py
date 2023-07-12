import torch

def init_xavier(module):
  if type(module) == torch.nn.Linear:
    torch.nn.init.xavier_uniform_(module.weight)

class FusionModule_CrossAttention(torch.nn.Module):

    def __init__(self, feature_map_channels, nhead=8, dim_feedforward=256,
                kv_dim=1024, dropout_rate=0.1):
        super().__init__()

        attention_embed_dim = feature_map_channels
        self.attention = torch.nn.MultiheadAttention(
                              embed_dim=attention_embed_dim, batch_first=True,
                              num_heads=nhead, dropout=dropout_rate,
                              kdim=kv_dim, vdim=kv_dim  # specify the embedding size of the query embedding
                          )
        # Feed forward block
        self.linear1 = torch.nn.Linear(attention_embed_dim, dim_feedforward)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.linear2 = torch.nn.Linear(dim_feedforward, feature_map_channels)

        self.norm1 = torch.nn.LayerNorm(attention_embed_dim)
        self.norm2 = torch.nn.LayerNorm(attention_embed_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Linear layers initialization
        self.linear1.apply(init_xavier)
        self.linear2.apply(init_xavier)

    def forward(self, feature_map, text_embedding):
        text_embedding = text_embedding.unsqueeze(1)    # [N, 1, 1024]

        # Flatten the feature map into a sequence
        permuted_fm = feature_map.flatten(start_dim=2).permute(0,2,1)           # [N, 49, 2048]
        permuted_fm = permuted_fm / permuted_fm.norm(dim=(-1), keepdim=True)

        # Cross attention
        attn_out, _ = self.attention(query=permuted_fm, key=text_embedding, value=text_embedding, need_weights=False)
        attn_out = self.dropout(attn_out)
        attn_out = self.norm1(permuted_fm + attn_out) # Add & norm

        # Feed forward block
        x = torch.nn.functional.relu(self.linear1(attn_out))
        x = self.dropout1(x)
        x = self.linear2(x)

        x = self.norm2(attn_out + x)  # Add & norm

        # Unflatten the feature map, i.e from sequence to feature map
        x = x.permute(0,2,1).unflatten(dim=2, sizes=(7,7))        # [N, 2048, 7, 7]

        return x