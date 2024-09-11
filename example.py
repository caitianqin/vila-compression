import torch
import torch.nn as nn

class PrunedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, pruned_heads):
        super(PrunedMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Pruned heads will be removed from the num_heads
        self.remaining_heads = num_heads - len(pruned_heads)
        self.pruned_heads = pruned_heads
        
        # Define new layers with pruned heads
        self.q_proj = nn.Linear(embed_dim, self.remaining_heads * self.head_dim)
        self.k_proj = nn.Linear(embed_dim, self.remaining_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.remaining_heads * self.head_dim)
        self.out_proj = nn.Linear(self.remaining_heads * self.head_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_length, _ = query.size()
        
        # Apply linear projections
        q = self.q_proj(query).view(batch_size, seq_length, self.remaining_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_length, self.remaining_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_length, self.remaining_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_weights += attn_mask
        
        attn_weights = self.softmax(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).reshape(batch_size, seq_length, -1)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

def prune_attention_layer(multihead_attn_layer, pruned_heads):
    """
    Function to prune specific heads from the multi-head attention layer.
    """
    embed_dim = multihead_attn_layer.embed_dim
    num_heads = multihead_attn_layer.num_heads
    head_dim = embed_dim // num_heads

    # New multi-head attention layer with pruned heads
    pruned_attn = PrunedMultiheadAttention(embed_dim, num_heads, pruned_heads)
    
    # Extract original weights and biases
    qkv_weights = multihead_attn_layer.in_proj_weight.data
    qkv_biases = multihead_attn_layer.in_proj_bias.data
    
    # Split qkv weights (query, key, value each has embed_dim x embed_dim)
    q_weights = qkv_weights[:embed_dim, :]
    k_weights = qkv_weights[embed_dim:2*embed_dim, :]
    v_weights = qkv_weights[2*embed_dim:, :]

    # Split qkv biases
    q_bias = qkv_biases[:embed_dim]
    k_bias = qkv_biases[embed_dim:2*embed_dim]
    v_bias = qkv_biases[2*embed_dim:]

    # Remove pruned heads from the weights
    keep_heads = [i for i in range(num_heads) if i not in pruned_heads]

    # Prune query, key, and value weights
    pruned_q_weights = q_weights.view(num_heads, head_dim, embed_dim)[keep_heads].reshape(len(keep_heads) * head_dim, embed_dim)
    pruned_k_weights = k_weights.view(num_heads, head_dim, embed_dim)[keep_heads].reshape(len(keep_heads) * head_dim, embed_dim)
    pruned_v_weights = v_weights.view(num_heads, head_dim, embed_dim)[keep_heads].reshape(len(keep_heads) * head_dim, embed_dim)
    
    # Prune query, key, and value biases
    pruned_q_bias = q_bias.view(num_heads, head_dim)[keep_heads].reshape(len(keep_heads) * head_dim)
    pruned_k_bias = k_bias.view(num_heads, head_dim)[keep_heads].reshape(len(keep_heads) * head_dim)
    pruned_v_bias = v_bias.view(num_heads, head_dim)[keep_heads].reshape(len(keep_heads) * head_dim)

    # Assign pruned weights to the new attention layer
    pruned_attn.q_proj.weight.data = pruned_q_weights
    pruned_attn.q_proj.bias.data = pruned_q_bias
    pruned_attn.k_proj.weight.data = pruned_k_weights
    pruned_attn.k_proj.bias.data = pruned_k_bias
    pruned_attn.v_proj.weight.data = pruned_v_weights
    pruned_attn.v_proj.bias.data = pruned_v_bias
    
    # Copy over the output projection weights and biases
    pruned_attn.out_proj.weight.data = multihead_attn_layer.out_proj.weight.data[:, keep_heads].reshape(embed_dim, -1)
    pruned_attn.out_proj.bias.data = multihead_attn_layer.out_proj.bias.data
    
    return pruned_attn



# 假设我们有一个多头注意力层
embed_dim = 128
num_heads = 8
pruned_heads = [0,1,3]  # 剪掉第 1, 2, 3, 4 个head

# 创建一个原始的多头注意力层
original_attention_layer = nn.MultiheadAttention(embed_dim, num_heads)
print(f"original_attention_layer:{original_attention_layer}")
torch.save(original_attention_layer.state_dict(), "original_attention_layer.pth")

# 执行剪枝操作
pruned_attention_layer = prune_attention_layer(original_attention_layer, pruned_heads)

# 保存模型
torch.save(pruned_attention_layer.state_dict(), "pruned_attention_layer.pth")

print("剪枝后的多头注意力层已保存！")