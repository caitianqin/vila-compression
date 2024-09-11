from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import json
import argparse
import os



# 定义剪枝函数
def prune_attention_layer(multihead_attn_layer, pruned_heads):
    """
    Function to prune specific heads from the multi-head attention layer.
    """
    embed_dim = multihead_attn_layer.hidden_size
    num_heads = multihead_attn_layer.num_heads
    head_dim = embed_dim // num_heads

    # 剪枝后的新的多头注意力层
    remaining_heads = num_heads - len(pruned_heads)

    # Split qkv weights (query, key, value each has embed_dim x embed_dim)
    q_weights = multihead_attn_layer.q_proj.weight.data
    k_weights = multihead_attn_layer.k_proj.weight.data
    v_weights = multihead_attn_layer.v_proj.weight.data
    o_weights = multihead_attn_layer.o_proj.weight.data
    

    # Prune the heads from the query, key, and value weights
    keep_heads = [i for i in range(num_heads) if i not in pruned_heads]
    
    # 原代码
    # Reshape the q, k, v weights and prune heads
    pruned_q_weights = q_weights.view(num_heads, head_dim, embed_dim)
    pruned_k_weights = k_weights.view(num_heads, head_dim, embed_dim)
    pruned_v_weights = v_weights.view(num_heads, head_dim, embed_dim)
    pruned_o_weights = o_weights.view(num_heads, head_dim, embed_dim)

    for i in range(num_heads):
        if i not in keep_heads:
            pruned_q_weights[i,:,:]=0.0
            pruned_k_weights[i,:,:]=0.0
            pruned_v_weights[i,:,:]=0.0
            pruned_o_weights[i,:,:]=0.0
    

    # Assign pruned weights and biases back to the layer
    multihead_attn_layer.q_proj.weight.data = pruned_q_weights.reshape(embed_dim, embed_dim)
    multihead_attn_layer.k_proj.weight.data = pruned_k_weights.reshape(embed_dim, embed_dim)
    multihead_attn_layer.v_proj.weight.data = pruned_v_weights.reshape(embed_dim, embed_dim)
    multihead_attn_layer.o_proj.weight.data = pruned_o_weights.reshape(embed_dim, embed_dim)
    

    return multihead_attn_layer


def prune_model_layers(model, prune_dict):
    """
    Function to prune attention heads in specific layers of the model.
    :param model: The Hugging Face model to prune
    :param prune_dict: Dictionary specifying the layers and heads to prune
                       For example: {'layer.0.attention': [0, 1, 2], 'layer.1.attention': [0, 2, 3]}
    """
    for name, module in model.named_modules():
        # 检查是否在剪枝字典中
        if name in prune_dict:
            pruned_heads = prune_dict[name]
            print(f"layer {name} has {module.num_heads} heads")
            print(f"Pruning heads {pruned_heads} in layer {name}")
            prune_attention_layer(module, pruned_heads)
            print(f"layer {name} remains {module.num_heads-len(pruned_heads)} heads")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pretrained_model", type=str, default="../../base_model/VILA1.5-3b")
    parser.add_argument("--prune_dict", type=str, default="prune_dict_vila.json")
    parser.add_argument("--output_path", type=str, default="outputs/")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.pretrained_model
    prune_dict = json.loads(open(args.prune_dict).read())
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    prune_model_layers(model, prune_dict)
    
    # 简单测试,输出内容不重要，跑通即可
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_text = "Tell me something about large language models:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Text:")
    print(generated_text)
    
    # 保存剪枝后的模型
    model.save_pretrained(args.output_path)

    
if __name__ == '__main__':
    main()

