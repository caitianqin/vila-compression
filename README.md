1. 下载需要剪枝的基础模型
2. 将需要删减的layer名和对应的head index保存至json文件，参考prune_dict_vila.json
3. 执行 实验.ipynb
4. 跳到vila模型删减head标题下，执行!python3 example_vila_mask.py --pretrained_model '../../base_model/VILA1.5-3b/llm' --prune_dict 'prune_dict_vila.json' --output_path 'outputs/vila-llm-pruned-mask'，得到删减后的模型
 
