import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
from datasets import load_dataset  # Hugging Face 的 datasets 库

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 每个数据类都继承Pytorch dataset类，方便训练

# 用于语言模型的无监督预训练（类似 GPT 的自回归预训练），核心任务是 “预测下一个 token”。其核心职责是将原始文本转换为滑动窗口训练样本，模型在其中学习预测下一个 token
# 默认最大长度为512，损失掩码策略为仅掩码填充 token，输出格式(X, Y, loss_mask)
# 数据集加载包含 {"text": "..." 对象的 JSONL 文件，将每个样本分词到固定的 max_length，并通过移位序列构建训练对：
# 输入 X 包含除最后一个外的所有 token，而目标 Y 包含除第一个外的所有 token。损失掩码标记有效（非填充）token，确保模型仅从实际文本数据中学习。
# 这里的关键设计模式是序列移位操作：X = input_ids[:-1] 和 Y = input_ids[1:] 结合 loss_mask[1:]。这种对齐确保在每个位置 i，模型预测 token i+1，并且仅在有效（非填充）位置接收损失信号。
"""
input_ids = [1, 256, 512, 1024, 0, 0]  # 0 = 填充 token
X = [1, 256, 512, 1024]              # 预测 256, 512, 1024, 0
Y = [256, 512, 1024, 0]              # 目标：每个 X token 之后的内容
loss_mask = [1, 1, 1, 0]             # 不计算填充的损失
"""
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # load_dataset返回的是
        """
        DatasetDict({
            'train': Dataset(...)
        })
        """
        # Dataset类似于(features：字段名（列）, num_rows：样本条数)
        """
        print出来形如
        Dataset({
            features: ['text'],
            num_rows: 3
        })
        """
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        # 加了return_tensors='pt'返回的格式形如（不加不是tensor格式）
        """
        {
            'input_ids': tensor([[101, 2769, ...]]),
            'attention_mask': tensor([[1, 1, ...]])
        }
        """
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',  # 不够长就补到 max_length
            truncation=True,  # 超过 max_length 的 token 直接截断
            return_tensors='pt'  # 直接返回 PyTorch Tensor
        )
        input_ids = encoding.input_ids.squeeze()  # 因为 shape 是(1, max_length)（理解为变成这样是因为转了tensor）， Dataset 里只处理单条样本，所以要变成(max_length,)
        # 为了清晰区分结构更清晰，这里自己造loss_mask而不用自带的attention_mask，attention_mask 最好是在将来给模型 forward 用，这里loss_mask 给 训练时计算loss 用
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # 生成一个“哪些 token 要参与 loss 计算”的掩码（padding部分不参数loss计算）

        # 自回归训练精髓，用第 t 个 token，去预测第 t+1 个 token（把同一条 token 序列，错开一位，变成 (输入, 目标)）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 因为loss是对Y算的
        return X, Y, loss_mask


# 用于语言模型的有监督指令微调（Supervised Fine-Tuning, SFT），核心任务是 “根据指令和对话历史，生成符合要求的回复”（如聊天机器人、问答系统），专门训练模型学会根据指令和对话历史生成合适的回复
# SFTDataset 引入了复杂的对话格式化和选择性损失掩码，用于监督微调。与预训练中所有 token 都对损失有贡献不同，SFT 要求模型从助手回复中学习，同时将用户指令视为上下文
# 默认最大长度为1024，损失掩码策略为掩码非助手回复，输出格式为(X, Y, loss_mask)
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        # 将文本格式的特殊标记转换为token ID列表，用于在序列中定位assistant回复的开始和结束位置
        # tokenizer.bos_token为tokenizer 的开始序列token例如<|im_start|>，tokenizer.eos_token：获取结束序列token例如<|im_end|>
        # add_special_tokens=False很关键，告诉tokenizer"不要再额外添加bos/eos/pad等特殊token"
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    # 将对话数据转换成模型能理解的统一文本格式
    def _create_chat_prompt(self, cs):  # cs 是传入的对话历史（conversations）
        messages = cs.copy()
        # 检查第一条是不是system消息，system 消息里有没有functions字段，若都满足就提取工具定义tools
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # 是否在末尾添加"assistant 开始生成"的提示
            tools=tools
        )

    # 用大量问答数据或聊天记录数据来微调模型（可以用来训练客服机器人、构建领域问答系统、让模型学会特定的对话风格）
    # 整个类的重点与关键创新，通过搜索 BOS（<s>assistant）和 EOS（</s>）token 来识别助手回复边界，然后仅对这些区域应用损失掩码
    # 只对assistant的回复部分计算损失，用户指令部分不参与训练，这样模型学会的是"如何回答问题"，而不是"如何复述用户的问题"
    def _generate_loss_mask(self, input_ids):
        # 初始化掩码
        loss_mask = [0] * len(input_ids)
        i = 0
        # 这个循环语句很关键
        while i < len(input_ids):
            # 检测assistant回复的开始，从i位置开始，提取与bos_id长度相同的一段子列表，不停滑动窗口用于比较是否匹配
            # 切片操作是安全的，不怕索引越界
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    # 检测assistant回复的结束
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                    # 这里的“end + len(self.eos_id) + 1”特别是“+1”不好理解。特别注意的是回复的的第一个token不应该算loss，第一个token由assistant预测出来，从第二个token开始才算“模型自己生成的”
                    # 可结合__getitem__里的掩码偏移与GPT编写的脚本SFTdataset_mask_understand.py理解  
                    # 手算一遍就清楚了，range要减1，掩码偏移要减1，刚好到<|im_end|>（假如结束符是<|im_end|>的话）
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 这一步是给外层while循环的，及时更新i，保证下一次不继续命中现在同一个<|im_start|>assistant，继续往后扫（手算很清晰）（这一步是工程写法效率更高，但其实写i += 1也可以）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)      
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # 处理换行等不可见字符
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        # # ================================
        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        self.data = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        ), answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt, answer = self._create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    pass
