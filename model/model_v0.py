from transformers import PretrainedConfig
import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


"""
模型v0版本

"""

# model config
class ModelConfig(PretrainedConfig):
    def __init__(
            self, 
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            # PyTorch 2.0+ 内置的 Flash Attention 实现（通过 torch.nn.functional.scaled_dot_product_attention），用于加速注意力计算，和 GQA无关
            flash_attn: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling

        self.flash_attn = flash_attn


# RMSNorm不减均值，只缩放；LayerNorm减均值，平移+缩放。RMSNorm更快，更稳定
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        square = x.pow(2)
        mean = square.mean(-1, keepdim=True)

        return x * torch.rsqrt(mean + self.eps)

    def forward(self, x):
        # 混合精度训练思想，在数值敏感的操作上用高精度，在其他操作上用低精度。这是混合精度训练的精髓，x.float()临时升级精度，例如假设 x 是 float16 类型，x.float() → 临时转为 float32 进行归一化计算，.type_as(x) → 计算完成后转回 float16。float16归一化不稳定
        return self.weight * self._norm(x.float()).type_as(x)


# precompute_freqs_cis在模型初始化时预计算所有位置的旋转角度
def precompute_freqs_cis(dim: int,  # 每个 head 的维度
                         end: int = int(32 * 1024),  # 最多支持的位置长度，模型能处理最长的上下文
                         rope_base: float = 1e6,  # 控制频率跨度（LLaMA 系常用 1e6）
                         ):
    # 计算逆频率(模仿复数的极坐标表示，不同维度对应不同频率的旋转，形成多尺度位置编码)。依然要保持低维对应高频（变化快）而高维对应低频（变化慢）的思想
    # torch.arange(0, dim, 2)：枚举旋转平面，每 2 个维度 = 一个旋转对。每个二维平面只需要一个频率例如(x0, x1)共用一个角度θ，(x2, x3)共用另一个θ;attn_factor防止长上下文下 attention logits 变小、变平
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim)), 1.0

    # 外积计算所有位置的角度
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # 复制两次，前后配对而不是相邻配对，更方便更工程化
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    # 最终输出形状freqs_cos.shape = [end, dim]，freqs_sin.shape = [end, dim]
    return freqs_cos, freqs_sin


# apply_rotary_pos_emb每次前向传播时用预计算的角度旋转Q和K
# 对 Q / K 做“旋转位置编码”
def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim = 1)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim = 1))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim = 1)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim = 1))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:  # 退化成原始KV
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


# 将两层注意力机制写为一个类
class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # KV Head 数量（GQA 的起点），如果没显式指定 num_key_value_head会退化成 普通 Multi-Head Attention(Q 多头，K/V 少头为了省显存 + 省算力)
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads  
        # 必须整除的原因是多个 Q head 要共享同一组 K/V
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # 每个 KV head 要被复制给多少个 Q head
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        # Q / K / V 投影（注意维度不同，Q的总维度更大，KV维度相同）（也是为了“推理效率 + 显存友好”）（GQA 的核心就在这里）
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 把多头拼回 hidden size
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        # Dropout & Flash Attention 开关
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
              
    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):

        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        # Decoder-only 必须用 RoPE原因是1、天然支持自回归 2、支持 KV cache 3、支持长度外推（YaRN）
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])  # 注意切片

        # kv_cache实现（当前 step 只算 新 token 的 K/V，之前的 K/V 直接拼过来）（推理复杂度从 O(T²) → O(T)）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 存储新的 cache，是否把新的 KV 返回，供下一步使用  
        past_kv = (xk, xv) if use_cache else None

        # 调整维度 + GQA 的关键一步
        # repeat_kv的原因是Q head 多，K/V head 少，Attention 计算要求 head 数一致（逻辑是一个 KV head → 服务多个 Q head）
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # seq_len > 1（不是单token生成），attention_mask 为空或全1（没有特殊padding）
        # # Flash Attention 分支（速度比普通注意力快）（Flash Attention 内部自动完成scaling、causal mask、softmax、matmul），手动计算显存占用为O(N²) 存储完整注意力矩阵，Flash Attention为O(N) 分块计算，不存储完整矩阵，且自动支持KV Cache
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        # 普通 Attention 分支（速度慢）
        else:
            # 计算 QKᵀ
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Causal Mask（Decoder-only 的灵魂，token i 看不到 token > i）
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0) 

            # Padding Mask
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 这里用-1e9，因果掩码用-inf，区别在于Causal Mask是硬约束表明未来必须绝对不能看，Padding Mask是软约束，主要为了避免噪声，提升稳定性
                # padding 位置：主要是"没有意义"，不是"绝对错误"，它们不包含有效信息，但不一定导致严重的语义错误
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9  
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
       
        output = self.resid_dropout(self.o_proj(output))
       
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.intermediate_size is None:  
            intermediate_size = int(config.hidden_size * 8 / 3)  # 确保 SwiGLU 架构的参数量与传统 FFN 相同
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  # 对齐到硬件友好的数值，GPU友好，提升训练性能
            # 偷懒可以这样写，刚好对齐时会增加一些开销config.intermediate_size = 64 * (intermediate_size // 64 + 1) 
        
        # SwiGLU（Swish-Gated Linear Unit）激活函数，引入了Swish门控，让模型可以动态地"开/关"特征通道，更灵活地控制信息流，在相同参数量下获得更好性能
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)    
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # ACT2FN 是一个激活函数映射字典，作用是将字符串标识符转换为对应的 PyTorch 激活函数

    def forward(self, x):
        # gate_proj + act_fn 产生一个"门控值"，up_proj 产生"信息内容"，相乘让门控值控制哪些信息通过、哪些被抑制
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class ModelBlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id

        self.layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states  # (B, T, H)，残差连接先保存原始数据
        hidden_states, present_key_value = self.self_attn(
                                                            self.layernorm(hidden_states),  # 使用Pre-LN
                                                            position_embeddings,  
                                                            past_key_value, 
                                                            use_cache, 
                                                            attention_mask
                                                        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.layernorm(hidden_states))
        return hidden_states, present_key_value


# 将各个层拼接起来
class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([ModelBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, 
                                                    rope_base=config.rope_theta,
                                                    )
        
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents


class ModelForCausalLM(PreTrainedModel, GenerationMixin):  # 继承 GenerationMixin：获得文本生成的采样、贪婪解码等方法
    config_class = ModelConfig

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        super().__init__(self.config)
        self.model = Model(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 语言建模头，将特征映射为词汇表概率分布
        self.model.embed_tokens.weight = self.lm_head.weight  # 词嵌入层和 lm_head 共享权重，语义一致性：token 到嵌入（用行）和嵌入到 logits（用列）映射相同，节省参数：减少约 vocab_size × hidden_size 个参数

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                # 计算所有 token 的 logits，只计算最后 N 个 token 的 logits（训练时常用 N=1），logits_to_keep = tensor([1,3,5])：只计算特定位置的 logits
                # 目的：避免对长序列的每个 token 都计算 logits，大幅节省显存和计算
                logits_to_keep: Union[int, torch.Tensor] = 0, 
                **args):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        return output

        # output =
        # {
        #   logits: Tensor,  [B, K, V]，K为保留的 token 数，V为vocab size，表示每个batch的最后一个token的预测分布
        #   past_key_values: List[Tuple],
        #   hidden_states: Tensor
        # }                                         