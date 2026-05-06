import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # nn.Embedding 初始化的向量 数值很小（大多在 -0.1 ~ 0.1）,而 位置编码（Positional Encoding） 的数值大概在 -1 ~ 1 之间
    # embedding 会被“位置编码”淹没掉，模型前期训练会很困难,乘以 sqrt(d_model) 是为了统一数值尺度
    # 但是为什么是根号d_model,要涉及到多头注意力里的细节。注意力层把数值除以了 sqrt(d_k),Embedding 这里乘上 sqrt(d_model)，保持不同模块之间特征值的统计一致性
    

# 位置编码下方会根据位置添加一个正弦波。每个维度的正弦波频率和偏移量都不同
# 对输入序列 x（形状通常是 [batch, seq_len, d_model]）加上固定的、与位置相关的向量 pe(pos)，再做 dropout，返回加了位置编码的表示。这样 Transformer 即便没有 RNN 结构，也能感知词序信息。论文使用（sin/cos）函数是因为：无参数、能表示任意相对位置（形式上方便模型通过线性变换推导出相对位置信息）且可泛化到更长序列。
# seq_len代表当前batch 的最大句子长度，输入张量在每个batch内是统一长度的（经过 padding），但不同 batch 的最大长度可能不一样，seq_len也就可能不一样
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe先创建一个 (max_len, d_model) 的零张量，用来填入位置编码
        pe = torch.zeros(max_len, d_model)
        # position是一个列向量 [[0],[1],[2],...,[max_len-1]]，用于计算不同位置的编码，unsqueeze(1)在索引 1 的位置插入一个长度为 1 的维度（即把一维变成二维的“列向量”）,形状由(max_len,)变为 (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        # div_term构造一系列不同频率的因子（只针对偶数维度），对应论文中公式里的 1 / 10000^{2i/d_model} 的对数形式。torch.arange(0, d_model, 2) 取偶数索引（0,2,4,...），exp(...) 给出频率缩放因子。
        # div_term 生成了一组随维度指数衰减的因子，用于把位置 pos 缩放到不同频率上，采用这个数值这么计算的原因是由论文里的公式决定的
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))  
        # 偶数维用 sin(pos * freq)，奇数维用 cos(pos * freq)。结果：每个位置 pos 对应一个长度为 d_model 的向量，维度间使用不同频率的 sin/cos。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe.unsqueeze(0) → 变成 (1, max_len, d_model)，方便后面直接和 x 广播相加,x 是 [batch, seq_len, d_model]。
        pe = pe.unsqueeze(0)
        # register_buffer('pe', pe)：把 pe 注册为 module 的 buffer（随 state_dict 保存/加载，但不是可学习参数，不会在优化器中更新），位置编码是固定的常量，不需要求梯度。
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # self.pe[:, :x.size(1)]：取出前 seq_len 个位置编码，形状为 (1, seq_len, d_model)，与 x 的 (batch, seq_len, d_model) 可以广播相加。
        # Variable(..., requires_grad=False)：这是旧版 PyTorch 风格（早期把 Variable 和张量区分开）。register_buffer 已经保证 pe 不会被训练，所以现代代码中不需要再用 Variable，直接写 x = x + self.pe[:, :x.size(1)] 即可。
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    

# attention返回两个东西context（加权后的 value，记作 x，这是 attention 的主要输出，会用于后续拼接多头并线性投影）与p_attn（注意力权重矩阵，表示每个 Query 在各个 Key 上的注意力分布）
def attention(query, key, value, mask, dropout=None):
    d_k = query.size(-1)
    # 计算打分，表示每个 Query（t）对每个 Key（s）的相似度
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)  # 用 √d_k 来“缩放”点积分数，保持数值稳定(为什么是除以这个数，是数学推导出来的),scores.shape = [batch, h, seq_len_q, seq_len_k]，每个 Query 去“打分”所有 Key
    # 如果有 mask，把 mask==0 的位置设为 -1e9（近似 -inf），使这些位置在 softmax 后权重≈0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 每个 Query 的权重分布（和为1），把“相似度分数”变成“概率权重”，每个 Query 对所有 Key 的打分转换为 非负且和为 1 的概率分布（注意力权重）
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # 计算加权和context,context = torch.matmul(p_attn, value) → context.shape = [batch, heads, seq_len, d_k]。p_attn形状为[batch, heads, seq_len_q, seq_len_k]
# context为得到每个Query的“上下文向量,相当于为Query聚合来自不同位置的信息,这样每个词都能看到其他所有词


# 多头注意力机制，把模型的注意力“分成多个平行的视角”去捕捉不同的特征，多头的意思是对token内部做拆分，将一个token拆分成几头，而不是拆分token
class MultiHeadedAttention(nn.Module):
    # h: 头的数量（heads）,d_model: 整个模型的维度，比如 512 或 768,每个头的维度：d_k = d_model // h
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    # query, key, value, 所有输入形状均为 [batch_size, seq_len, d_model]
    def forward(self, query, key, value, mask):
        # padding mask（用于屏蔽 pad）,(batch, src_len) 或 (batch, tgt_len);subsequent mask（防看未来）,(1, tgt_len, tgt_len) 或 (tgt_len, tgt_len);最终都广播到scores的[batch, heads, seq_len_q, seq_len_k]
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 这句用 3 个线性层分别对 query, key, value 做投影，然后把每个投影reshape 成多头格式并把 head 维挪到前面，最终得到每个张量的形状为：[batch, h, seq_len, d_k]
        # 对每对执行 l(x)，把 [B, L, D] 映射到 [B, L, D]（线性层输出维度仍为 d_model）,.view 用来重塑张量，-1 表示让 PyTorch 自动推断该维度（在这里就是 seq_len）,self.h 和 self.d_k 是多头拆分后两个维度
        # 列表推导会产出一个包含三个张量的列表，然后用解包赋值，query.shape = key.shape = value.shape = [B, h, seq_len, d_k]
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  
             for l, x in zip(self.linears, (query, key, value))]  # zip(...) 会把前三个线性层分别和 query, key, value 配对,(l0, query), (l1, key), (l2, value)
        
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 把多头的输出再拼接回来，形状变回 [batch, seq_len, d_model]
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# 前馈网络主要作用：对每个位置的表示独立地做非线性变换（所以叫 position-wise）；提升模型表达能力，让网络能学习更复杂的特征组合；这里的非线性是通过 ReLU 激活实现的
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 线性层，把输入从 d_model 维度映射到 d_ff 维度，原文中 d_ff 比 d_model 大，通常是 4 倍（例如 BERT: 768 → 3072），增加网络容量，允许捕获更复杂关系
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入 x: [batch_size, seq_len, d_model]
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  # 注意 是dropout后再进行第二次映射


# clones、SublayerConnection、LayerNorm是Encoder、Decoder以及其他一些地方共用的
def clones(module, N):
    "Produce N identical layers."
    # copy.deepcopy(module)保证每一层拥有独立的参数，而不是共享同一个对象，nn.ModuleList([...])让这些层可以注册到 PyTorch 的计算图中
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 子层连接（残差 + 归一化 + dropout）,每一个Encoder层里有两个子层（sub-layer）即多头自注意力层和前馈网络层，而在每个子层的外面，都包了一层这样的结构：output=𝑥+Dropout(𝑆𝑢𝑏𝑙𝑎𝑦𝑒𝑟(LayerNom(𝑥)))，这就是残差连接 + 层归一化 + dropout
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        # 对输入先做归一化（注意，这里跟原论文稍有区别，论文中是 先残差再归一化；这里为了代码简洁，作者反过来了）
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    

# 对输入的最后一维（即特征维度）做归一化，目标是对每个token内部的特征做规范化，使得每个token自己的特征的均值为0，方差为1，之后乘上缩放a再加偏移b，使得每个 token 的 embedding 内部分布稳定
# 减少协方差偏移
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        # features 表示要归一化的特征维度大小（即最后一维的大小）
        # a_2也称weight，b_2也称bias，都是可学习参数，.ones/.zeros初始化为全1/全0，nn.Parameter将其注册为模型参数（会被优化器更新），形状为 (features,)
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps  

    def forward(self, x):
        # x 是一个张量，通常形状为 (batch_size, seq_len, features) 或 (batch_size, features)，最后一维是 features（要规范化的维度）
        mean = x.mean(-1, keepdim=True) # 对每个样本在最后一维（-1）（特征维）上求平均值，keepdim=True 保持计算后该维度仍存在，形状变成 (batch_size, seq_len, 1)（如果输入是三维），这样便于后续广播（broadcasting）做减法
        std = x.std(-1, keepdim=True) # 计算x在最后一维的标准差
        # 广播行为self.a_2 和 self.b_2 的形状是 (features,)，与归一化结果的最后一维匹配，PyTorch 会自动把它们广播到 (batch_size, seq_len, features) 相乘/相加
        # 最终输出和输入 x 同形状，但在每个样本的最后一维上经过标准化并线性变换过
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # layer是一个单层编码器,由 Multi-Head Attention + FeedForward组成
        self.layers = clones(layer, N)
        # self.norm对整个编码器输出做一次最终的层归一化LayerNorm
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        # 输入的x通常是词向量 embedding + 位置编码
        "Pass the input (and mask) through each layer in turn."
        # 依次通过每一层 layer,最终通过一个 LayerNorm 输出
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn #多头自注意力（一般是一个 MultiHeadedAttention 类实例）
        self.feed_forward = feed_forward #前馈网络（PositionwiseFeedForward）
        self.sublayer = clones(SublayerConnection(size, dropout), 2) #为每个子层包上一个 SublayerConnection（所以一个 layer 内有两个残差连接）
        self.size = size #通常为 512 或 768（BERT 用的是 768）

    def forward(self, x, mask):

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward) # 前馈网络输出是基于attention结果的，这个输出也包含前一子层即自注意力层的结果


# Decoder 的任务是：给定 “前面的词” → 预测 “下一个词”
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # Masked Self-Attention（只能看见自己之前的词），和 Encoder 的自注意力几乎一样，不同的是加了 tgt_mask，用来阻止模型看到未来的 token
        # 第一个为自注意机制Self-Attention，第二个交叉注意力机制Encoder-Decoder Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Encoder-Decoder Attention（能看见Encoder输出），Decoder的关键创新点，该层对编码器的输出执行多头注意力机制
        # 这行的作用是：让 decoder 用当前的目标序列表示（作为 Query），去“询问” encoder 的输出（作为 Key 和 Value），从而把源句的信息融合进 decoder 的表示里
        # x一般为 [batch, tgt_len, d_model]，m（memory）：Encoder 的输出（也称 memory），形状一般为 [batch, src_len, d_model]
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    

# 生成一个形状为 (1, size, size) 的布尔矩阵，矩阵中上三角（未来位置）被标为 False（不可见），下三角和对角线为 True（可见），用于屏蔽掉“未来词”，保证在做 masked self-attention 时，位置 i 只能看到 ≤ i 的位置，不能看到未来（> i）的 token
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')   # np.triu(..., k=1) 生成上三角（不含对角线）的 1，其他为 0,每一行对应一个token
    return torch.from_numpy(subsequent_mask) == 0  # == 0 把 0 → True（可见），1 → False（被 mask）,返回的是 torch.BoolTensor（True/False），方便与 attention scores 一起使用


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

class Generator(nn.Module):    
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    

# 这里定义了一个接受超参数并生成完整模型的函数
def make_model(src_vocab, tgt_vocab, N, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy  # 在make_model中用它可以产生若干个相互独立的层（每层有自己的权重），而不是让多个层共享同一组参数
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():  # 遍历模型的所有可学习参数
        if p.dim() > 1:  # 只有维度大于 1 的张量才做初始化，这通常筛掉一维的偏置（例如bias）或标量参数，只对权重张量（矩阵、卷积核、嵌入矩阵等）初始化
            nn.init.xavier_uniform(p)  # 对该权重张量做 Xavier（Glorot）均匀分布初始化（把参数填成符合某个区间的均匀随机数）
    return model


# Batches and Masking
# 作用是创建并包装训练要用的src、tgt以及Encoder和Decoder分别的掩码等
# 训练标准编码器-解码器模型所需的一些工具。首先定义一个批处理对象，其中包含用于训练的源语句和目标语句，以及用于构建掩码的对象
# 在mask中,列（Key 方向）为 0 → 这个 token 不允许被任何人看到,行（Query 方向）为 0 → 这个 token 自己不能看别人（但一般不这么做）
# src=输入序列（给 Encoder 的）,用于编码输入内容,只进入 Encoder,形状一般为 [batch, src_len]。tgt=输出序列（给 Decoder 的）,用于训练时“教”模型如何生成下一词”,只进入 Decoder,形状一般为 [batch, tgt_len]
# Encoder输入src,可以全局看,理解句子整体含义;Decoder自注意力输入trg,不允许看未来词,保证自回归生成;Decoder → Encoder交叉注意力输入trg → src,可以全局看源句,输出应参考输入内容
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        # 生成用于 encoder 自注意力里屏蔽 padding 的 mask     
        self.src_mask = (src != pad).unsqueeze(-2)  #  # pad 是 padding 的 id（例如 0）,src != pad 会生成一个布尔张量，形状仍然是 (batch, src_len)
        if tgt is not None:
            # 训练用的典型做法是把目标序列错开一位,decoder 的输入是目标序列的左移版本（不包含最后一个 token），decoder 的训练目标是目标序列的右移版本（不包含第一个 token）
            # self.trg 与 self.trg_y 是一对 parallel sequences：训练时模型用 self.trg 预测 self.trg_y（每次向右一步）。trg是Decoder输入，trg_y是Decoder目标输出
            self.trg = tgt[:, :-1]
            self.trg_y = tgt[:, 1:] 
            # 调用静态方法生成decoder的subsequent mask，这个mask同时要屏蔽padding（trg 中 <pad> 的位置）以及future positions（防止 decoder 在训练时看到“未来的词”）
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 统计这个 batch 中有效目标 token 的总数（不包括 pad），用于在计算平均 loss 时除以有效 token 数
            self.ntokens = (self.trg_y != pad).data.sum()  # 记住为什么要用self.trg_y而不用self.trg,模型要学会什么时候让句子结束
    
    @staticmethod
    def make_std_mask(trg, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (trg != pad).unsqueeze(-2)                        
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(trg.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# 模型训练函数
# 接下来，我们创建一个通用的训练和评分函数来跟踪损失。我们传入一个通用的损失计算函数，该函数也负责处理参数更新
# Training Loop训练循环。输入多个 batch（data_iter）,对每个 batch 做一次 forward + loss + backward + 更新参数,同时记录损失和训练效率（吞吐量 Tokens/sec）
def run_epoch(data_iter, model, loss_compute):  # data_iters是一个能不断提供 Batch（批数据）的迭代器,Batch对象
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0  # tokens：用于每隔一段时间动态计算吞吐率（Tokens per Sec）
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,  # batch.src → Encoder 输入,batch.trg → Decoder 输入（错位后的序列，用于教模型预测下一词）
                            batch.src_mask, batch.trg_mask)  # model.forward 输出：out: [batch, tgt_len, vocab_size],表示模型对每个位置输出的词预测分布
        loss = loss_compute(out, batch.trg_y, batch.ntokens)  # loss_compute 内部loss = criterion(out, trg_y)
                                                                                #loss.backward()
                                                                                #optimizer.step()
                                                                                #optimizer.zero_grad()
        total_loss += loss
        total_tokens += batch.ntokens  # batch.ntokens 是本 batch 中 需要预测的 token 数（不包括 pad）
        tokens += batch.ntokens
        if i % 50 == 1:  # 每 50 个 batch 打印一次训练速度
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))  # loss / batch.ntokens：每个 token 的平均损失（更稳定）;tokens / elapsed是训练速度, 越大说明训练越快
            start = time.time()
            tokens = 0
    return total_loss / total_tokens  # 返回整个 epoch 的平均损失


# 训练数据和批处理，实现非固定 batch_size分批,因为样本长度有时差异较大
# 我们使用包含约 450 万个句子对的标准 WMT 2014 英德数据集进行训练
# 句子采用字节对编码，其源语言和目标语言共享词汇表约 37000 个词元。对于英法数据集，我们使用了规模更大的 WMT 2014 英法数据集，该数据集包含 3600 万个句子，并将词元拆分为 32000 个词段的词汇表
# 句子对按大致序列长度进行分组。每个训练批次包含一组句子对，其中大约包含 25000 个源词元和 25000 个目标词元
# 我们将使用 torchtext 进行批处理。下文将详细讨论这一点。这里，我们在 torchtext 函数中创建批次，以确保填充到最大批次大小后的批次大小不超过阈值（如果我们有 8 个 GPU，则为 25000）
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)  # 这里用了 + 2：通常是因为目标序列在送入 decoder 时会加上 <sos>（或 <bos>）和 <eos>
    # 实际 decoder input/label 的长度会比原始 new.trg 长 2（或作者为了保守估计而加 2）。所以这里把目标长度预留了开始/结束两个位置
    src_elements = count * max_src_in_batch  # 如果把当前 batch 中 count 个样本都 padding 到 max_src_in_batch，那么源侧总 token 数
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)  # 返回源侧和目标侧两种估算中更大的那个,作为“这个 candidate batch 的当前代价（token 数）”
                                            # 数据加载器会根据这个返回值来决定是否再加入新样本（例如保持返回值 <= 某个 token 上限），从而实现按 token 数动态定 batch


# Transformer论文里的学习率调度器（NoamLR）
# 设计原因是训练初期直接用很大的lr会破坏参数（尤其是 Transformer 里多层残差与 LayerNorm 的配合很敏感），warmup 阶段让模型“慢慢热身”到一个合适的学习率；之后按1/根号step衰减可以保证收敛稳定
# 我们在一台配备8个NVIDIA P100 GPU的机器上训练模型,对于本文所述的超参数基础模型，每个训练步骤耗时约0.4秒。基础模型总共训练了10万步，耗时12小时。对于大型模型，每步耗时1.0 秒。大型模型训练了30万步（3.5 天）
# NoamOpt是一个包装器（wrapper）,把一个标准的PyTorch优化器（比如 Adam包起来,每次更新参数前先按论文的Noam学习率策略计算并设置当前学习率,再调用优化器的step(),get_std_opt 是用论文推荐的超参创建一个封装器
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer  # 被封装的实际优化器（如 torch.optim.Adam(...)）
        self._step = 0  # 内部计数
        self.warmup = warmup  
        self.factor = factor  # 缩放因子（论文里常用1或2，代码用2）
        self.model_size = model_size  # Transformer的隐藏维度dmodel（例如512）
        self._rate = 0  # 记录当前 lr
        # 注意optimizer.zero_grad()仍需在外部或loss_compute里调用（NoamOpt不做zero_grad）
    
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate  # 把这个rate写入优化器的每个param_group的 'lr'
        self._rate = rate
        self.optimizer.step()  # 调用底层优化器self.optimizer.step()更新参数
        
    def rate(self, step = None):  # 这就是NoamLR的数学公式，等价于论文中。先线性增长（warmup 阶段），达到峰值后按1/根号step衰减
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):  # 创建并返回一个封装了Adam 的配置好参数的NoamOpt优化器
    # d_model从model.src_embed[0].d_model取得（通常src_embed是nn.Sequential或tuple，第一个元素有d_model属性）
    # factor=2、warmup=4000（论文推荐），并用 Adam(..., lr=0, betas=(0.9,0.98), eps=1e-9)作为内部优化器。lr=0是因为实际lr由NoamOpt动态设置
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) 


# 构造损失函数
# 在训练过程中，我们采用了标签平滑法来评估值,这会降低困惑度,因为模型会学习变得更加不确定,但会提高准确率和 BLEU 分数
# 标签平滑把硬one-hot标签变成一个“稍微被平滑过”的概率分布（把一小部分质量从正确类别分给其它类别），以降低模型过度自信、改善泛化与稳定训练。这里用的是KL散度把模型输出与这个平滑后的目标分布做匹配
# 我们使用KL div损失函数来实现标签平滑。我们不使用独热编码的目标分布，而是创建一个包含confidence正确单词和其余单词smoothing分布在整个词汇表中的分布
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):  
        super(LabelSmoothing, self).__init__()
        # nn.KLDivLoss 要求第一个参数是 log-probabilities（log P）,第二个参数是概率分布(Q),即计算∑𝑄log(𝑄/𝑃)
        self.criterion = nn.KLDivLoss(size_average=False)  # 创建KL散度损失函数,size_average=False意味着返回总和（sum），现代PyTorch用reduction='sum'
        self.padding_idx = padding_idx  # 表示 padding 对应的类别索引（通常 0），需要在目标分布里把它处理为 0（不分配概率）
        self.smoothing = smoothing  # 平滑强度s（例如 0.1）
        self.confidence = 1.0 - smoothing  # 正确类别被分配到的概率
        self.size = size  # size：输出类别数V
        self.true_dist = None  # 在 forward 过程中保存“构造后的标签平滑后的真实分布（target distribution）”，让你可以在训练后查看它、可视化它、debug 用

    def forward(self, x, target):  # target 就是训练中模型期望输出的正确答案
        # x是模型输出的logits，形状[batch_size, V]（通常先做 x = log_softmax(model_out) 再传入）；target：整型标签，形状 [N]，每个元素是 0..V-1。（若是序列任务通常把 batch 展平成 N）
        assert x.size(1) == self.size  # 检查类别维度一致
        # 先用 x.data.clone()（旧写法，得到一个 tensor 同 shape）创建一个张量占位，用来构造目标分布。注意：x.data 属于老 API，现代请用 x.detach().clone() 或在 torch.no_grad() 下操作
        true_dist = x.data.clone()
        # fill_ 把每个位置先填入平滑分配值，这里用了 self.smoothing / (self.size - 2)
        # 用-2因为在后续会把 padding_idx 的列设为0并把正确类别列设置为confidence，所以均匀分配的分母要减去 2（一个是正确类别，一个是 padding 类）—也就是说smoothing的质量均匀分配到除正确类别和padding之外的其他类别上
        true_dist.fill_(self.smoothing / (self.size - 2))
        # scatter_(1, index, value)：在 dim=1（类别维）按照 target 索引把 value 写入对应位置；作用：把每个样本的正确类别位置赋为 confidence（即 1 - smoothing）
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0  # 把 padding 类的概率强制为 0，保证 padding 不被当成目标分配概率
        # mask 找到那些样本其目标标签本身就是 padding（例如在 seq-to-seq 中，某些位置是 pad，并不参与预测）
        # 如果存在这样的样本，index_fill_ 把对应样本整行（整条样本的目标分布）置为 0，这样这些位置在计算 KL 时对损失没有贡献（等同于跳过这些位置）
        # 也就是为 trg_y == pad 的位置把目标分布全部置 0（不会参与 loss），对齐 ntokens 的做法
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        # 把构造好的目标分布保存到实例以便调试/检查（self.true_dist）,用 self.criterion（KLDivLoss）计算损失：第一个参数 x 应该是 log P，第二个参数是 Q（这里传 Variable(true_dist, requires_grad=False)，也属于老写法）
        # 注意：现代 PyTorch 不用 Variable，直接传 true_dist（确保 requires_grad=False 或者在 torch.no_grad()下构造即可）,返回的是KL散度的和（因为 size_average=False），通常训练代码会按ntokens做平均loss = loss_compute(...)/ntokens
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
   

# 第一个例子:先尝试一个简单的复制任务。训练一个小型 Transformer，让它学会把输入序列原样复制到输出（即输入==输出）。给定一组来自​​小型词汇表的随机输入符号，目标是生成与输入符号相同的序列
# 合成数据
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))  # 每个样本长度固定为 10，词 id 在 [1, V-1]（0 通常保留作 <pad>），并把第一个 token 强制设为 1（可能代表 <sos>/特殊起始符）
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)  # data 是 (batch, 10)，因此 src 和 tgt 都是 (batch, 10)
# 损失计算
# SimpleLossCompute 把模型 decoder 的原始输出映射到词表概率，计算按 token 归一化的损失（对 padding 做忽略），反向传播并调用优化器一步，最后返回总 loss（不是平均值）
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):  # x为模型的原始 decoder 输出
        x = self.generator(x)
        # criterion返回的是这个 batch（或 N 个 token）的总损失，norm是这个 batch 中真实 token 的数量，不算 pad，loss / norm 就是每个 token 的平均损失
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),  # 把 (batch, tgt_len, vocab) 展平成 (N, vocab)，把目标 y 从 (batch, tgt_len) 展平成 (N,)，其中 N = batch * tgt_len
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:  
            self.opt.step()  # self.opt.step()：调用外面传入的优化器包装（如 NoamOpt.step()）
            self.opt.optimizer.zero_grad()  # 把所有参数的 .grad 清零，为下一次迭代准备
        return loss.data[0] * norm  # 返回总的损失和（与最开始的 self.criterion(... ) 的和一致），因为前面 loss 被 / norm 平均了，因此乘回 * norm 恢复为总 loss 用于统计
# Greedy Decoding贪婪解码
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# 重复多次（epoch）把训练数据喂给 Transformer 做前向+反向传播并用 Noam 学习率调度器更新参数；在每个run_epoch后切换到评估模式在小批验证集上测一次 loss ，以监控模型是否学会了复制任务
for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, 
                    SimpleLossCompute(model.generator, criterion, None)))

# 为了简单起见，这段代码使用贪婪解码来预测翻译结果
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)  # 把输入句子 src 丢进 Encoder，得到它的语义表示 memory。类比读完一句话后“理解了它的含义”
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # 初始化输出，只放一个 <sos>（开始标记），告诉模型：“好，现在开始说话。”
    for i in range(max_len-1):  # 一口气生成 max_len 个词
        out = model.decode(memory, src_mask,   # 把已经生成的部分 ys 送进 Decoder,然后Decoder会根据输入句子的含义memory、已生成的内容ys、严格的因果 Mask不能偷看未来词来预测下一个词的向量表示
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])  # 取最后一个词的位置 → 映射到词表概率
        _, next_word = torch.max(prob, dim = 1)  # 贪婪策略：选概率最大的那个词。不考虑后果，不做权衡，就选眼前最优，所以叫 greedy
        next_word = next_word.data[0]  # 把选出来的词加到句子末尾，用于下一步继续输入，继续预测
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys  # 最终ys 就是模型生成的句子

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))



