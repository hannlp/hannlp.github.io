---
title: Some Pytorch practices of LSTM
date: 2021-04-05
tags:
- 深度学习
---

# 前言
本篇博客记录了我对LSTM的理论学习、PyTorch上LSTM和LSTMCell的学习，以及用LSTM对Seq2Seq框架+注意力机制的实现。还包括了很多有趣的细节，包括RNNs对批量序列数据Padding的处理，以及多层RNNs中Dropout的使用等等。

# 0 好奇心害死猫
最近在复现Bahdana的注意力机制NMT模型(PyTorch)，第一次使用LSTM作为编码器和解码器的基本结构。发现fairseq的源码中编码器使用LSTM，而解码器使用的却是LSTMCell；而OpenNMT的编码器和解码器实现都只使用了LSTM，遂产生好奇。原因竟然是...

# 1 LSTM速览
## 1.1 LSTM流程图
![](https://i.loli.net/2021/04/05/pCeGQALRIy2NVoc.png)

## 1.2 LSTM的关键
## 1.3 与RNN的对比
这里引用[知乎-予以初始](https://www.zhihu.com/question/439243827/answer/1712516368)的回答，非常通俗易懂

**RNN**用于信息传输通路只有一条，并且该通路上的计算包含多次非线性激活操作。长记忆丢失是因为梯度消失，而梯度消失的主谋就是多层激活函数的嵌套，导致梯度反传时越乘越小（激活函数的导数<=1），乃至下溢出。所以后面的梯度传递不到前方，无法建立长时依赖。

**LSTM**引入了两条计算通道(**C**和**h**) 用于信息传输，其中**C**通道上的计算相对简单，较多的是矩阵的线性转换，没有太多的非线性激活操作。梯度反传时可以在**C**通道上平稳的传输到前方，从而建立长时依赖。所以**C**通道主要用于建立长时依赖，**h**通道用于建立短时依赖。

要说的是，LSTM的设计只是较RNN**缓解**了梯度消失问题，并没有完全解决。与Transformer的自注意力相比，LSTM的顺序输入的方式影响了模型的并行性，但符合人对序列的理解方式。

# 2 多层LSTM
![](https://i.loli.net/2021/04/05/scw8fu5I27DQjSN.png)

# 3 PyTorch中的LSTM
由于深度学习框架对模型成熟的封装，RNN这类模型的输入输出、使用方法基本一致。这里以LSTM为例，可以很容易的掌握其他所有RNNs
## 3.1 LSTM

## 3.2 LSTMCell

# 4 PyTorch实践：Encoder-Decoder模型
## 4.1 用LSTM写Encoder
```python
class Encoder(nn.Module):
    def __init__(self, n_src_words, d_model, src_pdx, n_layers, p_drop, bidirectional):
        super().__init__()
        self.d_model, self.n_layers, self.src_pdx = d_model, n_layers, src_pdx
        self.n_directions = 2 if bidirectional else 1
        self.input_embedding = nn.Embedding(n_src_words, d_model, padding_idx=src_pdx)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                            dropout=p_drop, batch_first=True, bidirectional=bidirectional)
    
    def forward(self, src_tokens):
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens)
        batch_size, src_lens = src_tokens.size(0), src_tokens.ne(self.src_pdx).long().sum(dim=-1)
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(
            src_embed, src_lens.to('cpu'), batch_first=True, enforce_sorted=False
        )
        
        packed_encoder_out, (hiddens, cells) = self.lstm(packed_src_embed)

        # - encoder_out: (batch_size, src_len, n_directions * d_model) where 3rd is last layer [h_fwd; (h_bkwd)]
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(packed_encoder_out, batch_first=True)

        # - hiddens & cells: (n_layers, batch_size, n_directions * d_model)
        if self.n_directions == 2:
            hiddens = self._combine_bidir(hiddens, batch_size)
            cells = self._combine_bidir(cells, batch_size)

        return encoder_out, hiddens, cells

    def _combine_bidir(self, outs, batch_size):
        out = outs.view(self.n_layers, 2, batch_size, -1).transpose(1, 2).contiguous()
        return out.view(self.n_layers, batch_size, -1)
```
## 4.2 用LSTMCell写带attention的Decoder
```python
class AttentionLayer(nn.Module):
    # general attention in 2015 luong et.
    def __init__(self, d_src, d_tgt):
        super().__init__()
        self.linear_in = nn.Linear(d_tgt, d_src, bias=False)

    def forward(self, source, memory, mask):
        # - source: (batch_size, tgt_len, d_tgt), memory: (batch_size, src_len, d_src)
        score = torch.matmul(self.linear_in(source), memory.transpose(1, 2))
        score.masked_fill_(mask, -1e9)
        # - score: (batch_size, tgt_len, src_len)
        attn = F.softmax(score, dim=-1)
        return attn

class Decoder(nn.Module):
    def __init__(self, n_tgt_words, d_model, tgt_pdx, n_layers, p_drop, bidirectional):
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers
        self.n_directions = 2 if bidirectional else 1
        self.input_embedding = nn.Embedding(n_tgt_words, d_model, padding_idx=tgt_pdx)

        self.attention = AttentionLayer(d_src=self.n_directions * d_model, d_tgt=d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, 
                            dropout=p_drop, batch_first=True, bidirectional=False)

        self.linear_out = nn.Linear(self.n_directions * d_model + d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, prev_tgt_tokens, encoder_out, hiddens, cells, src_mask):
        # - tgt_embed: (batch_size, src_len, d_model)
        tgt_embed = self.input_embedding(prev_tgt_tokens)

        # - decoder_states: (batch_size, tgt_len, d_model)
        decoder_states, _ = self.lstm(tgt_embed)

        attn = self.attention(source=decoder_states, memory=encoder_out, mask=src_mask.unsqueeze(1))
        # - attn: (batch_size, tgt_len, src_len), encoder_out: (batch_size, src_len, d_src)
 
        context = torch.matmul(attn, encoder_out)
        # - context: (batch_size, tgt_len, d_src)
        
        decoder_out = self.dropout(self.linear_out(torch.cat([context, decoder_states], dim=-1)))
        # - decoder_out: (batch_size, tgt_len, d_model)
        return decoder_out
```

# 参考资料
1. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
2. [Fully understand LSTM network and input, output, hidden_size and other parameters](https://programmersought.com/article/91264364976/)
3. [LSTM细节分析理解（pytorch版）](https://zhuanlan.zhihu.com/p/79064602)
4. [混合前端的seq2seq模型部署](https://cloud.tencent.com/developer/article/1507554)