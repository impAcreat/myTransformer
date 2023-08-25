import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import english_tokenizer_load
from utils import chinese_tokenizer_load

import config
DEVICE = config.device


def subsequent_mask(size):
    """Mask out subsequent positions."""
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)

    # np.triu(matrix, k): 以矩阵的主对角线向上平移 k 单位的对角线为分界，得到矩阵位于该对角线及以上的部分
    # astype(data_type): 明确数据类型
    # 
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵：
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # torch.from_numpy(np.array): 利用 numpy 数组创建 tensor
    #   注：== 0 起到了取反的作用
    # 
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵：
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            trg = trg.to(DEVICE)
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path):
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=True)
        # 加载 Tokenizer
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()  # 0：padding：确定PAD值
        self.BOS = self.sp_eng.bos_id()  # 2：begin of a sentence
        self.EOS = self.sp_eng.eos_id()  # 3：end of a sentence

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        # 输入：seq 为数据集
        # 输出：依据数据集中句子的长度排序，输出排序后的 index 的 list
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """把中文和英文按照同样的顺序排序, 以英文句子长度排序的(句子下标)顺序为基准"""
        dataset = json.load(open(data_path, 'r'))   # dataset 通过json格式存储，load后为 list格式
        out_en_sent = []    # 英文输出 list
        out_cn_sent = []    # 中文输出 List
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])
            out_cn_sent.append(dataset[idx][1])
        if sort:            # 若选择对数据集进行排序
            sorted_index = self.len_argsort(out_en_sent)    # sorted_index 为依据英文句子长度排序后的 索引集合
            out_en_sent = [out_en_sent[i] for i in sorted_index]    # 使用列表生成式，依据sorted_index,排列中英输出集合
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):     # 依据index检索，查询对应的中英语句
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):              # dataset的长度
        return len(self.out_en_sent)

    def collate_fn(self, batch):    # 整理数据为自定义的 Batch 格式
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        # 使用 tokenizer：除句子分词结果外，添加开始符 BOS 以及结束符 EOS
        # tokenize:
        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # pad_sequence(sequences, batch_first=False, padding_value=0) 方法
        #   作用：由于神经网络训练要求 batch 内数据长度相同，故使用该方法将变长 tensor 填充到等长
        #   参数：被填充值 padding_value=0，填充对象sequences
        # 
        # padding:
        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)
