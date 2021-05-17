# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
# from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
import numpy as np

PAD = 1
# s:0, pad:1, </s>:2, unk:3

class PositionwiseFeedForward(nn.Module):  # đây chỉ là 1 tập hợp fully connected thôi

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # sao lại dùng conv ta ???????????
        self.w_1_real = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_1_phase = nn.Conv1d(d_in, d_hid, 1)  # position-wise

        self.w_2_real = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.w_2_phase = nn.Conv1d(d_hid, d_in, 1)  # position-wise

        self.layer_norm = nn.LayerNorm(
            d_in)  # ở đây có thể hiểu cái này giống batchnormalization vậy, nhưng mình nghĩ cũng có thể không cần, vì nghe nói lớp này bị loại bỏ
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_real, x_phase):
        residual_real = x_real
        residual_phase = x_phase
        cnn_real = x_real.transpose(1, 2)
        cnn_phase = x_phase.transpose(1, 2)

        w1_real = F.relu(self.w_1_real(cnn_real) - self.w_1_phase(cnn_phase))
        w1_phase = F.relu(self.w_1_real(cnn_phase) + self.w_1_phase(cnn_real))

        output_real = self.w_2_real(w1_real) - self.w_2_phase(w1_phase)
        output_phase = self.w_2_real(w1_phase) + self.w_2_phase(w1_real)

        output_real = output_real.transpose(1, 2)
        output_phase = output_phase.transpose(1, 2)

        output_real = self.dropout(output_real)
        output_phase = self.dropout(output_phase)

        output_real = self.layer_norm(output_real + residual_real)
        output_phase = self.layer_norm(output_phase + residual_phase)
        return output_real, output_phase

class ScaledDotProductAttention(nn.Module):  # Hình như đây là cái khác nhau so với transformer bình thường nè

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q_real, k_real, v_real, q_phase, k_phase, v_phase, mask=None, continue_complex=True):
        # Chú ý công thức tính có phần giao thoa giữa thực và ảo nè
        # torch.bmm(q_real, k_real.transpose(1, 2)) đây là công thức khi mình tính attention mối quan hệ giữa 1 từ với các từ còn lại trong câu nè
        # du liệu sẽ được tính theo batch nên dùng hàm torch.bmm
        attn_real = torch.bmm(q_real, k_real.transpose(1, 2)) - torch.bmm(q_phase, k_phase.transpose(1, 2))

        attn_phase = torch.bmm(q_real, k_phase.transpose(1, 2)) + torch.bmm(q_phase, k_real.transpose(1, 2))

        if (continue_complex):
            attn_real = attn_real / self.temperature  # chia cái này giống như chia căn bậc 2 của d_k á
            attn_phase = attn_phase / self.temperature
            if mask is not None:
                attn_real = attn_real.masked_fill(mask, -np.inf)
                attn_phase = attn_phase.masked_fill(mask, -np.inf)

            attn_real = self.softmax(attn_real)
            attn_real = self.dropout(attn_real)

            attn_phase = self.softmax(attn_phase)
            attn_phase = self.dropout(attn_phase)

            output_real = torch.bmm(attn_real, v_real) - torch.bmm(attn_phase, v_phase)

            output_phase = torch.bmm(attn_real, v_phase) + torch.bmm(attn_phase, v_real)

        else:

            attn = attn_real * attn_real + attn_phase * attn_phase
            # attn=attn/self.temperature
            attn = torch.sqrt(attn)
            attn = attn / self.temperature

            if mask is not None:
                attn = attn.masked_fill(mask, -np.inf)

            attn = self.softmax(attn)

            output_real = torch.bmm(attn, v_real)

            output_phase = torch.bmm(attn, v_phase)

        return output_real, output_phase, attn_real  # Tại sao phải trả ra thêm phần attention của phần thực nữa ta

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        # self.n_head = n_head
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)  # nó cho học trọng số bằng linear, tưởng là nó sẽ tạo 1 ma trận trọng số chứ. Mặc dù biết cái này cũng là học
        self.w_ks = nn.Linear(d_model, n_head * d_k)  # trọng số, nhưng có thể thấy nó đã mất đi tính không gian
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k * 2, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v,
                            d_model)  # tạo ma trận để có kể kết hợp hết mấy cái head trả về output là vector
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q_real, k_real, v_real, q_phase, k_phase, v_phase, mask=None):
        """
        Hiểu tại sao lại ghi riêng q, k, v không. Trước mình nghĩ mình có input là 1 vector, rồi đem nhân với 3 ma trận trọng số. Ở trong code này
        thì đúng là như vậy, nhưng thật ra tổng quát hơn thì có thể có 3 input vector khác nhau ứng với q, k , v -> rồi mới đem nhân tương ứng
        với 3 ma trận trọng số q, k, v
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b_real, len_q_real, _ = q_real.size()
        sz_b_real, len_k_real, _ = k_real.size()
        sz_b_real, len_v_real, _ = v_real.size()

        sz_b_phase, len_q_phase, _ = q_phase.size()
        sz_b_phase, len_k_phase, _ = k_phase.size()
        sz_b_phase, len_v_phase, _ = v_phase.size()

        residual_real = q_real
        residual_phase = q_phase
        # Hiểu tại sao nó là tensor 4 chiều nha, chiều đầu tiên là số lượng câu, thứ 2 là số từ trong mỗi câu, số 3 là số head và cuối cùng sô chiều ứng với d
        q_real = self.w_qs(q_real).view(sz_b_real, len_q_real, n_head, d_k)
        k_real = self.w_ks(k_real).view(sz_b_real, len_k_real, n_head, d_k)
        v_real = self.w_vs(v_real).view(sz_b_real, len_v_real, n_head, d_v)

        q_phase = self.w_qs(q_phase).view(sz_b_phase, len_q_phase, n_head, d_k)
        k_phase = self.w_ks(k_phase).view(sz_b_phase, len_k_phase, n_head, d_k)
        v_phase = self.w_vs(v_phase).view(sz_b_phase, len_v_phase, n_head, d_v)

        # permute là hoán vị trí của các chiều,
        # Khi dữ liệu được tạo ra lần đầu tiên với 1 shape được chỉ định, thì khi đó các phần tử trong storage được sắp xếp liên tục theo shape đó
        # Nên khi mình reshape lại thì sẽ tạo 1 dữ liệu shape mới cùng trỏ vào storage đó, và dữ liệu trong storage không được sắp xếp liên tục theo shape mới
        # này. Nên khi check is_contiguous sẽ trả ra false. Mà chỉ những biến nào là contiguous thì mới dùng hàm view được, nên phải đưa nó về contiguous trước.
        q_real = q_real.permute(2, 0, 1, 3).contiguous().view(-1, len_q_real, d_k)  # (n*b) x lq x dk
        k_real = k_real.permute(2, 0, 1, 3).contiguous().view(-1, len_k_real, d_k)  # (n*b) x lk x dk
        v_real = v_real.permute(2, 0, 1, 3).contiguous().view(-1, len_v_real, d_v)  # (n*b) x lv x dv

        q_phase = q_phase.permute(2, 0, 1, 3).contiguous().view(-1, len_q_phase, d_k)  # (n*b) x lq x dk
        k_phase = k_phase.permute(2, 0, 1, 3).contiguous().view(-1, len_k_phase, d_k)  # (n*b) x lk x dk
        v_phase = v_phase.permute(2, 0, 1, 3).contiguous().view(-1, len_v_phase, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x .. chưa hiểu cái mask ở đây

        output_real, output_phase, attn = self.attention(q_real, k_real, v_real, q_phase, k_phase, v_phase, mask=mask,
                                                         continue_complex=False)

        output_real = output_real.view(n_head, sz_b_real, len_q_real,d_v)  # (n_head, batch_size, maxlen, d) -> (1,2,0,3)->(batch_size, maxlen, n_head, d)
        output_real = output_real.permute(1, 2, 0, 3).contiguous().view(sz_b_real, len_q_real, -1)  # b x lq x (n*dv)

        output_phase = output_phase.view(n_head, sz_b_phase, len_q_phase, d_v)
        output_phase = output_phase.permute(1, 2, 0, 3).contiguous().view(sz_b_phase, len_q_phase,-1)  # b x lq x (n*dv)

        output_real = self.dropout(self.fc(output_real))  # (batch_size, maxlen, d_model)
        output_real = self.layer_norm(output_real + residual_real)

        output_phase = self.dropout(self.fc(output_phase))
        output_phase = self.layer_norm(output_phase + residual_phase)

        return output_real, output_phase

class ComplexTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8

        self.self_attn = self.build_self_attention(self.embed_dim, args)  # Chỗ cần sửa

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before

        self.final_layer_norm = LayerNorm(self.embed_dim)

        self.pos_ffn = PositionwiseFeedForward(self.embed_dim, args.encoder_ffn_embed_dim, dropout=args.dropout)


    def build_self_attention(self, embed_dim, args):
        # n_head, d_model, d_k, d_v, dropout=0.1)
        return MultiHeadAttention(
            args.encoder_attention_heads,
            embed_dim,
            d_k=64,
            d_v=64,
            dropout=args.attention_dropout,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    # x là real, còn img là phần ảo -> hạn chế thay đổi code nhất có thể ở thời điểm hiện tại
    def forward(self,
                enc_output_real,
                enc_output_phase,
                encoder_padding_mask: Optional[Tensor],
                non_padding_mask,
                attn_mask: Optional[Tensor] = None,
                ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        enc_output_real, enc_output_phase = self.self_attn(
            q_real=enc_output_real,
            k_real=enc_output_real,
            v_real=enc_output_real,
            q_phase=enc_output_phase,
            k_phase=enc_output_phase,
            v_phase=enc_output_phase,
            mask=encoder_padding_mask,                # Không biết tg này đúng cái shape chưa
        )
        # shape (max_len, batchsize, dim)
        # Ở trong kia có bước lấy embed trả ra * non_pad_mask nữa, chắc mình thử bỏ qua bước này trước.    (***)
        enc_output_real = enc_output_real * non_padding_mask
        enc_output_phase = enc_output_phase * non_padding_mask

        enc_output_real, enc_output_phase = self.pos_ffn(enc_output_real, enc_output_phase)

        enc_output_real = enc_output_real * non_padding_mask
        enc_output_phase = enc_output_phase * non_padding_mask

        return enc_output_real, enc_output_phase


class ComplexTransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True
        self.pos_ffn = PositionwiseFeedForward(self.embed_dim, args.decoder_ffn_embed_dim, dropout=args.dropout)
        self.onnx_trace = False

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiHeadAttention(
            args.decoder_attention_heads,
            embed_dim,
            d_k=64,
            d_v=64,
            dropout=args.attention_dropout,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiHeadAttention(
            args.decoder_attention_heads,
            embed_dim,
            d_k=64,
            d_v=64,
            dropout=args.attention_dropout,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        dec_input_real,
        dec_input_phase,
        enc_output_real,
        enc_output_phase,
        self_attn_mask: Optional[torch.Tensor] = None,
        dec_enc_attn_mask: Optional[torch.Tensor] = None,
        non_padding_mask = None,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        dec_output_real, dec_output_phase = self.self_attn(
            q_real=dec_input_real,
            k_real=dec_input_real,
            v_real=dec_input_real,
            q_phase=dec_input_phase,
            k_phase=dec_input_phase,
            v_phase=dec_input_phase,
            mask=self_attn_mask,
        )

        dec_output_real = dec_output_real * non_padding_mask
        dec_output_phase = dec_output_phase * non_padding_mask

        dec_output_real, dec_output_phase = self.encoder_attn(
            dec_output_real,
            enc_output_real,
            enc_output_real,
            dec_output_phase,
            enc_output_phase,
            enc_output_phase,
            mask=dec_enc_attn_mask)

        dec_output_real = dec_output_real * non_padding_mask
        dec_output_phase = dec_output_phase * non_padding_mask

        dec_output_real, dec_output_phase = self.pos_ffn(dec_output_real, dec_output_phase)

        dec_output_real = dec_output_real * non_padding_mask
        dec_output_phase = dec_output_phase * non_padding_mask

        return dec_output_real,dec_output_phase

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
