from include.header import *

def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256,seq_len=29, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
    
        x = x + self.pe
        return self.dropout(x), self.pe

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.scale = 1 / (d_head ** 0.5)
    
    def forward(self, Q, K, V):
        # (bs, n_head, n_q_seq, n_k_seq)

        # QK^T/d_k^0.5
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale) # Query Key MatMul / Scale

        # masking은 필요가 업음
        # scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)

        attn_prob = nn.Softmax(dim=-1)(scores) # softmax
        # (bs, n_head, n_q_seq, d_v)

        # QK^T/d_k^0.5 * V
        context = torch.matmul(attn_prob, V) # MatMul
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)

        return context, attn_prob

class MultiHeadAttention(nn.Module):
    def __init__(self, i_hidden=256,d_hidden=256, n_head=8,dropout=0.1,activation_query=None,activation_value=None,bias=False):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.d_head = d_hidden // n_head

        # print('d_hidden : ',d_hidden, ' / n_head : ',n_head,' / d_head : ',self.d_head)

        self.W_Q = nn.Linear(i_hidden, d_hidden,bias=bias)
        self.W_K = nn.Linear(i_hidden, d_hidden,bias=bias)
        self.W_V = nn.Linear(i_hidden, d_hidden,bias=bias)
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.active = F.gelu
        self.linear = nn.Linear(d_hidden, i_hidden,bias=bias)
        self.activation_query = activation_query
        self.activation_value = activation_value
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        # (bs, n_head, n_q_seq, d_head)
        # print('input Q size : ',Q.shape)
        if self.activation_query == None:
            # print('Not use multi head active')
            q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
            # (bs, n_head, n_k_seq, d_head)
            k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
            # (bs, n_head, n_v_seq, d_head)
            v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        else:
            # print('use multi head active')
            q_s = self.activation_query(self.W_Q(Q)).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
            # (bs, n_head, n_k_seq, d_head)
            k_s = self.activation_query(self.W_K(K)).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
            # (bs, n_head, n_v_seq, d_head)
            v_s = self.activation_query(self.W_V(V)).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        
        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s)
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_hidden)
        # (bs, n_head, n_q_seq, e_embd)
        
        output = self.linear(context)
        
        if self.activation_value is not None:
            # print('use multi head linear active')
            output = self.activation_value(output)
        
        if self.dropout_prob != 0.:
            # print('use dropout')
            output = self.dropout(output)
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob

class Gated_MultiHeadAttention(nn.Module):
    def __init__(self, i_hidden=256,d_hidden=256, n_head=8,dropout=0.1,activation_query=None,activation_value=None,bias=False):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.d_head = d_hidden // n_head

        # print('d_hidden : ',d_hidden, ' / n_head : ',n_head,' / d_head : ',self.d_head)

        self.W_Q = nn.Linear(i_hidden, d_hidden,bias=bias)
        self.W_K = nn.Linear(i_hidden, d_hidden,bias=bias)
        self.W_V = nn.Linear(i_hidden, d_hidden,bias=bias)
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.active = F.gelu
        self.linear = nn.Linear(d_hidden, i_hidden,bias=bias)
        self.activation_query = activation_query
        self.activation_value = activation_value

        self.v_gate = nn.Linear(self.d_head,self.d_head,bias=bias)
        # self.q_gate = nn.Linear(self.d_head,self.d_head,bias=bias)
        # self.k_gate = nn.Linear(self.d_head,self.d_head,bias=bias)

        self.sigmoid = nn.Sigmoid()
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        # (bs, n_head, n_q_seq, d_head)
        # print('input Q size : ',Q.shape)
        # gate_v = self.v_gate(V)
        # gate_v = self.sigmoid(gate_v)

        # V = torch.mul(gate_v,V)

        if self.activation_query == None:
            # print('Not use multi head active')
            q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
            # (bs, n_head, n_k_seq, d_head)
            k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
            # (bs, n_head, n_v_seq, d_head)
            v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        else:
            # print('use multi head active')
            q_s = self.activation_query(self.W_Q(Q)).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
            # (bs, n_head, n_k_seq, d_head)
            k_s = self.activation_query(self.W_K(K)).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
            # (bs, n_head, n_v_seq, d_head)
            v_s = self.activation_query(self.W_V(V)).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        gate_v = self.v_gate(v_s)
        gate_v = self.sigmoid(gate_v)        
        # gate_v = self.v_gate(v_s)
        # gate_k = self.k_gate(k_s)
        # gate_q = self.q_gate(q_s)

        # gate_v = self.sigmoid(gate_v + gate_k + gate_q)

        v_s = torch.mul(gate_v,v_s)


        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s)
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_hidden)
        # (bs, n_head, n_q_seq, e_embd)
        
        output = self.linear(context)
        
        if self.activation_value is not None:
            # print('use multi head linear active')
            output = self.activation_value(output)
        
        if self.dropout_prob != 0.:
            # print('use dropout')
            output = self.dropout(output)
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob


class PoswiseFeedForwardNet(nn.Module): #
    def __init__(self, i_hidden=256,d_hidden=256,f_hidden=512,dropout=0.1,first_active=F.gelu,second_active=None,bias=False):
        super().__init__()

        self.fc1 = nn.Linear(i_hidden, f_hidden,bias=bias)
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(f_hidden, i_hidden,bias=bias)
        self.first_active=first_active
        self.second_active=second_active

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    # 첫번째 FC에 대해서는 Activation Function 적용
    # 각 FC 이후 Dropout 적용 (pytorch transformer)
    def forward(self, inputs): # 
        # (bs, d_ff, n_seq)
        output = self.fc1(inputs)
        if self.first_active is not None:
            # print('use first active')
            output = self.first_active(output)

        if self.dropout_prob != 0.:
            # print('use Dropout')
            output = self.dropout(output)
        # (bs, n_seq, d_hidn)
        output = self.fc2(output)
        if self.second_active is not None:
            # print('use second active')
            output = self.second_active(output)

        if self.dropout_prob != 0.:
            # print('use Dropout')
            output = self.dropout(output)
        # output = self.active(self.fc2(output))
        # output = self.dropout(output)
        # (bs, n_seq, d_hidn)
        return output



class EncoderLayer(nn.Module):
    def __init__(self,i_hidden=256, d_hidden=256,f_hidden=512,n_head=8,dropout=0.1,layer_norm_epsilon=1e-12,
    mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,layer_norm=True,layer_norm_first=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(i_hidden=i_hidden,d_hidden=d_hidden, n_head=8,dropout=dropout,activation_query=mq_active,activation_value=mv_active)
        self.layer_norm1 = nn.LayerNorm(i_hidden, eps=layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(i_hidden=i_hidden,d_hidden=d_hidden,f_hidden=f_hidden,dropout=dropout,first_active=f_first_active,second_active=f_second_active)
        self.layer_norm2 = nn.LayerNorm(i_hidden, eps=layer_norm_epsilon)
        self.layer_norm_first = layer_norm_first
        self.layer_norm = layer_norm
    def forward(self, inputs):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        if self.layer_norm_first:
            skip = inputs
            if self.layer_norm is not None:
                inputs = self.layer_norm1(inputs)
            
            att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs)
            # att_outputs = inputs+att_outputs
            
            # print('use Layer Norm')
            att_outputs = skip + att_outputs
            # (bs, n_enc_seq, d_hidn)
            skip = att_outputs
            if self.layer_norm is not None:
                att_outputs = self.layer_norm2(att_outputs)
            
            ffn_outputs = self.pos_ffn(att_outputs)

            # print('Not use Layer Norm')
            ffn_outputs = ffn_outputs + skip 
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        else:
            att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs)
            # att_outputs = inputs+att_outputs
            if self.layer_norm is not None:
                # print('use Layer Norm')
                att_outputs = self.layer_norm1(inputs + att_outputs)
            else:
                # print('Not use Layer Norm')
                att_outputs = inputs+att_outputs
            # (bs, n_enc_seq, d_hidn)

            ffn_outputs = self.pos_ffn(att_outputs)
            if self.layer_norm is not None:
                # print('use Layer Norm')
                ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
            else:
                # print('Not use Layer Norm')
                ffn_outputs = ffn_outputs + att_outputs 
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)

        return ffn_outputs, attn_prob

class Gated_EncoderLayer(nn.Module):
    def __init__(self,i_hidden=256, d_hidden=256,f_hidden=512,n_head=8,dropout=0.1,layer_norm_epsilon=1e-12,
    mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,layer_norm=True,layer_norm_first=False):
        super().__init__()
        self.self_attn = Gated_MultiHeadAttention(i_hidden=i_hidden,d_hidden=d_hidden, n_head=8,dropout=dropout,activation_query=mq_active,activation_value=mv_active)
        self.layer_norm1 = nn.LayerNorm(i_hidden, eps=layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(i_hidden=i_hidden,d_hidden=d_hidden,f_hidden=f_hidden,dropout=dropout,first_active=f_first_active,second_active=f_second_active)
        self.layer_norm2 = nn.LayerNorm(i_hidden, eps=layer_norm_epsilon)
        self.layer_norm_first = layer_norm_first
        self.layer_norm = layer_norm
    def forward(self, inputs):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        if self.layer_norm_first:
            skip = inputs
            if self.layer_norm is not None:
                inputs = self.layer_norm1(inputs)
            
            att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs)
            # att_outputs = inputs+att_outputs
            
            # print('use Layer Norm')
            att_outputs = skip + att_outputs
            # (bs, n_enc_seq, d_hidn)
            skip = att_outputs
            if self.layer_norm is not None:
                att_outputs = self.layer_norm2(att_outputs)
            
            ffn_outputs = self.pos_ffn(att_outputs)

            # print('Not use Layer Norm')
            ffn_outputs = ffn_outputs + skip 
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        else:
            att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs)
            # att_outputs = inputs+att_outputs
            if self.layer_norm is not None:
                # print('use Layer Norm')
                att_outputs = self.layer_norm1(inputs + att_outputs)
            else:
                # print('Not use Layer Norm')
                att_outputs = inputs+att_outputs
            # (bs, n_enc_seq, d_hidn)

            ffn_outputs = self.pos_ffn(att_outputs)
            if self.layer_norm is not None:
                # print('use Layer Norm')
                ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
            else:
                # print('Not use Layer Norm')
                ffn_outputs = ffn_outputs + att_outputs 
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)

        return ffn_outputs, attn_prob


class Encoder(nn.Module):
    def __init__(self, seq_length=29,i_hidden=256,hidden_size=256,f_hidden=512,dropout=0.1,n_head=8,num_layers=6,
    mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,layer_norm=True,layer_norm_first=False):
        super().__init__()

        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(seq_length, hidden_size))
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(seq_length, hidden_size))
        # embedding으로 할 경우 overfitting이 발생
        # self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        # self.pos_encoder = PositionalEncoding(hidden_size,seq_length, dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, i_hidden))
        self.layers = nn.ModuleList([EncoderLayer(i_hidden=i_hidden,d_hidden=hidden_size,f_hidden=f_hidden,dropout=dropout,n_head=n_head
                                                    ,mq_active=mq_active,mv_active=mv_active,f_first_active=f_first_active,f_second_active=f_second_active,
                                                    layer_norm=layer_norm,layer_norm_first=layer_norm_first) for _ in range(num_layers)])
        
    def forward(self, inputs):
        # positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() 
        # positions = torch.arange(inputs.size(1), device=inputs.device, dtype=torch.long).repeat(inputs.size(0),1)
        # outputs = inputs + self.pos_emb(positions)
        
        # outputs = self.pos_encoder(inputs)
        
        outputs = inputs + self.pos_embedding
        
        # print('outputs shape : ',outputs.shape)
        attn_probs = []
        
        for layer in self.layers:
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs)
            attn_probs.append(attn_prob)
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs

class Encoder_new(nn.Module):
    def __init__(self, seq_length=29,i_hidden=256,hidden_size=256,f_hidden=512,dropout=0.1,n_head=8,num_layers=6,
    mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,layer_norm=True,layer_norm_first=False):
        super().__init__()

        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(seq_length, hidden_size))
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(seq_length, hidden_size))
        # embedding으로 할 경우 overfitting이 발생
        # self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        # self.pos_encoder = PositionalEncoding(hidden_size,seq_length, dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, i_hidden))
        self.layers = nn.ModuleList([EncoderLayer(i_hidden=i_hidden,d_hidden=hidden_size,f_hidden=f_hidden,dropout=dropout,n_head=n_head
                                                    ,mq_active=mq_active,mv_active=mv_active,f_first_active=f_first_active,f_second_active=f_second_active,
                                                    layer_norm=layer_norm,layer_norm_first=layer_norm_first) for _ in range(num_layers)])
        
    def forward(self, inputs):
        # positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() 
        # positions = torch.arange(inputs.size(1), device=inputs.device, dtype=torch.long).repeat(inputs.size(0),1)
        # outputs = inputs + self.pos_emb(positions)
        
        # outputs = self.pos_encoder(inputs)
        
        outputs = inputs + self.pos_embedding
        
        # print('outputs shape : ',outputs.shape)
        attn_probs = []
        outputs_list = []
        for layer in self.layers:
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs)
            outputs_list.append(outputs)
            attn_probs.append(attn_prob)
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs, outputs_list

class Gated_Encoder(nn.Module):
    def __init__(self, seq_length=29,i_hidden=256,hidden_size=256,f_hidden=512,dropout=0.1,n_head=8,num_layers=6,
    mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,layer_norm=True,layer_norm_first=False):
        super().__init__()

        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(seq_length, hidden_size))
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(seq_length, hidden_size))
        # embedding으로 할 경우 overfitting이 발생
        # self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        # self.pos_encoder = PositionalEncoding(hidden_size,seq_length, dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, i_hidden))
        self.layers = nn.ModuleList([Gated_EncoderLayer(i_hidden=i_hidden,d_hidden=hidden_size,f_hidden=f_hidden,dropout=dropout,n_head=n_head
                                                    ,mq_active=mq_active,mv_active=mv_active,f_first_active=f_first_active,f_second_active=f_second_active,
                                                    layer_norm=layer_norm,layer_norm_first=layer_norm_first) for _ in range(num_layers)])
        
    def forward(self, inputs):
        # positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() 
        # positions = torch.arange(inputs.size(1), device=inputs.device, dtype=torch.long).repeat(inputs.size(0),1)
        # outputs = inputs + self.pos_emb(positions)
        
        # outputs = self.pos_encoder(inputs)
        
        outputs = inputs + self.pos_embedding
        
        # print('outputs shape : ',outputs.shape)
        attn_probs = []
        for layer in self.layers:
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs)
            attn_probs.append(attn_prob)
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs

class DecoderLayer(nn.Module):
    def __init__(self, i_hidden=256, d_hidden=256,f_hidden=512,n_head=8,dropout=0.1,layer_norm_epsilon=1e-12,
                mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,layer_norm=True,layer_norm_first=False):
        super().__init__()

        self.self_attn = MultiHeadAttention(i_hidden=i_hidden,d_hidden=d_hidden, n_head=8,dropout=dropout,activation_query=mq_active,activation_value=mv_active)
        self.layer_norm1 = nn.LayerNorm(i_hidden, eps=layer_norm_epsilon)

        
        self.attn = MultiHeadAttention(i_hidden=i_hidden,d_hidden=d_hidden, n_head=8,dropout=dropout,activation_query=mq_active,activation_value=mv_active)
        self.layer_norm2 = nn.LayerNorm(i_hidden, eps=layer_norm_epsilon)

        self.pos_ffn = PoswiseFeedForwardNet(i_hidden=i_hidden,d_hidden=d_hidden,f_hidden=f_hidden,dropout=dropout,first_active=f_first_active,second_active=f_second_active)
        self.layer_norm3 = nn.LayerNorm(i_hidden, eps=layer_norm_epsilon)
        self.layer_norm_first = layer_norm_first
        self.layer_norm = layer_norm

    # 인코더의 출력 값(enc_src)을 어텐션(attention)하는 구조
    def forward(self, trg, enc_src):
        if self.layer_norm_first:
            skip = trg
            if self.layer_norm is not None:
                trg = self.layer_norm1(trg)
            att_outputs, self_attn_prob = self.self_attn(trg,trg,trg)

            att_outputs = skip + att_outputs

            skip = att_outputs
            if self.layer_norm is not None:
                att_outputs = self.layer_norm2(att_outputs)
            att_outputs, attn_prob = self.attn(trg,enc_src,enc_src)

            att_outputs = skip + att_outputs

            skip = att_outputs
            if self.layer_norm is not None:
                att_outputs = self.layer_norm3(att_outputs)
            
            ffn_outputs = self.pos_ffn(att_outputs)

            ffn_outputs = skip + ffn_outputs
        else:
            skip = trg
            att_outputs, self_attn_prob = self.self_attn(trg,trg,trg)

            if self.layer_norm is not None:
                att_outputs = self.layer_norm1(skip+att_outputs)
            else:
                att_outputs = skip + att_outputs

            skip = att_outputs

            
            att_outputs, attn_prob = self.attn(trg,enc_src,enc_src)
            if self.layer_norm is not None:
                att_outputs = self.layer_norm2(skip+att_outputs)
            else:
                att_outputs = skip + att_outputs

            skip = att_outputs
            
            ffn_outputs = self.pos_ffn(att_outputs)

            if self.layer_norm is not None:
                att_outputs = self.layer_norm3(att_outputs)
            else:
                ffn_outputs = skip + ffn_outputs


        return ffn_outputs, attn_prob


class Decoder(nn.Module):
    def __init__(self, seq_length=29,i_hidden=256,hidden_size=256,f_hidden=512,dropout=0.1,n_head=8,num_layers=6,
    mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,layer_norm=True,layer_norm_first=False):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, i_hidden))
        self.layers = nn.ModuleList([DecoderLayer(i_hidden=i_hidden,d_hidden=hidden_size,f_hidden=f_hidden,dropout=dropout,n_head=n_head
                                                    ,mq_active=mq_active,mv_active=mv_active,f_first_active=f_first_active,f_second_active=f_second_active,
                                                    layer_norm=layer_norm,layer_norm_first=layer_norm_first) for _ in range(num_layers)])
        
    def forward(self, inputs,enc_context):
        outputs = inputs + self.pos_embedding
        
        attn_probs = []
        for layer in self.layers:
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs,enc_context)
            attn_probs.append(attn_prob)
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs
