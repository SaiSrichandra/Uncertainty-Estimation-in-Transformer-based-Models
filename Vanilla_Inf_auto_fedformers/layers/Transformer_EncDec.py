import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",configs=None,lnum = None):
        super(EncoderLayer, self).__init__()
        self.configs = configs
        self.lnum = lnum
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )

        if self.configs.is_training == False and self.configs.noiseATT == True and self.configs.beforeNorm == True and self.lnum == self.configs.layerNum:
            print("Model - {} , Position - {}, Norm - {}, Layer - {} ".format(self.configs.model, 'Attention', 'Before Norm', self.lnum))
            new_x = self.configs.add_noise(new_x,self.configs.noiseType,self.configs.nParam_mu,self.configs.nParam_sigma,self.configs.nParam_rate)

        x = x + self.dropout(new_x)

        x = self.norm1(x)
        
        if self.configs.is_training == False and self.configs.noiseATT == True and self.configs.afterNorm == True and self.lnum == self.configs.layerNum:
            print("Model - {} , Position - {}, Norm - {}, Layer - {} ".format(self.configs.model, 'Attention', 'After Norm', self.lnum))
            x = self.configs.add_noise(x ,self.configs.noiseType,self.configs.nParam_mu,self.configs.nParam_sigma,self.configs.nParam_rate)         

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        if self.configs.is_training == False and self.configs.noiseFF == True and self.configs.beforeNorm == True and self.lnum == self.configs.layerNum:
            print("Model - {} , Position - {}, Norm - {}, Layer - {} ".format(self.configs.model, 'FeedForward', 'Before Norm', self.lnum))
            y = self.configs.add_noise(y ,self.configs.noiseType,self.configs.nParam_mu,self.configs.nParam_sigma,self.configs.nParam_rate) 
        
        p = self.norm2(x + y)

        if self.configs.is_training == False and self.configs.noiseFF == True and self.configs.afterNorm == True and self.lnum == self.configs.layerNum:
            print("Model - {} , Position - {}, Norm - {}, Layer - {} ".format(self.configs.model, 'FeedForward', 'After Norm', self.lnum))
            p = self.configs.add_noise(p ,self.configs.noiseType,self.configs.nParam_mu,self.configs.nParam_sigma,self.configs.nParam_rate)

        return p , attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, configs = None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.configs = configs

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []

        if self.configs.is_training == False and self.configs.noiseType != 'default' and self.configs.layerNum == 0 :
            print("Model - {} , Position - {}, Norm - {}, Layer - {} ".format(self.configs.model, 'Input', 'NA', self.configs.layerNum))
            x = self.configs.add_noise(x,self.configs.noiseType,self.configs.nParam_mu,self.configs.nParam_sigma,self.configs.nParam_rate) 

        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
