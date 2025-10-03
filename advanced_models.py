import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Transformer için pozisyonel encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(context)

class TransformerEncoder(nn.Module):
    """Transformer Encoder for time series"""
    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        # x shape: (batch, 1, seq_len) -> (batch, seq_len, 1)
        x = x.transpose(1, 2)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoding
        encoded = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        return self.classifier(pooled)

class ResidualBlock1D(nn.Module):
    """1D Residual Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ResNet1D(nn.Module):
    """1D ResNet for time series"""
    def __init__(self, input_length=3197, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class LSTMAttention(nn.Module):
    """LSTM with Attention mechanism"""
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 1, seq_len) -> (batch, seq_len, 1)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size*2)
        
        return self.classifier(context)

class WaveNet(nn.Module):
    """WaveNet-inspired architecture for time series"""
    def __init__(self, input_channels=1, residual_channels=32, skip_channels=32, 
                 end_channels=128, num_classes=2, num_layers=10):
        super().__init__()
        self.num_layers = num_layers
        
        self.start_conv = nn.Conv1d(input_channels, residual_channels, 1)
        
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            self.dilated_convs.append(
                nn.Conv1d(residual_channels, residual_channels * 2, 2, 
                         dilation=dilation, padding=dilation)
            )
            self.residual_convs.append(
                nn.Conv1d(residual_channels, residual_channels, 1)
            )
            self.skip_convs.append(
                nn.Conv1d(residual_channels, skip_channels, 1)
            )
        
        self.end_conv1 = nn.Conv1d(skip_channels, end_channels, 1)
        self.end_conv2 = nn.Conv1d(end_channels, num_classes, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.start_conv(x)
        skip_connections = []
        
        for i in range(self.num_layers):
            # Dilated convolution
            conv_out = self.dilated_convs[i](x)
            
            # Gated activation
            filter_out, gate_out = conv_out.chunk(2, dim=1)
            gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)
            
            # Residual connection
            residual = self.residual_convs[i](gated)
            x = x + residual
            
            # Skip connection
            skip = self.skip_convs[i](gated)
            skip_connections.append(skip)
        
        # Combine skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        # Final convolutions
        out = F.relu(self.end_conv1(skip_sum))
        out = self.end_conv2(out)
        
        # Global pooling
        out = self.global_pool(out).squeeze(-1)
        
        return out

class EfficientNet1D(nn.Module):
    """EfficientNet-inspired 1D architecture"""
    def __init__(self, input_length=3197, num_classes=2):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )
        
        # MBConv blocks (simplified)
        self.blocks = nn.Sequential(
            self._make_mbconv(32, 64, 3, 1, 2),
            self._make_mbconv(64, 128, 3, 2, 2),
            self._make_mbconv(128, 256, 5, 2, 3),
            self._make_mbconv(256, 512, 3, 2, 2),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def _make_mbconv(self, in_channels, out_channels, kernel_size, stride, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(MBConvBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size,
                stride if i == 0 else 1
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio=6):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        expanded_channels = in_channels * expand_ratio
        
        # Expansion
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, expanded_channels, 1),
            nn.BatchNorm1d(expanded_channels),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv1d(expanded_channels, expanded_channels, kernel_size, 
                     stride, padding=kernel_size//2, groups=expanded_channels),
            nn.BatchNorm1d(expanded_channels),
            nn.SiLU()
        )
        
        # Squeeze and Excitation
        self.se = SqueezeExcitation(expanded_channels)
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv1d(expanded_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        
        if self.use_residual:
            x = x + residual
        
        return x

class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation module"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.SiLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)