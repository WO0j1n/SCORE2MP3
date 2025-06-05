import sys
sys.path.append('/content/drive/MyDrive/MusicMNIST_Project/models')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.layers import LayerNormalization, Add, MultiHeadAttention
from tensorflow.keras.layers import TimeDistributed
from official.nlp.modeling.layers.position_embedding import PositionEmbedding

# 하이퍼파라미터
sequence_length = 72
embedding_dim = 128
vocab_size = 10000


def transformer_encoder_block(x, num_heads, ff_dim):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dense(embedding_dim)(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


def transformer_decoder_block(x, encoder_output, num_heads, ff_dim):
    # 셀프 어텐션 (디코더 내부)
    attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x, x)
    x = Add()([x, attn1])
    x = LayerNormalization(epsilon=1e-6)(x)

    # 크로스 어텐션 (인코더와 연결)
    attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, encoder_output, encoder_output)
    x = Add()([x, attn2])
    x = LayerNormalization(epsilon=1e-6)(x)

    # 피드포워드
    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dense(embedding_dim)(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


# ✅ 인코더
encoder_inputs = Input(shape=(sequence_length,))
x = Embedding(vocab_size, embedding_dim)(encoder_inputs)
x = PositionEmbedding(max_length=sequence_length)(x)
for _ in range(6):
    x = transformer_encoder_block(x, num_heads=4, ff_dim=128)
encoder_outputs = x

# ✅ 디코더
decoder_inputs = Input(shape=(sequence_length,))
x = Embedding(vocab_size, embedding_dim)(decoder_inputs)
x = PositionEmbedding(max_length=sequence_length)(x)
for _ in range(6):
    x = transformer_decoder_block(x, encoder_outputs, num_heads=4, ff_dim=128)

# ✅ 여기 핵심: 시퀀스 전체 예측 (TimeDistributed 사용)
decoder_outputs = TimeDistributed(Dense(88, activation='softmax'))(x)

# ✅ 모델 생성
transformer = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
