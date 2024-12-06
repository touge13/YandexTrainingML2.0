import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    # Вычисляем attention scores
    attention_scores = np.dot(np.dot(decoder_hidden_state.T, W_mult), encoder_hidden_states)
    
    # Применяем softmax для получения весов
    attention_weights = softmax(attention_scores)
    
    # Итоговый attention vector
    attention_vector = attention_weights.dot(encoder_hidden_states.T).T
    return attention_vector


def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    # Преобразуем состояния энкодера и декодера
    encoder_transformed = W_add_enc.dot(encoder_hidden_states)
    decoder_transformed = W_add_dec.dot(decoder_hidden_state)
    
    # Вычисляем attention scores
    attention_scores = v_add.T.dot(np.tanh(encoder_transformed + decoder_transformed))
    
    # Применяем softmax для получения весов
    attention_weights = softmax(attention_scores)
    
    # Итоговый attention vector
    attention_vector = attention_weights.dot(encoder_hidden_states.T).T
    return attention_vector
