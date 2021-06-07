
from model import Wav2Vec2Config,Wav2Vec2ForCTC
from wav2vec2_tokenizer import Wav2Vec2Tokenizer
import paddle
import torch
import numpy as np
paddle.set_device('gpu:0')
import pickle
if __name__ == '__main__':
    config = Wav2Vec2Config()
    model = Wav2Vec2ForCTC(config)
    with open('./wav2vec2-base-960h.pkl','rb') as f:
        state_dict = pickle.load(f)

   # state_dict = paddle.load('wav2vec2-base-960h.pdparam')
        model.load_dict(state_dict)
    
    input_values = np.load('input_values.npy')
    pd_input = paddle.Tensor(input_values)
    print('input',float(paddle.max(pd_input)))
    model.eval()
    with paddle.no_grad():
        logits = model(pd_input)
        print(logits)
    print('logits=',paddle.mean(logits))
    
    tokenizer = Wav2Vec2Tokenizer('./vocab.json')
    idx = paddle.argmax(logits,-1)
    text = tokenizer.decode(idx[0])
    print(text)
