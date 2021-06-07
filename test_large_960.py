
from model import Wav2Vec2Config,Wav2Vec2ForCTC
import paddle
import torch
if __name__ == '__main__':
    config = Wav2Vec2Config()
    model = Wav2Vec2ForCTC(config)
    state_dict = paddle.load('wav2vec2-base-960h.pdparam')
    model.load_dict(state_dict)
    
    input_values = torch.load('input.pt')
    pd_input = paddle.Tensor(input_values.numpy())
    model.eval()
    with paddle.no_grad():
        logits = model(pd_input)
    #     print(logits)
    print(paddle.mean(logits))
    
    tokenizer = Wav2Vec2Tokenizer('./vocab.json')
