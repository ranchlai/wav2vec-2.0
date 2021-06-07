# Wav2vec2 in [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)

This is paddle-paddle version of Facebook's Wav2vec2.0 [1], with code and pre-trained weighted ported from [Fairseq](https://github.com/pytorch/fairseq/) and [huggingface](https://github.com/huggingface/transformers). 

## Dependency
Install PaddlePaddle 2.0.1
``` bash
pip install PaddlePaddle-gpu==2.0.1

```
Install [PaddleAudio](https://github.com/PaddlePaddle/models/tree/develop/PaddleAudio) by
```
git clone https://github.com/PaddlePaddle/models
cd models/PaddleAudio
pip install -e .
``` 
## Supported configs

|name|Finetuning split| Dataset |
| :--- | :--- | :---  |
|wav2vec2-base-960h|960h| [Librispeech](http://www.openslr.org/12)|
|wav2vec2-large-960h|960h| [Librispeech](http://www.openslr.org/12)|
|wav2vec2-base-960h-lv60|960h| [Librispeech](http://www.openslr.org/12) + [Libri-Light](https://github.com/facebookresearch/libri-light)|
|wav2vec2-base-960h-lv60-self|960h| [Librispeech](http://www.openslr.org/12) + [Libri-Light](https://github.com/facebookresearch/libri-light) + Self Training |

## Quickstart

Clone the project,
``` bash
git clone https://github.com/ranchlai/wav2vec2.paddle
cd wav2vec2.paddle
```
Run the speech recognition test with your audio file,
``` bash
python test.py --device "gpu:0" --audio "LJ001-0186.wav" --config "wav2vec2-large-960h-lv60"
```
If successful, you will see output like this, 
``` 
pred==> position of our society that a work of utility might be also a work of art if we cared to make it so
```

If you do not have gpu or run out of gpu memory, try cpu:
``` bash
python test.py --device "cpu" --audio "LJ001-0186.wav" --config "wav2vec2-large-960h-lv60"
```

## Reference
[1] Baevski, Alexei, et al. “Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.” Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 12449–12460.
