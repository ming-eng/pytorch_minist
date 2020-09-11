
import numpy as np
import json

from PIL import Image
##进行one-hot 编码
def text2vec(text,sample_conf ):
    """
    text to one-hot vector
    :param text: source text
    :return: np array
    """

    CAPTCHA_LENGTH=sample_conf['max_captcha']
    if len(text) > CAPTCHA_LENGTH:
        return False
    VOCAB=sample_conf['char_set']
    VOCAB_LENGTH=len(sample_conf['char_set'])
    vector = np.zeros(CAPTCHA_LENGTH * VOCAB_LENGTH)
    for i, c in enumerate(text):
        index = i * VOCAB_LENGTH + VOCAB.index(c)
        vector[index] = 1
    return vector

def vec_text(vec,sample_conf):
    char_set = sample_conf['char_set']
    iter_times=int(len(vec) / len(char_set))
    char_length=len(char_set)
    text=[]
    for v in range(0, iter_times):
        vec_time = vec[v * char_length:(v + 1) *char_length]
        for ve, char in zip(vec_time, char_set):
            if ve == 1:
                text.append(char)
    return ''.join(text)

def img_loader(img_path):
    img = Image.open(img_path)
    img=img.resize((180, 100), Image.ANTIALIAS)
    return img.convert('RGB')
if __name__ == '__main__':
    json={
        "train_image_dir": "./data/train/",
        "test_image_dir": "./data/test/",
        "model_save_dir": "./checkpoints",
        "model_save_path": "./checkpoints/model.pth",
        "image_width": 180,
        "image_height": 100,
        "max_captcha": 4,
        "image_suffix": "png",
        "char_set": "0123456789abcdefghijklmnopqrstuvwxyz",
        "server_url": "http://127.0.0.1:6100/captcha/",
        "acc_stop": 0.99,
        "cycle_save": 500,
        "enable_gpu": 1,
        "train_batch_size": 128,
        "test_batch_size": 100,
        "base_lr": 0.001,
        "epoch": 2000
    }
    vec=text2vec('ch34',json)
    print(vec_text(vec,sample_conf=json))




