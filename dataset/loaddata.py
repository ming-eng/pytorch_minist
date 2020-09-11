import os
from utils.tools import text2vec
import json
def make_dataset(data_path,sample_conf):
    """

    :param data_path: 图片路径
    :param alphabet:
    :param num_class:
    :param num_char:
    :return:
    """

    num_char=sample_conf['max_captcha']
    img_names = os.listdir(data_path)
    samples = []
    for img_name in img_names:
        img_path = os.path.join(data_path, img_name)
        target_str = img_name.split('_')[0][-1]
        assert len(target_str) == num_char
        try:
            target=text2vec(target_str,sample_conf)
            samples.append((img_path, target))
        except:
            pass
    return samples
if __name__ == '__main__':
    with open("../conf/sample_config.json", "r",encoding='utf-8') as f:
        sample_conf = json.load(f)
    train_path=sample_conf['train_image_dir']
    # samples=make_dataset(train_path,sample_conf)
    # print(samples)
    from PIL import Image
    img_path=r'D:\\BaiduNetdiskDownload\\数据集识别\\极验汉字分类_36w\\data\\334600贡_3.jpg'
    img = Image.open(img_path)
    img=img.resize((180, 100), Image.ANTIALIAS)
    img.save('1.jpg')

