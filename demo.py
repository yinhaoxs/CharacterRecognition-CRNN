import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets_v14 as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import os
 
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/OWN_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='output/checkpoints/v12_checkpoint_9_acc_0.9396.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # ratio resize
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    h, w = img.shape
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('results: {0}'.format(sim_pred))
    return sim_pred


if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    # model.load_state_dict(torch.load(args.checkpoint))
    # model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    ### 加载迁移学习后训练好的模型
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')["state_dict"])

    # ### 单张图片进行测试
    # started = time.time()
    # img_paths = args.image_path
    # img = cv2.imread(args.image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    # recognition(config, img, model, converter, device)
    # finished = time.time()
    # print('elapsed time: {0}'.format(finished - started))


    ### 多张图片进行测试
    started = time.time()
    img_paths = args.image_path
    punctuation_all = " .-()（）【】[]//‘’“”〝〞（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】。，、＇：∶；?ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡ ¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊aa︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼"
    punctuation_sense = " .-()（）【】[]//——"


    file = open("crop.txt", "r")
    file_compare = open("context.txt", "w")
    label_dict = dict()
    for line in file.readlines():
        img_name, label = line.strip().split(",")
        label_dict[img_name] = label

    ### 批量测试
    num, num_blank, num_all = 0, 0, 0
    for img_path in os.listdir(img_paths):
        img_dir = os.path.join(img_paths+os.sep, img_path)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
        results = recognition(config, img, model, converter, device)
        label = label_dict[img_path]
        if str(results.strip()) == str(label.strip()):
            num += 1
        else:
            print("{}: {}".format(img_path, results))
            file_compare.write(img_path+": "+results.strip()+"          ("+label.strip()+")"+"\n")

        ### num_all测试
        results_all, label_all = results, label
        for flag in punctuation_sense:
            label_all = label_all.replace(flag, '')

        for flag in punctuation_sense:
            results_all = results_all.replace(flag, '')

        if str(results_all.strip()) == str(label_all.strip()):
            num_all += 1
        else:
            print("{}: {}".format(img_path, results))
            # file_compare.write(img_path + ": " + results.strip() + "(" + label.strip() + ")" + "\n")

    print(num / 635)
    print(num_all / 635)

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))


