import clip
import torch
from torch.utils.data import Dataset
from preprocess_data import encode
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
#import cv2 as cv
from imageio import imread
from random import *
from PIL import Image
import skimage.io as io


class LEVIRCCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, encoder_network, data_folder, list_path, split, token_folder = None, word_vocab = None, max_length = 40, allow_unk = 0, max_iters=None):
        """
        :param data_folder: folder where image files are stored
        :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param token_folder: folder where token files are stored
        :param vocab_file: the name of vocab file
        :param max_length: the maximum length of each caption sentence
        :param max_iters: the maximum iteration when loading the data
        :param allow_unk: whether to allow the tokens have unknow word or not
        """
        self.encoder_network = encoder_network
        if 'CLIP' in self.encoder_network:
            encoder_network = encoder_network.replace('CLIP-','')
            clip_model, self.preprocess = clip.load(encoder_network, device='cpu', jit=False)

        # self.mean=[100.6790,  99.5023,  84.9932]
        # self.std=[50.9820, 48.4838, 44.7057]
        self.mean = [0.39073*255,  0.38623*255, 0.32989*255]
        self.std = [0.15329*255,  0.14628*255, 0.13648*255]
        self.list_path = list_path
        self.split = split
        self.max_length = max_length

        assert self.split in {'train', 'val', 'test'}
        self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]

        self.word_vocab = word_vocab
        self.allow_unk = allow_unk
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
        self.files = []
        if split =='train':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name.split('-')[0])
                img_fileB = img_fileA.replace('A', 'B')

                imgA = img_fileA  # imread(img_fileA)
                imgB = img_fileB  # imread(img_fileB)

                if '-' in name:
                    token_id = name.split('-')[-1]
                else:
                    token_id = None
                    # print('\n\n\n\n training dataset!!!!\n\n\n')
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA_path": img_fileA,
                    "imgB_path": img_fileB,
                    "imgA": imgA,
                    "imgB": imgB,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name.split('-')[0]
                })
        elif split =='val':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
                img_fileB = img_fileA.replace('A', 'B')

                imgA = img_fileA  # imread(img_fileA)
                imgB = img_fileB  # imread(img_fileB)

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA_path": img_fileA,
                    "imgB_path": img_fileB,
                    "imgA": imgA,
                    "imgB": imgB,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name.split('.')[0]
                })
        elif split =='test':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
                img_fileB = img_fileA.replace('A', 'B')

                imgA = img_fileA#imread(img_fileA)
                imgB = img_fileB#imread(img_fileB)

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA_path": img_fileA,
                    "imgB_path": img_fileB,
                    "imgA": imgA,
                    "imgB": imgB,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name.split('.')[0]
                })
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        if 'CLIP' not in self.encoder_network:
            imgA = imread(datafiles["imgA"])
            imgB = imread(datafiles["imgB"])
            imgA = np.asarray(imgA, np.float32).transpose(2, 0, 1)
            imgB = np.asarray(imgB, np.float32).transpose(2, 0, 1)
            for i in range(len(self.mean)):
                imgA[i, :, :] -= self.mean[i]
                imgA[i, :, :] /= self.std[i]
                imgB[i, :, :] -= self.mean[i]
                imgB[i, :, :] /= self.std[i]
        else:
            imgA = io.imread(datafiles["imgA_path"])
            imgB = io.imread(datafiles["imgB_path"])
            # CLIP process
            imgA = self.preprocess(Image.fromarray(imgA))
            imgB = self.preprocess(Image.fromarray(imgB))

        if datafiles["token"] is not None:
            caption = open(datafiles["token"])
            caption = caption.read()
            caption_list = json.loads(caption)

            #token = np.zeros((1, self.max_length), dtype=int)
            #j = randint(0, len(caption_list) - 1)
            #tokens_encode = encode(caption_list[j], self.word_vocab,
            #            allow_unk=self.allow_unk == 1)
            #token[0, :len(tokens_encode)] = tokens_encode
            #token_len = len(tokens_encode)

            token_all = np.zeros((len(caption_list), self.max_length),dtype=int)
            token_all_len = np.zeros((len(caption_list),1),dtype=int)
            for j, tokens in enumerate(caption_list):
                nochange_cap = ['<START>', 'the', 'scene', 'is', 'the', 'same', 'as', 'before', '<END>']
                if nochange_cap in caption_list:
                    change_flag = 0
                else:
                    change_flag = 1
                if self.split == 'train' and change_flag==0:
                    tokens = nochange_cap
                tokens_encode = encode(tokens, self.word_vocab,
                                    allow_unk=self.allow_unk == 1)
                token_all[j,:len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                j = randint(0, len(caption_list) - 1)
                # print(f'\n\n\n\n {j}\n\n\n')
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            token_all = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = np.zeros(1, dtype=int)
            token_all_len = np.zeros(1, dtype=int)

        # imgA, imgB, token_all, token_all_len, token, np.array(token_len), name
        out_dict = {
            'imgA': imgA,
            'imgB': imgB,
            'token_all': token_all,
            'token_all_len': token_all_len,
            'token': token,
            'token_len': token_len,
            'name': name
        }
        return out_dict

