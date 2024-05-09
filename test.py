import torch.optim
from torch.utils import data
import argparse
import json
from tqdm import tqdm
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from model.model_encoder_attMamba import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_captions(pred_caption, ref_caption, hypotheses, references, name, save_path):
    name = name[0]
    # return 0
    score_dict = get_eval_score([references], [hypotheses])
    Bleu_4 = score_dict['Bleu_4']
    Bleu_4_str = round(Bleu_4, 4)
    Bleu_3 = score_dict['Bleu_3']
    Bleu_3_str = round(Bleu_3, 4)

    json_name = os.path.join(save_path, 'score.json')
    if not os.path.exists(json_name):
        with open(json_name, 'a+') as f:
            key = name.split('.')[0]
            json.dump({f'{key}': {'x': 0}}, f)
        # 关闭文件
        f.close()
    else:
        # 读取JSON文件
        with open(os.path.join(save_path, 'score.json'), 'r') as file:
            data = json.load(file)
            key = name.split('.')[0]
            data[key] = {'x': 0}
        # 写入修改后的数据至同一文件
        with open(os.path.join(save_path, 'score.json'), 'w') as file:
            json.dump(data, file)
        # 关闭文件
        file.close()

    with open(os.path.join(save_path, 'score.json'), 'r') as file:
        data = json.load(file)
        key = name.split('.')[0]
        data[key]['Bleu_3'] = Bleu_3_str
        data[key]['Bleu_4'] = Bleu_4_str
    # 写入修改后的数据至同一文件
    with open(os.path.join(save_path, 'score.json'), 'w') as file:
        json.dump(data, file)
    # 关闭文件
    file.close()

    with open(os.path.join(save_path, name.split('.')[0] + f'_cap.txt'), 'w') as f:
        f.write('pred_caption: ' + pred_caption + '\n')
        f.write('ref_caption: ' + ref_caption + '\n')

def main(args):
    """
    Testing.
    """
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Load checkpoint
    snapshot_full_path = args.checkpoint#os.path.join(args.savepath, args.checkpoint)
    checkpoint = torch.load(snapshot_full_path)

    args.result_path = os.path.join(args.result_path, os.path.basename(snapshot_full_path).replace('.pth', ''))
    if os.path.exists(args.result_path) == False:
        os.makedirs(args.result_path)
    else:
        print('result_path is existed!')
        # 清空文件夹
        for root, dirs, files in os.walk(args.result_path):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))


    # encoder = Encoder(args.network)
    # encoder_trans = AttentiveEncoder(train_stage=args.train_stage, n_layers=args.n_layers,
    #                                       feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
    #                                       heads=args.n_heads, dropout=args.dropout)
    # decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
    #                                   vocab_size=len(word_vocab), max_lengths=args.max_length,
    #                                   word_vocab=word_vocab, n_head=args.n_heads,
    #                                   n_layers=args.decoder_n_layers, dropout=args.dropout)

    encoder = Encoder(args.network)
    encoder_trans = AttentiveEncoder(n_layers=args.n_layers,
                                          feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                          heads=args.n_heads, dropout=args.dropout)
    decoder = DecoderTransformer(decoder_type=args.decoder_type,embed_dim=args.embed_dim,
                                      vocab_size=len(word_vocab), max_lengths=args.max_length,
                                      word_vocab=word_vocab, n_head=args.n_heads,
                                      n_layers=args.decoder_n_layers, dropout=args.dropout)


    encoder.load_state_dict(checkpoint['encoder_dict'])
    encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'])
    decoder.load_state_dict(checkpoint['decoder_dict'])
    # Move to GPU, if available
    encoder.eval()
    encoder = encoder.cuda()
    encoder_trans.eval()
    encoder_trans = encoder_trans.cuda()
    decoder.eval()
    decoder = decoder.cuda()
    print('load model success!')

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        # LEVIR:
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
                LEVIRCCDataset(args.network, args.data_folder, args.list_path, 'test', args.token_folder, word_vocab, args.max_length, args.allow_unk),
                batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    l_resize1 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    l_resize2 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    # Epochs
    test_start_time = time.time()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        for ind, batch_data in enumerate(
                tqdm(test_loader, desc='test_' + " EVALUATING AT BEAM SIZE " + str(1))):
            # Move to GPU, if available
            imgA = batch_data['imgA']
            imgB = batch_data['imgB']
            token_all = batch_data['token_all']
            token_all_len = batch_data['token_all_len']
            name = batch_data['name']
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            token_all = token_all.squeeze(0).cuda()
            # Forward prop.
            if encoder is not None:
                feat1, feat2 = encoder(imgA, imgB)
            feat = encoder_trans(feat1, feat2)
            seq = decoder.sample(feat, k=1)

            # for captioning
            except_tokens = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}
            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in except_tokens],
                        img_token))  # remove <start> and pads
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in except_tokens]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)
            # # 判断有没有变化
            pred_caption = ""
            ref_caption = ""
            for i in pred_seq:
                pred_caption += (list(word_vocab.keys())[i]) + " "
            ref_caption = ""
            for i in img_tokens[0]:
                ref_caption += (list(word_vocab.keys())[i]) + " "
            ref_captions = ""
            for i in img_tokens:
                for j in i:
                    ref_captions += (list(word_vocab.keys())[j]) + " "
                ref_captions += ".    "

            # save_captions(pred_caption, ref_captions, hypotheses[-1], references[-1], name, args.result_path)


        test_time = time.time() - test_start_time

        # Fast test during the training
        # Calculate evaluation scores
        score_dict = get_eval_score(references, hypotheses)
        Bleu_1 = score_dict['Bleu_1']
        Bleu_2 = score_dict['Bleu_2']
        Bleu_3 = score_dict['Bleu_3']
        Bleu_4 = score_dict['Bleu_4']
        Meteor = score_dict['METEOR']
        Rouge = score_dict['ROUGE_L']
        Cider = score_dict['CIDEr']
        print('Testing:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t'
              'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
              .format(test_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning')

    # Data parameters
    parser.add_argument('--sys', default='linux', choices=('linux'), help='system')
    parser.add_argument('--data_folder', default='/mnt/share_folder_c/lcy/dataset/Levir-CC-dataset/images',
                        help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=42, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC", help='base name shared by data files.')

    # Test
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default='', help='path to checkpoint, None if none.')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    parser.add_argument('--test_batchsize', default=1, help='batch_size for validation')
    parser.add_argument('--workers', type=int, default=8,
                        help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Validation
    parser.add_argument('--result_path', default="./predict_result/")

    # backbone parameters
    parser.add_argument('--decoder_type', default='transformer_decoder', help='mamba or gpt or transformer_decoder')
    parser.add_argument('--network', default='CLIP-ViT-B/32',help='define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=768, help='the dim of extracted features of backbone ')
    parser.add_argument('--feat_size', type=int, default=16, help='size of extracted features of backbone')
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=768, help='embedding dimension')
    args = parser.parse_args()
    if args.sys == 'linux':
        args.data_folder = '/data_ssd_4t/lcy/dataset/Levir-CC-dataset/images'
        if os.path.exists(args.data_folder) == False:
            args.data_folder = '/data/lcy/dataset/Levir-CC-dataset/images'
            if os.path.exists(args.data_folder) == False:
                args.data_folder = '/mnt/share_folder_c/lcy/dataset/Levir-CC-dataset/images'  # '/mnt/levir_datasets/LCY/Dataset/Levir-CC-dataset/images'
    print('list_path:', args.list_path)
    if args.network == 'CLIP-RN50':
        clip_emb_dim = 1024
        args.encoder_dim, args.feat_size = 2048, 7
    elif args.network == 'CLIP-RN101':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 2048, 7
    elif args.network == 'CLIP-RN50x4':
        clip_emb_dim = 640
        args.encoder_dim, args.feat_size = 2560, 9
    elif args.network == 'CLIP-RN50x16':
        clip_emb_dim = 768
        args.encoder_dim, args.feat_size = 3072, 12
    elif args.network == 'CLIP-ViT-B/16' or args.network == 'CLIP-ViT-L/16':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 768, 14
    elif args.network == 'CLIP-ViT-B/32' or args.network == 'CLIP-ViT-L/32':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 768, 7
    elif args.network == 'segformer-mit_b1':
        args.encoder_dim, args.feat_size = 512, 8

    args.embed_dim = args.encoder_dim
    main(args)
