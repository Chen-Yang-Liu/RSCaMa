from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json, random
from tqdm import tqdm
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from model.model_encoder_attMamba import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
class Trainer(object):
    def __init__(self, args):
        """
        Training and validation.
        """
        self.args = args
        random_str = str(random.randint(1, 10000))
        name = args.decoder_type+f'layers{args.decoder_n_layers}'+ time_file_str() +'_' +random_str
        self.args.savepath = os.path.join(args.savepath, name)
        if os.path.exists(self.args.savepath)==False:
            os.makedirs(self.args.savepath)
        self.log = open(os.path.join(self.args.savepath, '{}.log'.format(name)), 'w')
        print_log('=>datset: {}'.format(args.data_name), self.log)
        print_log('=>network: {}'.format(args.network), self.log)
        print_log('=>encoder_lr: {}'.format(args.encoder_lr), self.log)
        print_log('=>decoder_lr: {}'.format(args.decoder_lr), self.log)
        print_log('=>num_epochs: {}'.format(args.num_epochs), self.log)
        print_log('=>train_batchsize: {}'.format(args.train_batchsize), self.log)

        self.best_bleu4 = 0.4  # BLEU-4 score right now
        self.start_epoch = 0
        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)
        # Initialize / load checkpoint
        self.build_model()

        # Loss function
        self.criterion_cap = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_cap_cls = torch.nn.CrossEntropyLoss().cuda()

        # Custom dataloaders
        if args.data_name == 'LEVIR_CC':
            self.train_loader = data.DataLoader(
                LEVIRCCDataset(args.network,args.data_folder, args.list_path, 'train', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                LEVIRCCDataset(args.network, args.data_folder, args.list_path, 'test', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

        self.l_resizeA = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
        self.l_resizeB = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
        self.index_i = 0
        self.hist = np.zeros((args.num_epochs*2 * len(self.train_loader), 5))
        # Epochs

        self.best_model_path = None
        self.best_epoch = 0

    def build_model(self):
        args = self.args
        # Initialize / load checkpoint
        self.encoder = Encoder(args.network)
        self.encoder.fine_tune(args.fine_tune_encoder)
        self.encoder_trans = AttentiveEncoder(n_layers=args.n_layers, feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                              heads=args.n_heads, dropout=args.dropout)
        self.decoder = DecoderTransformer(decoder_type=args.decoder_type,
                                          embed_dim=args.embed_dim,
                                          vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                                          word_vocab=self.word_vocab, n_head=args.n_heads,
                                          n_layers=args.decoder_n_layers, dropout=args.dropout)

        # set optimizer
        self.encoder_optimizer = torch.optim.Adam(params=self.encoder.parameters(),
                                                  lr=args.encoder_lr) if args.fine_tune_encoder else None
        self.encoder_trans_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
            lr=args.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=args.decoder_lr)

        # Move to GPU, if available
        self.encoder = self.encoder.cuda()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder = self.decoder.cuda()
        self.encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=5,
                                                                    gamma=1.0) if args.fine_tune_encoder else None
        self.encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_trans_optimizer, step_size=5,
                                                                          gamma=1.0)
        self.decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=5,
                                                                    gamma=1.0)

    def training(self, args, epoch):
        self.encoder.train()
        self.encoder_trans.train()
        self.decoder.train()  # train mode (dropout and batchnorm is used)

        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad()
        self.encoder_trans_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        for id, batch_data in enumerate(self.train_loader):
            # if id == 10:
            #    break
            start_time = time.time()
            accum_steps = 64//args.train_batchsize

            # Move to GPU, if available
            imgA = batch_data['imgA']
            imgB = batch_data['imgB']
            token = batch_data['token']
            token_len = batch_data['token_len']
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            token = token.cuda()
            token_len = token_len.cuda()
            # Forward prop.
            feat1, feat2 = self.encoder(imgA, imgB)
            feat = self.encoder_trans(feat1, feat2)
            scores, caps_sorted, decode_lengths, sort_ind = self.decoder(feat, token, token_len)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            # Calculate loss
            loss = self.criterion_cap(scores, targets.to(torch.int64))

            # Back prop.
            loss = loss / accum_steps
            loss.backward()
            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(self.decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(self.encoder_trans.parameters(), args.grad_clip)
                if self.encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.encoder.parameters(), args.grad_clip)

            # Update weights
            if (id + 1) % accum_steps == 0 or (id + 1) == len(self.train_loader):
                self.decoder_optimizer.step()
                self.encoder_trans_optimizer.step()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.step()

                # Adjust learning rate
                self.decoder_lr_scheduler.step()
                self.encoder_trans_lr_scheduler.step()
                if self.encoder_lr_scheduler is not None:
                    self.encoder_lr_scheduler.step()

                self.decoder_optimizer.zero_grad()
                self.encoder_trans_optimizer.zero_grad()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()

            # Keep track of metrics
            self.hist[self.index_i, 0] = time.time() - start_time #batch_time
            self.hist[self.index_i, 1] = loss.item()  # train_loss
            self.hist[self.index_i, 2] = accuracy_v0(scores, targets, 5) #top5

            self.index_i += 1
            # Print status
            if self.index_i % args.print_freq == 0:
                print_log('Training Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Cap_loss: {4:.5f}\t'
                    'Text_Top-5 Acc: {5:.3f}'
                    .format(epoch, id, len(self.train_loader),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,0])*args.print_freq,
                                         np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,1]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,2])
                                ), self.log)

    # One epoch's validation
    def validation(self, epoch):
        word_vocab = self.word_vocab
        self.decoder.eval()  # eval mode (no dropout or batchnorm)
        self.encoder_trans.eval()
        if self.encoder is not None:
            self.encoder.eval()

        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        with torch.no_grad():
            # Batches
            for ind, batch_data in enumerate(
                    tqdm(self.val_loader, desc='val_' + "EVALUATING AT BEAM SIZE " + str(1))):
                # if ind == 20:
                #     break
                # Move to GPU, if available
                # (imgA, imgB, token_all, token_all_len, _, _, _)
                imgA = batch_data['imgA']
                imgB = batch_data['imgB']
                token_all = batch_data['token_all']
                token_all_len = batch_data['token_all_len']
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                token_all = token_all.squeeze(0).cuda()
                # Forward prop.
                if self.encoder is not None:
                    feat1, feat2 = self.encoder(imgA, imgB)
                feat = self.encoder_trans(feat1, feat2)
                seq = self.decoder.sample(feat, k=1)

                # for captioning
                except_tokens = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}
                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in except_tokens],
                        img_token))  # remove <start> and pads
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in except_tokens]
                hypotheses.append(pred_seq)
                assert len(references) == len(hypotheses)

                if ind % self.args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_seq:
                        pred_caption += (list(word_vocab.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_vocab.keys())[j]) + " "
                        ref_caption += ".    "
            val_time = time.time() - val_start_time
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
            print_log('Captioning_Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t' 
                'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
                .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider), self.log)

        # Check if there was an improvement
        if Bleu_4 > self.best_bleu4:
            self.best_bleu4 = max(Bleu_4, self.best_bleu4)
            # save_checkpoint
            print('Save Model')
            state = {'encoder_dict': self.encoder.state_dict(),
                     'encoder_trans_dict': self.encoder_trans.state_dict(),
                     'decoder_dict': self.decoder.state_dict()
                     }
            metric = f'Bleu4_{round(100000 * self.best_bleu4)}'
            model_name = f'{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}.pth'
            if epoch > 4:
                torch.save(state, os.path.join(self.args.savepath, model_name.replace('/','-')))
            # save a txt file
            text_path = os.path.join(self.args.savepath, model_name.replace('/','-'))
            with open(text_path.replace('.pth', '.txt'), 'w') as f:
                f.write('Bleu_1: ' + str(Bleu_1) + '\t')
                f.write('Bleu_2: ' + str(Bleu_2) + '\t')
                f.write('Bleu_3: ' + str(Bleu_3) + '\t')
                f.write('Bleu_4: ' + str(Bleu_4) + '\t')
                f.write('Meteor: ' + str(Meteor) + '\t')
                f.write('Rouge: ' + str(Rouge) + '\t')
                f.write('Cider: ' + str(Cider) + '\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    # Data parameters
    parser.add_argument('--sys', default='win', choices=('linux'), help='system')
    parser.add_argument('--data_folder', default='/mnt/share_folder_c/lcy/dataset/Levir-CC-dataset/images',help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC_v3/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC_v3/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=42, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC",help='base name shared by data files.')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    # Training parameters
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=80, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=16, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--decoder_type', default='transformer_decoder', help='mamba or gpt or transformer_decoder')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_ckpt/")
    # backbone parameters
    parser.add_argument('--network', default='CLIP-ViT-B/32', help=' define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=768, help='the dim of extracted features of backbone ')
    parser.add_argument('--feat_size', type=int, default=16, help='size of extracted features of backbone')
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=768, help='embedding dimension')
    args = parser.parse_args()


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

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.args.num_epochs)

    for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
        trainer.training(trainer.args, epoch)
        # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        trainer.validation(epoch)

