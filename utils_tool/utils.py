import os
import torch
import numpy as np
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
import time


def save_checkpoint(args, data_name, epoch, encoder, encoder_feat, decoder, encoder_optimizer,
                encoder_feat_optimizer, decoder_optimizer, best_bleu4):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'best_bleu-4': best_bleu4,
             'encoder': encoder,
             'encoder_feat': encoder_feat,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'encoder_feat_optimizer': encoder_feat_optimizer,
             'decoder_optimizer': decoder_optimizer,
             }
    #filename = 'checkpoint_' + data_name + '_' + args.network + '.pth.tar'
    path = args.savepath #'./models_checkpoint/mymodel/3-times/'
    if os.path.exists(path)==False:
        os.makedirs(path)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    torch.save(state, os.path.join(path, 'BEST_' + data_name))

    # torch.save(state, os.path.join(path, 'checkpoint_' + data_name +'_epoch_'+str(epoch) + '.pth.tar'))


def accuracy_v0(masked_scores, masked_targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = masked_targets.size(0)
    _, ind = masked_scores.topk(k, 1, True, True)
    correct = ind.eq(masked_targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
def accuracy(output, target, target_mask=None, topk=(1, 5)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        total_num = target_mask.sum().item()

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        try:
            correct = correct * target_mask.view(1, -1).expand_as(pred)
        except:
            print('eerrrpo')
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=False)
            res.append(correct_k.mul_(100.0 / total_num))
        return res


def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        #print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
                
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_file_str():
    ISOTIMEFORMAT='%Y-%m-%d-%H-%M-%S'
    string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string #+ '-{}'.format(random.randint(1, 10000))

def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


if __name__ == '__main__':

    # nochange_list = [[1,2,3,4],[1,2,3,5],[6,2,7],[6,8,7],[3,4,8,9],[3,4,2,10],[3,5,11]]
    # % nochange_list = ["there is no change", "there is no difference", "the two scenes seem identical",
    #                    "the scene is the same as before", "the scene remains the same", "nothing is changed",
    #                    "nothing has changed", "no change has occurred", "no change is made", "no difference exists"]
    references = [ [[1,2,'ball',4],[1,2,3,5],[6,8,7,8],[3,4,8,9],[3,4,2,10]],
                   # [[1, 2, 3, 4], [1, 2, 3, 5], [6, 8, 7, 8], [3, 4, 8, 9], [3, 4, 2, 10]],
                   # [[1, 2, 3, 4], [1, 2, 3, 5], [6, 8, 7, 8], [3, 4, 8, 9], [3, 4, 2, 10]]
                   ]
    #references：两个图片对应的参考标注描述语句（本例子中假设每个图片有5个标注语句）
    hypotheses = [ [1,2,'balls',4],
                   # [88, 28, 38, 48],[88, 28, 38, 48]
                    ]

    # references = [
    #     [[176, 14, 481, 62, 6, 7, 37, 38, 127, 24, 16, 88, 517, 127, 114, 18, 128, 16, 280, 554, 14, 224, 114, 517],
    #      [88, 496, 174, 3, 48, 49, 33, 20, 117, 222, 14, 88, 496, 212, 525, 100, 7, 274], [88, 496, 212, 10, 42, 1256, 18, 7, 227, 253],
    #      [7, 142, 76, 300, 14, 528, 532, 100, 88, 511], [14, 88, 496, 212, 525, 100, 7, 274]],
    #     ]
    # references = [
    #     [[88, 496, 174, 3, 48, 49, 33, 20, 117,888, 222, 14, 88, 496, 212, 525, 100, 7, 274],[49, 33, 20, 117,888, 222, 14, 88, 496, 212, 525, 100, 7, 274],[18, 496, 174, 3, 48, 49, 33, 20, 117,888, 222, 14, 88, 496, 212, 525, 100, 7, 274],[18, 496, 174, 3, 48, 49, 33, 20, 117,888, 222, 14, 88, 496, 212, 525, 100, 7, 274],[18, 496, 174, 3, 48, 49, 33, 20, 117,888, 222, 14, 88, 496, 212, 525, 100, 7, 274]]
    # ]
    # # references：两个图片对应的参考标注描述语句（本例子中假设每个图片有5个标注语句）
    # hypotheses = [[88, 496, 174, 3, 48, 49, 33, 20, 117,888, 222, 14, 88, 496, 212, 525, 100, 7, 274],
    #               ]

    #hypotheses：模型在测试时，两个测试图片对应分别输出的2个描述语句
    metrics = get_eval_score(references+references+references, hypotheses+hypotheses+hypotheses)
    print(metrics)


