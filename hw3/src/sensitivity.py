import random
import argparse

import dataset
import models
import trainer
import utils

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main():
    random.seed(0)

    argp = argparse.ArgumentParser()
    argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
    argp.add_argument('variant', help="Choose vanilla or rope")
    argp.add_argument('pretrain_corpus_path', default=None)
    argp.add_argument('--reading_params_path',default=None)
    argp.add_argument('--writing_params_path',default=None)
    argp.add_argument('--finetune_corpus_path', default=None)
    argp.add_argument('--eval_corpus_path', default=None)
    argp.add_argument('--outputs_path', default=None)
    argp.add_argument('--pretrain_lr', default=6e-3, type=float)
    argp.add_argument('--finetune_lr', default=6e-4, type=float)
    argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                    default='run')
    args = argp.parse_args()

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    elif torch.backends.mps.is_available() and args.variant == 'vanilla':
        device = 'mps'

    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/%s/%s_%s_pt_lr_%f_ft_lr_%f' % (
        args.function,
        args.tb_expt_name,
        args.variant,
        args.pretrain_lr,
        args.finetune_lr))

    # Keep the block size 128
    # Why is the pretraining corpus always required (even if we're not pretraining?)
    # It's because we're using it as a hack to always have the same vocabulary
    # (that is, the same mapping from character to integer, and we build the
    # vocab from the pretraining corpus.)
    block_size = 128
    text = open(args.pretrain_corpus_path, encoding='utf-8').read()
    pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

    # We don't suggest you change these hyperparameters, as they're known to work.
    # use them for both the vanilla and the RoPE models
    mconf = models.GPTConfig(
        pretrain_dataset.vocab_size,
        pretrain_dataset.block_size,
        n_layer=4,
        n_head=8,
        n_embd=256)

    # define models.
    # note: models should moved to device defined on lines 30-34.

    model = None
    if args.variant == 'vanilla':
        # TODO: [part c] Make some model here
        ### YOUR CODE HERE ###
        model = models.GPT(mconf).to(device)
        ### END YOUR CODE ###
    elif args.variant == 'rope':
        # TODO: [part g] Make some other model here
        # set mconf.rope parameter
        ### YOUR CODE HERE ###
        mconf.rope = True
        model = models.GPT(mconf).to(device)
        ### END YOUR CODE ###
    else:
        raise ValueError("Unknown model variant")

    print('Model on device: ', next(model.parameters()).device)

    # Perform evaluation
    assert args.function == 'evaluate'
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
            x_orig = line.split('\t')[0]
            x = f"What is the birthplace of {x_orig[10:-6]}?" + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print(f'Correct: {correct} out of {total}: {correct/total*100}%')
    else:
        print(f'Predictions written to {args.outputs_path}; no targets provided')


if __name__ == '__main__':
    main()
