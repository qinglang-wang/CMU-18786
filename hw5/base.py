import argparse
import os

import imageio
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from data_loader import get_data_loader


SEED = 11

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def create_base_parser():
    """Creates a parser with the shared Part 2 options."""
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--beta1', type=float, default=None)
    parser.add_argument('--beta2', type=float, default=None)
    parser.add_argument('--critic_updates', type=int, default=5)
    parser.add_argument('--gp_lambda', type=float, default=10.0)

    # Data sources
    parser.add_argument('--data', type=str, default='cat/grumpifyBprocessed')
    parser.add_argument('--data_preprocess', type=str, default='advanced')
    parser.add_argument('--ext', type=str, default='*.png')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', default='./checkpoints_part2')
    parser.add_argument('--sample_dir', type=str, default='./part2')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--checkpoint_every', type=int, default=400)

    return parser


class BaseGANTrainer:
    """Shared training template for Part 2 variants."""

    variant_name = None

    def __init__(self, opts):
        if self.variant_name is None:
            raise ValueError('variant_name must be set on subclasses.')

        self.opts = opts
        self.logger = None
        self.resolve_training_hparams()
        self.prepare_output_paths()

    def resolve_training_hparams(self):
        """Assign default training hyper-parameters for the variant."""
        if self.opts.lr is None:
            self.opts.lr = 0.0002
        if self.opts.beta1 is None:
            self.opts.beta1 = 0.5
        if self.opts.beta2 is None:
            self.opts.beta2 = 0.999

    def prepare_output_paths(self):
        """Attach variant-specific suffixes to checkpoint and sample paths."""
        variant_tag = (
            f'{self.variant_name}_{os.path.basename(self.opts.data)}_'
            f'{self.opts.data_preprocess}'
        )
        self.opts.sample_dir = os.path.join('output/', self.opts.sample_dir, variant_tag)
        self.opts.checkpoint_dir = os.path.join(self.opts.checkpoint_dir, variant_tag)

    def create_model(self):
        """Create and return (G, D). Subclasses must implement this."""
        raise NotImplementedError

    def train_step(self, G, D, real_images, g_optimizer, d_optimizer):
        """Perform one training step and return a stats dict."""
        raise NotImplementedError

    def format_log_message(self, iteration, total_train_iters, stats):
        """Format the default training log message."""
        return (
            f'Iteration [{iteration:4d}/{total_train_iters:4d}] | '
            f'D_real_loss: {stats["d_real"].item():6.4f} | '
            f'D_fake_loss: {stats["d_fake"].item():6.4f} | '
            f'G_loss: {stats["g_total"].item():6.4f}'
        )

    def log_scalars(self, iteration, stats):
        """Write the default scalar summaries."""
        self.logger.add_scalar('D/real', stats['d_real'], iteration)
        self.logger.add_scalar('D/fake', stats['d_fake'], iteration)
        self.logger.add_scalar('D/total', stats['d_total'], iteration)
        self.logger.add_scalar('G/total', stats['g_total'], iteration)

    def print_models(self, G, D):
        """Print model information for the generator and discriminator."""
        print("                    G                  ")
        print("---------------------------------------")
        print(G)
        print("---------------------------------------")

        print("                    D                  ")
        print("---------------------------------------")
        print(D)
        print("---------------------------------------")

    def create_optimizers(self, G, D):
        """Create optimizers for the generator and discriminator."""
        g_optimizer = optim.Adam(G.parameters(), self.opts.lr, [self.opts.beta1, self.opts.beta2])
        d_optimizer = optim.Adam(D.parameters(), self.opts.lr, [self.opts.beta1, self.opts.beta2])
        return g_optimizer, d_optimizer

    def create_image_grid(self, array, ncols=None):
        """Arrange generated samples into a square-ish image grid."""
        num_images, channels, cell_h, cell_w = array.shape

        if not ncols:
            ncols = int(np.sqrt(num_images))
        nrows = num_images // ncols
        result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
        for i in range(0, nrows):
            for j in range(0, ncols):
                result[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w, :] = array[i * ncols + j].transpose(1, 2, 0)

        if channels == 1:
            result = result.squeeze()
        return result

    def checkpoint(self, iteration, G, D):
        """Save the parameters of the generator and discriminator."""
        G_path = os.path.join(self.opts.checkpoint_dir, f'G_iter{iteration}.pkl')
        D_path = os.path.join(self.opts.checkpoint_dir, f'D_iter{iteration}.pkl')
        torch.save(G.state_dict(), G_path)
        torch.save(D.state_dict(), D_path)

    def save_samples(self, G, fixed_noise, iteration):
        """Save generated samples for the current training iteration."""
        generated_images = G(fixed_noise)
        generated_images = utils.to_data(generated_images)

        grid = self.create_image_grid(generated_images)
        grid = np.uint8(255 * (grid + 1) / 2)

        path = os.path.join(self.opts.sample_dir, f'sample-{iteration:06d}.png')
        imageio.imwrite(path, grid)
        print(f'Saved {path}')

    def save_images(self, images, iteration, name):
        """Save a grid of images under a descriptive name."""
        grid = self.create_image_grid(utils.to_data(images))

        path = os.path.join(self.opts.sample_dir, f'{name}-{iteration:06d}.png')
        grid = np.uint8(255 * (grid + 1) / 2)
        imageio.imwrite(path, grid)
        print(f'Saved {path}')

    def sample_noise(self, batch_size):
        """Generate a noise tensor of shape (batch_size, dim, 1, 1)."""
        return utils.to_var(torch.rand(batch_size, self.opts.noise_size) * 2 - 1).unsqueeze(2).unsqueeze(3)

    def prepare_run(self):
        """Create output directories and the TensorBoard logger."""
        utils.create_dir(self.opts.checkpoint_dir)
        utils.create_dir(self.opts.sample_dir)
        self.logger = SummaryWriter(self.opts.sample_dir)

    def training_loop(self, train_dataloader):
        """Runs the training loop for the selected variant."""
        G, D = self.create_model()
        self.print_models(G, D)

        if torch.cuda.is_available():
            G.cuda()
            D.cuda()
            print('Models moved to GPU.')

        g_optimizer, d_optimizer = self.create_optimizers(G, D)
        fixed_noise = self.sample_noise(self.opts.batch_size)

        iteration = 1
        total_train_iters = self.opts.num_epochs * len(train_dataloader)

        for _ in range(self.opts.num_epochs):
            for batch in train_dataloader:
                real_images = utils.to_var(batch)
                stats = self.train_step(G, D, real_images, g_optimizer, d_optimizer)

                if iteration % self.opts.log_step == 0:
                    print(self.format_log_message(iteration, total_train_iters, stats))
                    self.log_scalars(iteration, stats)

                if iteration % self.opts.sample_every == 0:
                    self.save_samples(G, fixed_noise, iteration)
                    self.save_images(real_images, iteration, 'real')

                if iteration % self.opts.checkpoint_every == 0:
                    self.checkpoint(iteration, G, D)

                iteration += 1

    def run(self):
        """Load data, prepare output directories, and start training."""
        self.prepare_run()
        print(self.opts)
        dataloader = get_data_loader(self.opts.data, self.opts)

        try:
            self.training_loop(dataloader)
        finally:
            if self.logger is not None:
                self.logger.close()
