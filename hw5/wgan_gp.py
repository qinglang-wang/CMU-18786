import torch

from model_variants import DCDiscriminator, DCGenerator
from base import BaseGANTrainer, create_base_parser


class WGANGPGANTrainer(BaseGANTrainer):
    """Train a Wasserstein GAN with gradient penalty."""

    variant_name = 'wgan_gp'

    def resolve_training_hparams(self):
        if self.opts.lr is None:
            self.opts.lr = 0.0001
        if self.opts.beta1 is None:
            self.opts.beta1 = 0.0
        if self.opts.beta2 is None:
            self.opts.beta2 = 0.9

    def create_model(self):
        G = DCGenerator(noise_size=self.opts.noise_size, conv_dim=self.opts.conv_dim)
        D = DCDiscriminator(conv_dim=self.opts.conv_dim)
        return G, D

    def compute_gradient_penalty(self, D, real_images, fake_images):
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_images.device)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)

        scores = D(interpolated)
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def train_step(self, G, D, real_images, g_optimizer, d_optimizer):
        for _ in range(self.opts.critic_updates):
            noise = self.sample_noise(real_images.size(0))
            fake_images = G(noise)
            real_scores = D(real_images)
            fake_scores = D(fake_images.detach())
            gradient_penalty = self.compute_gradient_penalty(D, real_images, fake_images.detach())
            D_total_loss = fake_scores.mean() - real_scores.mean() + self.opts.gp_lambda * gradient_penalty

            d_optimizer.zero_grad()
            D_total_loss.backward()
            d_optimizer.step()

        noise = self.sample_noise(real_images.size(0))
        fake_images = G(noise)
        G_loss = -D(fake_images).mean()

        g_optimizer.zero_grad()
        G_loss.backward()
        g_optimizer.step()

        return {
            'd_real': real_scores.mean(),
            'd_fake': fake_scores.mean(),
            'd_total': D_total_loss,
            'g_total': G_loss,
            'gp': gradient_penalty,
        }

    def format_log_message(self, iteration, total_train_iters, stats):
        return (
            f'Iteration [{iteration:4d}/{total_train_iters:4d}] | '
            f'D_real: {stats["d_real"].item():6.4f} | '
            f'D_fake: {stats["d_fake"].item():6.4f} | '
            f'GP: {stats["gp"].item():6.4f} | '
            f'G_loss: {stats["g_total"].item():6.4f}'
        )

    def log_scalars(self, iteration, stats):
        super().log_scalars(iteration, stats)
        self.logger.add_scalar('D/gp', stats['gp'], iteration)


def main():
    parser = create_base_parser()
    opts = parser.parse_args()
    trainer = WGANGPGANTrainer(opts)
    trainer.run()

if __name__ == '__main__':
    main()
