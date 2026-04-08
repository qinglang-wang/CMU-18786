import torch

from base import BaseGANTrainer, create_base_parser
from model_variants import HierarchicalLatentGenerator, MultiScaleDiscriminator


class HierarchicalLatentGANTrainer(BaseGANTrainer):
    """Custom GAN with hierarchical latent injection and a multi-scale discriminator."""

    variant_name = 'hierarchical_latent'

    def create_model(self):
        G = HierarchicalLatentGenerator(
            noise_size=self.opts.noise_size,
            conv_dim=self.opts.conv_dim
        )
        D = MultiScaleDiscriminator(conv_dim=self.opts.conv_dim)
        return G, D

    def train_step(self, G, D, real_images, g_optimizer, d_optimizer):
        real_scores = D(real_images)
        D_real_loss = 0.5 * torch.nn.functional.mse_loss(
            real_scores, torch.ones_like(real_scores)
        )

        noise = self.sample_noise(real_images.size(0))
        fake_images = G(noise)
        fake_scores = D(fake_images.detach())
        D_fake_loss = 0.5 * torch.nn.functional.mse_loss(
            fake_scores, torch.zeros_like(fake_scores)
        )
        D_total_loss = D_real_loss + D_fake_loss

        d_optimizer.zero_grad()
        D_total_loss.backward()
        d_optimizer.step()

        noise = self.sample_noise(real_images.size(0))
        fake_images = G(noise)
        fake_scores = D(fake_images)
        G_loss = 0.5 * torch.nn.functional.mse_loss(
            fake_scores, torch.ones_like(fake_scores)
        )

        g_optimizer.zero_grad()
        G_loss.backward()
        g_optimizer.step()

        return {
            'd_real': D_real_loss,
            'd_fake': D_fake_loss,
            'd_total': D_total_loss,
            'g_total': G_loss,
        }


def main():
    parser = create_base_parser()
    opts = parser.parse_args()
    trainer = HierarchicalLatentGANTrainer(opts)
    trainer.run()


if __name__ == '__main__':
    main()
