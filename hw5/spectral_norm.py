import torch

from model_variants import DCGenerator, SNDCDiscriminator
from base import BaseGANTrainer, create_base_parser


class SpectralNormGANTrainer(BaseGANTrainer):
    """Train a DCGAN with spectral normalization in the discriminator."""

    variant_name = 'spectral_norm'

    def create_model(self):
        G = DCGenerator(noise_size=self.opts.noise_size, conv_dim=self.opts.conv_dim)
        D = SNDCDiscriminator(conv_dim=self.opts.conv_dim)
        return G, D

    def train_step(self, G, D, real_images, g_optimizer, d_optimizer):
        real_logits = D(real_images)
        D_real_loss = torch.nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))

        noise = self.sample_noise(real_images.size(0))
        fake_images = G(noise)
        fake_logits = D(fake_images.detach())
        D_fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        D_total_loss = D_real_loss + D_fake_loss

        d_optimizer.zero_grad()
        D_total_loss.backward()
        d_optimizer.step()

        noise = self.sample_noise(real_images.size(0))
        fake_images = G(noise)
        fake_logits = D(fake_images)
        G_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

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
    trainer = SpectralNormGANTrainer(opts)
    trainer.run()

if __name__ == '__main__':
    main()
