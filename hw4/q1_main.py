import torch
from data import get_cifar100_loaders
from models import FCNN, CNN
from trainer import train_model
from utils import set_seed, plot_loss, plot_accuracy, visualize_predictions

torch.set_float32_matmul_precision('high')

def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = get_cifar100_loaders(batch_size=512, augment=False)

    # Fix 5 random indices for visualization
    vis_indices = torch.randint(0, len(val_loader.dataset), (5,)).tolist()

    print("Experiment 1: FCNN")
    fcnn = FCNN(input_dim=3*32*32, hidden_dims=[1024, 512, 256], num_classes=100, dropout=0.4)
    fcnn_config = dict(
        epochs=50,
        lr=4e-3,
        weight_decay=1e-4,
        scheduler='cosine',
        device=device,
        save_path='results/q1_fcnn_best.pth',
    )
    fcnn_history = train_model(fcnn, train_loader, val_loader, fcnn_config)
    plot_loss(fcnn_history, title='FCNN', save_to='results/q1_fcnn_loss.png')
    plot_accuracy(fcnn_history, title='FCNN', save_to='results/q1_fcnn_acc.png')
    visualize_predictions(fcnn, val_loader.dataset, vis_indices, title='FCNN Predictions', save_to='results/q1_fcnn_preds.png', device=device)

    print("Experiment 2: CNN")
    cnn = CNN(num_classes=100)
    cnn_config = dict(
        epochs=10,
        lr=3e-3,
        weight_decay=1e-4,
        scheduler='cosine',
        device=device,
        save_path='results/q1_cnn_best.pth',
    )
    cnn_history = train_model(cnn, train_loader, val_loader, cnn_config)
    plot_loss(cnn_history, title='CNN', save_to='results/q1_cnn_loss.png')
    plot_accuracy(cnn_history, title='CNN', save_to='results/q1_cnn_acc.png')
    visualize_predictions(cnn, val_loader.dataset, vis_indices, title='CNN Predictions', save_to='results/q1_cnn_preds.png', device=device)

if __name__ == '__main__':
    main()
