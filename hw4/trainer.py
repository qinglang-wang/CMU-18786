import torch

def train_model(model, train_loader, val_loader, config):
    torch.backends.cudnn.benchmark = True

    device = config['device']
    epochs = config['epochs']

    model = model.to(device)
    criterion = config['criterion']()

    # Init optimizer
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.get('lr', 0.1),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 5e-4),
        )
    elif config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 0),
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    # Init scheduler
    scheduler = config.get('scheduler', None)
    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get('step_size', 30), gamma=0.1)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(X)

            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += X.size(0)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(X)

                loss = criterion(logits, y)

                val_loss += loss.item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += X.size(0)

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total
        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
              f"LR: {lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = config.get('save_path')
            if save_path:
                torch.save(model.state_dict(), save_path)

    # Summary
    sep = "=" * 60
    print(f"\n{sep}")
    print("TRAINING SUMMARY")
    print(sep)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"\nModel Architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {total_params:,} total, {trainable:,} trainable")
    print(f"\nHyper-parameters:")
    for k, v in config.items():
        if k not in {'device', 'save_path'}:
            print(f"  {k}: {v}")
    print(sep)

    return history