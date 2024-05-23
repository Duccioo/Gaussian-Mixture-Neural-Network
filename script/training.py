import torch
import torch.nn.functional as F


# ---
from utils.metrics import calculate_metrics


def validation(model, X_train, Y_train, X_val, Y_val, device, loss_type):

    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    validation_loss: float = 0
    training_loss: float = 0
    metrics_val: dict = {}
    metrics_train: dict = {}

    model.eval()
    with torch.no_grad():
        output_train = model(X_train.to(device))
        training_loss = loss_type(output_train, Y_train.to(device)).item()
        metrics_train = calculate_metrics(
            Y_train.numpy(), output_train.detach().cpu().numpy(), 5
        )
        if X_val.size(0) > 0:
            output_val = model(X_val.to(device))
            validation_loss = loss_type(output_val, Y_val.to(device)).item()
            metrics_val = calculate_metrics(
                Y_val.clone().detach().cpu().numpy(),
                output_val.detach().cpu().numpy(),
                5,
            )

    return training_loss, validation_loss, metrics_train, metrics_val


def training(
    model,
    X_train,
    Y_train,
    X_val,
    Y_val,
    lr,
    epochs,
    batch_size,
    optimizer_name,
    criterion,
    device,
):

    train_loss_list = list()
    val_loss_list = list()
    train_metrics_list = list()
    val_metrics_list = list()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    xy_train = torch.cat((X_train, Y_train), 1)

    train_loader = torch.utils.data.DataLoader(
        xy_train,
        batch_size=batch_size,
        shuffle=True,
    )

    model.to(device)

    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    loss = getattr(F, criterion)

    # Training of the model.
    for epoch in range(epochs):
        model.train()
        for batch_idx, train_data in enumerate(train_loader):
            data = train_data[:, 0]
            target = train_data[:, 1]
            # Limiting training data for faster epochs.
            data, target = data.view(data.size(0), 1).to(device), target.view(
                target.size(0), 1
            ).to(device)

            optimizer.zero_grad()
            output = model(data)

            loss_value = loss(output, target)
            loss_value.backward()
            optimizer.step()

        # finita un epocha faccio il validation test:
        if len(X_val) > 0:
            train_loss_epoch, val_loss_epoch, train_metric_epoch, val_metric_epoch = (
                validation(model, X_train, Y_train, X_val, Y_val, device, loss)
            )

            train_loss_list.append(train_loss_epoch)
            val_loss_list.append(val_loss_epoch)
            train_metrics_list.append(train_metric_epoch)
            val_metrics_list.append(val_metric_epoch)
    return train_loss_list, val_loss_list, train_metrics_list, val_metrics_list


def evaluation(model, test_X, test_Y, device):
    model.eval()
    with torch.no_grad():
        pdf_predicted = model(torch.tensor(test_X, dtype=torch.float32).to(device))
        pdf_predicted = pdf_predicted.detach().cpu().numpy()
        metrics = calculate_metrics(test_Y, pdf_predicted)

    return metrics, pdf_predicted
