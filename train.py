
import argparse

# ARGUMENTS CLI
parser = argparse.ArgumentParser()
parser.add_argument('--framework', type=str, required=True,
                    choices=['pytorch', 'tensorflow'],
                    help="Choose framework")
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_dir', type=str, default='data')

args = parser.parse_args()


# PYTORCH TRAINING
if args.framework == "pytorch":

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from models.model_pytorch import IntelCNNPyTorch
    from utils.data_loader import get_data_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader, classes = get_data_loaders(
        args.data_dir, args.batch_size)

    model = IntelCNNPyTorch().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TRAIN LOOP
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {running_loss:.4f} Accuracy: {acc:.2f}%")

    # SAVE MODEL
    torch.save(model.state_dict(), "saved_models/ahmed_model.pth")
    print("Model saved as ahmed_model.pth")


# TENSORFLOW TRAINING
elif args.framework == "tensorflow":

    import tensorflow as tf

    from models.model_tensorflow import create_intel_cnn_tf
    from utils.data_loader_tf import get_datasets

    train_dataset, test_dataset = get_datasets(args.data_dir)

    model = create_intel_cnn_tf()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=args.epochs
    )

    # SAVE MODEL
    model.save("saved_models/ahmed_model.keras")
    print("Model saved as ahmed_model.keras")
    
