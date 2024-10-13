import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import os

# ----------------------------
# 1. Base Few-Shot Learning Class
# ----------------------------

class FewShotModel:
    """
    Base class for Few-Shot Learning models. This class serves as a template for different few-shot methods.
    """
    def __init__(self, model_name, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.model_name = model_name

    def save_model(self, model, dataset_name, config_index, path="saved_models"):
        """
        Save the model with a name that includes the configuration index.
        """
        os.makedirs(path, exist_ok=True)
        model_file = f"{dataset_name}_{self.model_name}_configuration_{config_index}.pth"
        torch.save(model.state_dict(), os.path.join(path, model_file))
        print(f"Model saved as {model_file}")

# ----------------------------
# 2. Prototypical Networks
# ----------------------------
class ConvNet(nn.Module):
    """
    Configurable Convolutional Neural Network to extract image embeddings.
    """
    def __init__(self, embedding_dim=64, num_layers=3, num_filters=[64, 128, 256], dropout_rate=0.5, use_batchnorm=True):
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList()
        input_channels = 3  # RGB images
        
        # Dynamically build convolutional layers
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(input_channels, num_filters[i], kernel_size=3, padding=1))
            if use_batchnorm:
                self.layers.append(nn.BatchNorm2d(num_filters[i]))  # BatchNorm for faster convergence
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2, 2))  # Max Pooling
            input_channels = num_filters[i]
        
        self.fc = nn.Linear(num_filters[-1] * (244 // 2**num_layers)**2, embedding_dim)  # Adjust based on image size
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout for regularization

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)        # Apply dropout
        x = self.fc(x)             # Fully connected layer
        return x

class PrototypicalNet(nn.Module):
    """
    Prototypical Network for few-shot learning.
    The network computes embeddings for support and query sets.
    """
    def __init__(self, embedding_dim=64, num_layers=3, num_filters=[64, 128, 256], dropout_rate=0.5, use_batchnorm=True):
        super(PrototypicalNet, self).__init__()
        self.encoder = ConvNet(embedding_dim, num_layers, num_filters, dropout_rate, use_batchnorm)  # Use ConvNet to extract embeddings

    def forward(self, support, query):
        """
        Forward pass for support and query sets to compute embeddings.
        support: Support set images
        query: Query set images
        """
        support = support.view(-1, *support.size()[2:])  # Flatten to [batch_size * n_way * k_shot, C, H, W]
        query = query.view(-1, *query.size()[2:])        # Flatten to [batch_size * n_way * query_size, C, H, W]

        support_emb = self.encoder(support)  # [n_way * k_shot, embedding_dim]
        query_emb = self.encoder(query)      # [n_way * query_size, embedding_dim]
        
        return support_emb, query_emb


class PrototypicalNetModel(FewShotModel):
    """
    Prototypical Networks for Few-Shot Learning.
    """
    def __init__(self, embedding_dim=64, num_layers=3, num_filters=[64, 128, 256], dropout_rate=0.5, use_batchnorm=True):
        super().__init__("PrototypicalNet", embedding_dim)
        self.model = self.build_model(embedding_dim, num_layers, num_filters, dropout_rate, use_batchnorm)

    def build_model(self, embedding_dim, num_layers, num_filters, dropout_rate, use_batchnorm):
        return PrototypicalNet(embedding_dim, num_layers=num_layers, num_filters=num_filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)

    def train(self, dataset, subset_indices, device, n_way, k_shot, query_size, dataset_name, config_index, num_episodes=1000, learning_rate=0.001, batch_size=1):
        model = self.model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        loss_fn = nn.CrossEntropyLoss()

        trainer = Trainer(model, device, optimizer, loss_fn, n_way=n_way, k_shot=k_shot, query_size=query_size)
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        # Create DataLoader for Few-Shot Learning episodes
        data_loader = create_few_shot_dataloader(dataset, subset_indices, n_way, k_shot, query_size, batch_size=batch_size)

        print(f"Starting Prototypical Network Training for {dataset_name}...")

        # Initialize episode counter
        episode = 0
        while episode < num_episodes:
            # Loop over batches in the DataLoader
            for support, support_labels, query, query_labels in data_loader:
                episode += 1
                if scaler:
                    with torch.cuda.amp.autocast():
                        loss, acc = trainer.train_episode(support, support_labels, query, query_labels, scaler=scaler)
                else:
                    loss, acc = trainer.train_episode(support, support_labels, query, query_labels)

                # Print learning rate after each step
                print(f"Episode {episode}/{num_episodes} - Loss: {loss:.4f}, Accuracy: {acc * 100:.2f}%, Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

                # Adjust learning rate dynamically
                scheduler.step(loss)

                # Break if we've reached the target number of episodes
                if episode >= num_episodes:
                    break

        # Save the model after training
        self.save_model(model, dataset_name, config_index)


# ----------------------------
# 3. Trainer Class
# ----------------------------

class Trainer:
    """
    Trainer class to handle training and evaluation of the Prototypical Network.
    """
    def __init__(self, model, device, optimizer, loss_fn, n_way=2, k_shot=5, query_size=5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size

    def euclidean_dist(self, x, y):
        """
        Compute the Euclidean distance between two tensors.
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)  # [query_size, n_way]

    def train_episode(self, support, support_labels, query, query_labels, scaler=None):
        """
        Train on a single episode.
        """
        self.model.train()
        support, support_labels = support.to(self.device), support_labels.to(self.device)
        query, query_labels = query.to(self.device), query_labels.to(self.device)

        # Forward pass
        support_emb, query_emb = self.model(support, query)

        # Flatten support_labels to ensure correct shape for comparison
        support_labels = support_labels.view(-1)

        # Move torch.eye to the same device as support_labels
        support_labels_onehot = torch.eye(self.n_way, device=self.device)[support_labels]

        # Compute prototypes (vectorized)
        prototypes = support_emb.unsqueeze(1).mul(support_labels_onehot.unsqueeze(2)).sum(0).div(support_labels_onehot.sum(0).unsqueeze(1))

        # Compute distances between the query set embeddings and prototypes using cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)

        # Adjust query_labels to match the number of query samples
        query_labels = query_labels.view(-1)

        # Compute loss
        if scaler:
            loss = self.loss_fn(cosine_similarity, query_labels)
        else:
            loss = self.loss_fn(-cosine_similarity, query_labels)

        # Backward and optimize
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Compute accuracy
        _, predictions = torch.max(-cosine_similarity, 1)
        correct = (predictions == query_labels).sum().item()
        total = query_labels.size(0)
        accuracy = correct / total

        return loss.item(), accuracy


# ----------------------------
# 4. Few-Shot Dataset
# ----------------------------

class FewShotDataset(Dataset):
    """
    Custom Dataset for Few-Shot Learning that provides support and query sets in a batch-friendly format.
    """
    def __init__(self, full_dataset, subset_indices, n_way=2, k_shot=5, query_size=5):
        self.full_dataset = full_dataset  # This is the original ImageFolder dataset
        self.subset_indices = subset_indices  # Indices of the subset (either training or testing)
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.classes = self._get_classes()

    def _get_classes(self):
        """
        Retrieve the list of classes from the original dataset.
        """
        return list(self.full_dataset.class_to_idx.keys())

    def __len__(self):
        """
        The length of the dataset is the number of episodes (can be the number of possible combinations of classes and images).
        """
        return len(self.subset_indices)

    def __getitem__(self, idx):
        """
        Generate and return a single episode. An episode consists of:
        - Support set images
        - Support set labels
        - Query set images
        - Query set labels
        """
        selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for class_idx, cls in enumerate(selected_classes):
            cls_idx = self.full_dataset.class_to_idx[cls]
            # Get all the image indices that belong to the current class
            cls_images = [i for i in self.subset_indices if self.full_dataset.targets[i] == cls_idx]
            
            # Adjust k_shot and query_size based on the number of available images
            available_images = len(cls_images)
            if available_images < 2:
                raise ValueError(f"Not enough images for class '{cls}' (found {available_images}, need at least 2).")
            
            adjusted_k_shot = min(self.k_shot, available_images // 2)
            adjusted_query_size = available_images - adjusted_k_shot
            
            if adjusted_k_shot + adjusted_query_size > available_images:
                raise ValueError(f"Not enough images for class '{cls}' (found {available_images}, need {self.k_shot + self.query_size}).")
            
            # Randomly select support and query images for the current class
            selected_images = np.random.choice(cls_images, adjusted_k_shot + adjusted_query_size, replace=False)
            support_idxs = selected_images[:adjusted_k_shot]
            query_idxs = selected_images[adjusted_k_shot:]

            for si in support_idxs:
                support_images.append(self.full_dataset[si][0])
                support_labels.append(class_idx)  # Labels are 0 to n_way-1

            for qi in query_idxs:
                query_images.append(self.full_dataset[qi][0])
                query_labels.append(class_idx)

        # Convert lists to tensors
        support_images = torch.stack(support_images)  # [n_way * k_shot, C, H, W]
        support_labels = torch.tensor(support_labels)  # [n_way * k_shot]
        query_images = torch.stack(query_images)      # [n_way * query_size, C, H, W]
        query_labels = torch.tensor(query_labels)      # [n_way * query_size]

        return support_images, support_labels, query_images, query_labels



# ----------------------------
# 5. Dataset Loader and Dynamic Parameter Calculation
# ----------------------------

def create_few_shot_dataloader(full_dataset, subset_indices, n_way, k_shot, query_size, batch_size=1, shuffle=True, num_workers=4, pin_memory=True):
    """
    Create a DataLoader for Few-Shot Learning episodes.
    """
    few_shot_dataset = FewShotDataset(full_dataset, subset_indices, n_way=n_way, k_shot=k_shot, query_size=query_size)
    data_loader = torch.utils.data.DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader


def load_dataset(dataset_path, image_size=244):
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    class_counts = Counter(full_dataset.targets)
    print(f"Number of images per class after adding more pictures: {class_counts}")
    return full_dataset


def load_support_set(support_folder, image_size=244):
    """
    Load the support images and their corresponding labels from a folder.
    The folder should be organized with subdirectories for each class.
    """
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    support_dataset = datasets.ImageFolder(root=support_folder, transform=transform)
    return support_dataset

def calculate_dynamic_parameters(full_dataset):
    """
    Dynamically calculate the number of classes (n_way), k_shot, and query_size based on the dataset.
    """
    class_counts = {class_idx: 0 for class_idx in full_dataset.class_to_idx.values()}
    for _, label in full_dataset:
        class_counts[label] += 1
    
    n_way = len(class_counts)  # Number of classes
    k_shot = min(class_counts.values()) // 2  # Use half the images for k_shot
    query_size = min(class_counts.values()) - k_shot  # Use the remaining images for query

    return n_way, k_shot, query_size

# ----------------------------
# 6. Evaluation Function
# ----------------------------
def evaluate_model(model, dataset, test_indices, device, n_way, k_shot, query_size, batch_size=1):
    model.eval()  # Set the model to evaluation mode
    data_loader = create_few_shot_dataloader(dataset, test_indices, n_way, k_shot, query_size, batch_size=batch_size)
    total_loss = 0
    total_acc = 0
    count = 0

    # Set up the loss function
    loss_fn = nn.CrossEntropyLoss()

    # No optimizer required for evaluation
    with torch.no_grad():  # Disable gradient computation for evaluation
        for support, support_labels, query, query_labels in data_loader:
            support, support_labels = support.to(device), support_labels.to(device)
            query, query_labels = query.to(device), query_labels.to(device)

            # Forward pass
            support_emb, query_emb = model(support, query)

            # Compute prototypes
            support_labels = support_labels.view(-1)
            support_labels_onehot = torch.eye(n_way, device=device)[support_labels]
            prototypes = support_emb.unsqueeze(1).mul(support_labels_onehot.unsqueeze(2)).sum(0).div(support_labels_onehot.sum(0).unsqueeze(1))

            # Compute distances between query set embeddings and prototypes
            cosine_similarity = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)

            # Flatten cosine_similarity to match query_labels' shape
            cosine_similarity = cosine_similarity.view(-1, n_way)

            # Compute loss using CrossEntropyLoss
            loss = loss_fn(cosine_similarity, query_labels.view(-1))

            # Compute accuracy
            _, predictions = torch.max(cosine_similarity, 1)
            correct = (predictions == query_labels.view(-1)).sum().item()
            total = query_labels.size(0)
            accuracy = correct / total

            # Accumulate loss and accuracy for all batches
            total_loss += loss.item()
            total_acc += accuracy
            count += 1

    # Calculate average loss and accuracy over all batches
    avg_loss = total_loss / count
    avg_acc = total_acc / count

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc * 10:.2f}%")
    return avg_loss, avg_acc

# ----------------------------
# 7. Main Execution
# ----------------------------

def main(dataset_paths, support_folder):
    image_size = 244  # Set image size to 244
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"The device is {device}")
    num_train_episodes = 200
    learning_rate = 0.001
    batch_size = 4  # Adjust the batch size as needed

    configurations = [
        {"Configuration": 1, "num_layers": 3, "num_filters": [64, 128, 256], "dropout_rate": 0.5, "use_batchnorm": True},
        {"Configuration": 2, "num_layers": 3, "num_filters": [64, 128, 256], "dropout_rate": 0.5, "use_batchnorm": False},
        {"Configuration": 3, "num_layers": 4, "num_filters": [64, 128, 256, 512], "dropout_rate": 0.4, "use_batchnorm": True},
        {"Configuration": 4, "num_layers": 5, "num_filters": [32, 64, 128, 256, 512], "dropout_rate": 0.3, "use_batchnorm": True},

    ]

    # Load the support set
    support_dataset = load_support_set(support_folder, image_size=image_size)
    support_indices = np.arange(len(support_dataset))
    np.random.shuffle(support_indices)

    best_acc = 0  # To track the best accuracy
    best_model = None  # To store the model with the highest accuracy
    best_config_index = None  # To track the best configuration

    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        full_dataset = load_dataset(dataset_path, image_size=image_size)
        indices = np.arange(len(full_dataset))
        np.random.shuffle(indices)
        split = int(0.6 * len(full_dataset))  # 80% for training, 20% for testing
        train_indices, test_indices = indices[:split], indices[split:]

        # Dynamically calculate n_way, k_shot, and query_size for the query set (images folder)
        n_way, k_shot, query_size = calculate_dynamic_parameters(full_dataset)
        print(f"Dataset: {dataset_name}, n_way: {n_way}, k_shot: {k_shot}, query_size: {query_size}")

        for config in configurations:
            config_index = config["Configuration"]
            print(f"Training model with configuration {config_index}: {config}")

            # Initialize model with the given configuration parameters
            proto_model = PrototypicalNetModel(
                embedding_dim=64,
                num_layers=config["num_layers"],
                num_filters=config["num_filters"],
                dropout_rate=config["dropout_rate"],
                use_batchnorm=config["use_batchnorm"]
            )

            # Train the model using `train_indices` on the query set (from images folder)
            proto_model.train(
                full_dataset, train_indices, device, n_way=n_way, k_shot=k_shot, query_size=query_size,
                dataset_name=dataset_name, config_index=config_index, num_episodes=num_train_episodes,
                learning_rate=learning_rate, batch_size=batch_size
            )

            # Evaluate the trained model using the test set from query images (from images folder)
            print(f"Evaluating model with configuration {config_index}")
            _, accuracy = evaluate_model(proto_model.model, full_dataset, test_indices, device, n_way, k_shot, query_size, batch_size=batch_size)
        
            # Check if this model has the best accuracy
            if accuracy > best_acc:
                best_acc = accuracy
                best_model = proto_model.model
                best_config_index = config_index
                print(f"New best model found with accuracy {best_acc:.2f}%")

    # Save the best model
    if best_model is not None:
        print(f"Saving the best model with configuration {best_config_index} and accuracy {best_acc:.2f}%")
        best_model_file = f"{dataset_name}_best_PrototypicalNet_configuration_{best_config_index}.pth"
        torch.save(best_model.state_dict(), os.path.join("saved_models", best_model_file))
        print(f"Best model saved as {best_model_file}")


if __name__ == '__main__':
    dataset_paths = ["images"]  # Replace with actual paths for the query set
    support_folder = "support_images"  # Path to the support set folder
    main(dataset_paths, support_folder)
