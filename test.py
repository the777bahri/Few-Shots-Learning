import torch
from torchvision import transforms
from PIL import Image
import os

# Define the PrototypicalNet class and ConvNet architecture from your training script
class ConvNet(torch.nn.Module):
    def __init__(self, embedding_dim=64, num_layers=3, num_filters=[64, 128, 256], dropout_rate=0.5, use_batchnorm=True, input_image_size=244):
        super(ConvNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        input_channels = 3  # RGB images

        # Dynamically build convolutional layers based on the training architecture
        for i in range(num_layers):
            self.layers.append(torch.nn.Conv2d(input_channels, num_filters[i], kernel_size=3, padding=1))
            if use_batchnorm:
                self.layers.append(torch.nn.BatchNorm2d(num_filters[i]))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.MaxPool2d(2, 2))  # Max pooling reduces the size by half each time
            input_channels = num_filters[i]

        # Calculate the size of the output after the convolutional layers
        final_image_size = input_image_size // (2 ** num_layers)  # Divide by 2 for each pooling layer
        fc_input_size = num_filters[-1] * final_image_size * final_image_size  # Final flattened size

        # Dynamically adjust the fully connected layer to match the output size of conv layers
        self.fc = torch.nn.Linear(fc_input_size, embedding_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x


class PrototypicalNet(torch.nn.Module):
    def __init__(self, embedding_dim=64, num_layers=3, num_filters=[64, 128, 256], dropout_rate=0.5, use_batchnorm=True, input_image_size=244):
        super(PrototypicalNet, self).__init__()
        self.encoder = ConvNet(embedding_dim, num_layers, num_filters, dropout_rate, use_batchnorm, input_image_size)

    def forward(self, x):
        return self.encoder(x)


def load_model(model_path, embedding_dim=64, config=None, device=None):
    # Automatically detect GPU or fallback to CPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}")

    if config is None:
        raise ValueError("Configuration (config) must be provided!")

    # Extract configuration parameters from the config dictionary
    num_layers = config['num_layers']
    num_filters = config['num_filters']
    dropout_rate = config['dropout_rate']
    use_batchnorm = config['use_batchnorm']

    # Load the trained model using the extracted configuration
    model = PrototypicalNet(embedding_dim, num_layers=num_layers, num_filters=num_filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_image(image_path, image_size=244, device=None):
    # Automatically detect GPU or fallback to CPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Apply the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')  # Ensure the image is RGB
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device


def compute_prototypes(model, support_images, support_labels, n_way):
    """
    Compute the prototypes for each class based on the support images.
    """
    model.eval()
    with torch.no_grad():
        support_embeddings = model(support_images)  # Compute embeddings for support images
        prototypes = []
        for i in range(n_way):
            class_emb = support_embeddings[support_labels == i]  # Get embeddings for class i
            prototype = class_emb.mean(0)  # Compute the mean of embeddings (prototype)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)  # [n_way, embedding_dim]
    return prototypes


def classify_images(model, image_folder, support_images, support_labels, n_way, class_names, device=None):
    """
    Classify images into predefined classes without using a distance threshold.
    """
    # Compute the prototypes from the support set
    class_prototypes = compute_prototypes(model, support_images, support_labels, n_way)

    # Load and sort test images by their filename
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'))]

    if len(image_paths) == 0:
        print(f"No images found in {image_folder} with the specified formats.")
        return {}

    # Dictionary to store the predicted classes
    predicted_classes = {}

    # Iterate over images in the folder
    for image_path in image_paths:
        image = preprocess_image(image_path, device=device)  # Preprocess image

        # Get embedding for the test image
        with torch.no_grad():
            image_embedding = model(image)

        # Compare image embedding with class prototypes
        distances = torch.cdist(image_embedding, class_prototypes)  # Compute distances to each prototype
        min_distance, predicted_class_idx = torch.min(distances, dim=1)  # Find the smallest distance
        predicted_class = class_names[predicted_class_idx.item()]

        # Strip the file extension and store the predicted class
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Store the predicted class in the dictionary
        predicted_classes[image_name] = predicted_class
        print(f"\nImage {image_name} is classified as {predicted_class}")

    return predicted_classes


def load_support_set(support_folder, device=None):
    # Load the support images and their corresponding labels from a folder.
    support_images = []
    support_labels = []

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a transformation (resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((244, 244)),  # Fixed to match training size
        transforms.ToTensor(),
    ])

    # Get the list of class subdirectories (e.g., 'bird', 'butterfly', 'unknown')
    class_names = os.listdir(support_folder)  # Dynamically get class names

    # Loop through each class directory and load the images
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(support_folder, class_name)
        if os.path.isdir(class_dir):  # Ensure it's a directory
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if any(image_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']):
                    image = Image.open(image_path).convert('RGB')
                    image = transform(image).to(device)
                    support_images.append(image)
                    support_labels.append(idx)  # Class index is based on folder order

    support_images = torch.stack(support_images)  # Stack images into a tensor
    support_labels = torch.tensor(support_labels).to(device)  # Convert labels to tensor

    return support_images, support_labels, class_names  # Return class names as well
 
def model_evaluation(predicted_classes):
    # Ground truth classes for test images
    actual_classes = {
        "test_1": "bird",
        "test_2": "bird",
        "test_3": "bird",
        "test_4": "unknown",
        "test_5": "butterfly",
        "test_6": "butterfly",
        "test_7": "bird",
        "test_8": "bird",
        "test_9": "bird",
        "test_10": "bird",
        "test_11": "butterfly",
        "test_12": "butterfly",
        "test_13": "butterfly",
        "test_14": "unknown",
        "test_15": "unknown",
        "test_16": "unknown",
        "test_17": "bird",
        "test_18": "butterfly",
        "test_19": "butterfly",
        "test_20": "unknown"
    }

    # Count correct predictions
    correct_predictions = 0
    total_predictions = len(actual_classes)

    # Print to check for mismatches
    for image_name, actual_class in actual_classes.items():
        predicted_class = predicted_classes.get(image_name, "missing")
        print(f"\nImage: {image_name}, Predicted: {predicted_class}, Actual: {actual_class}")

        if predicted_class == actual_class:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions * 100
    print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
    return accuracy

def main():
    # Configuration
    model_folder = 'saved_models'  # Path where all the .pth models are saved
    test_folder = 'test_images'  # Replace with the actual path to the test folder
    support_folder = 'support_images'  # Replace with your support set folder path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configuration list (must match your training configurations)
    configurations = [
        {"Configuration": 1, "num_layers": 3, "num_filters": [64, 128, 256], "dropout_rate": 0.5, "use_batchnorm": True},
        {"Configuration": 2, "num_layers": 3, "num_filters": [64, 128, 256], "dropout_rate": 0.5, "use_batchnorm": False},
        {"Configuration": 3, "num_layers": 4, "num_filters": [64, 128, 256, 512], "dropout_rate": 0.4, "use_batchnorm": True},
        {"Configuration": 4, "num_layers": 5, "num_filters": [32, 64, 128, 256, 512], "dropout_rate": 0.3, "use_batchnorm": True},
    ]

    # Load the support images, labels, and class names dynamically from folder structure
    support_images, support_labels, class_names = load_support_set(support_folder, device=device)

    # Get all models from the model folder
    model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.pth')]

    # Loop through each model, evaluate it, and print the accuracy
    for model_path in model_paths:
        # Extract the configuration index from the model name
        config_index = int(model_path.split('_configuration_')[1].split('.pth')[0])
        config = configurations[config_index - 1]  # Get the corresponding configuration

        print(f"\nEvaluating model: {model_path}")
        model = load_model(model_path, embedding_dim=64, config=config, device=device)

        # Classify the images in the test folder using dynamically computed prototypes
        predicted_classes = classify_images(model, test_folder, support_images, support_labels, n_way=len(class_names), class_names=class_names, device=device)

        # Evaluate accuracy
        accuracy = model_evaluation(predicted_classes)
        print(f"Model {model_path} Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
