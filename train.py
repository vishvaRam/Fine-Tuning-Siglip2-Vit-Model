import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler
import evaluate
from datasets import Dataset, Image, ClassLabel, load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
    DefaultDataCollator,
    EarlyStoppingCallback
)
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Resize,
    RandomRotation,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
    ColorJitter,
    RandomAffine,
    RandomGrayscale,
    RandomCrop,
    RandomResizedCrop,
)

from PIL import Image, ExifTags
from PIL import Image as PILImage
from PIL import ImageFile
# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load your dataset
dataset_path = "./dataset"  # Replace with the actual path to your 'dataset' folder
dataset = load_dataset("imagefolder", data_dir=dataset_path)
dataset = dataset['train'].train_test_split(test_size=0.2, shuffle=True, stratify_by_column="label")

train_data = dataset['train']
test_data = dataset['test']

print(train_data)
print(test_data)

# Get the class names from the dataset
labels_list = train_data.features['label'].names
print("Labels:", labels_list)

label2id, id2label = {}, {}
for i, label in enumerate(labels_list):
    label2id[label] = i
    id2label[i] = label

print("Mapping of IDs to Labels:", id2label, '\n')
print("Mapping of Labels to IDs:", label2id)

from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification

# Use AutoImageProcessor instead of AutoProcessor
model_str = "google/siglip2-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_str)

# Extract preprocessing parameters
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# Define training transformations
_train_transforms = Compose([
    Resize((size, size)),
    RandomResizedCrop(size=(size, size), scale=(0.8, 1.0), ratio=(0.75, 1.333)), # Randomly crop and resize
    RandomRotation(degrees=20),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Adjust color properties
    RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)), # Apply affine transformations
    RandomGrayscale(p=0.1),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])

# Define validation transformations
_val_transforms = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])

# Apply transformations to dataset
def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

train_data.set_transform(train_transforms)
test_data.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

model = SiglipForImageClassification.from_pretrained(model_str, num_labels=len(labels_list), id2label=id2label, label2id=label2id)

print(model.num_parameters(only_trainable=True) / 1e6)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']

    return {
        "accuracy": acc_score
    }

args = TrainingArguments(
    output_dir="siglip2-person-looking",
    logging_dir='./logs-person-looking',
    eval_strategy="epoch",
    learning_rate=5e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10, 
    weight_decay=0.01,
    warmup_steps=100,
    remove_unused_columns=False,
    save_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.evaluate()
trainer.train()
trainer.evaluate()
outputs = trainer.predict(test_data)

print(outputs.metrics)

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")  # Save the plot as a PNG file
    plt.close() # Close the plot to free memory

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

print()
print("Classification report:")
print()
print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

trainer.save_model("siglip2-person-looking-finetuned")


# {'test_loss': 0.32023242115974426, 'test_model_preparation_time': 0.0012, 'test_accuracy': 0.875, 'test_runtime': 0.5798, 'test_samples_per_second': 68.993, 'test_steps_per_second': 5.175}
# Accuracy: 0.8750
# F1 Score: 0.8749

# Classification report:

#                                    precision    recall  f1-score   support

#    a person looking at the camera     0.8947    0.8500    0.8718        20
# a person looking away from camera     0.8571    0.9000    0.8780        20

#                          accuracy                         0.8750        40
#                         macro avg     0.8759    0.8750    0.8749        40
#                      weighted avg     0.8759    0.8750    0.8749        40
