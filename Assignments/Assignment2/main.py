# loading dataset
from datasets import load_dataset
tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
tiny_imagenet_test = load_dataset('Maysee/tiny-imagenet', split='test')



from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

inputs = image_processor(image, return_tensors="pt")

with torch.infre():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])