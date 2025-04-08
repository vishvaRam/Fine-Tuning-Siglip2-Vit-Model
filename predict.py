from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load the finetuned model and processor
model_path = "./siglip2-person-looking-finetuned"
processor = AutoImageProcessor.from_pretrained(model_path)
model = SiglipForImageClassification.from_pretrained(model_path)

# Define the class labels (obtained from your training script)
labels_list = ['a person looking at the camera', 'a person looking away from camera']

def predict_image(image_path):
    """
    Predicts the class of an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The predicted class label.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return f"Error: Image not found at {image_path}"
    except Exception as e:
        return f"Error loading image: {e}"

    # Prepare the image for the model
    inputs = processor(images=image, return_tensors="pt")

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class index
    predicted_class_idx = torch.argmax(logits, dim=-1).item()

    # Get the predicted class label
    predicted_label = labels_list[predicted_class_idx]

    return predicted_label

if __name__ == "__main__":
    image_to_predict = "./Images/away.jpg"  
    prediction = predict_image(image_to_predict)
    print(f"Prediction for {image_to_predict}: {prediction}")

    image_to_predict_2 = "./Images/look.jpg" 
    prediction_2 = predict_image(image_to_predict_2)
    print(f"Prediction for {image_to_predict_2}: {prediction_2}")

    image_to_predict_3 = "./Images/img2.jpg" 
    prediction_3 = predict_image(image_to_predict_3)
    print(f"Prediction for {image_to_predict_3}: {prediction_3}")
