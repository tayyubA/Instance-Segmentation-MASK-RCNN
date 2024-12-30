import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  

transform = transforms.Compose([transforms.ToTensor()])

# Load the image (replace with your image path)
image_path = '000000000080.jpg'  
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  
image_np = np.array(image)
image_pil = image.copy()
draw = ImageDraw.Draw(image_pil)


with torch.no_grad():
    prediction = model(image_tensor)

masks = prediction[0]['masks']  
boxes = prediction[0]['boxes']  
labels = prediction[0]['labels']  
scores = prediction[0]['scores']  


confidence_threshold = 0.9
high_confidence_indices = scores > confidence_threshold
masks = masks[high_confidence_indices]
boxes = boxes[high_confidence_indices]
labels = labels[high_confidence_indices]



font = ImageFont.load_default()


for i in range(len(masks)):
    mask = masks[i, 0] > 0.5  
    mask = mask.byte().cpu().numpy()  


    if mask.max() == 0:
        continue

    mask_overlay = np.zeros_like(image_np, dtype=np.uint8)
    mask_overlay[mask == 1] = np.random.randint(0, 256, size=3)  
    mask_overlay = Image.fromarray(mask_overlay)

    image_pil.paste(mask_overlay, (0, 0), mask=Image.fromarray(mask * 255))

    xmin, ymin, xmax, ymax = boxes[i].cpu().numpy()
    draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=3)

    label = f'{labels[i].item()} ({scores[i].item():.2f})'

    text_bbox = draw.textbbox((xmin, ymin), label, font=font)
    label_width = text_bbox[2] - text_bbox[0]  
    label_height = text_bbox[3] - text_bbox[1]  

    draw.rectangle([xmin, ymin - label_height, xmin + label_width, ymin], fill="blue")

    draw.text((xmin, ymin - label_height), label, fill="white", font=font)

output_path = 'output_image.png'
image_pil.save(output_path)
image_pil.show()
