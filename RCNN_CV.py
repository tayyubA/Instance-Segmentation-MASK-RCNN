import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

image_path = '000000000080.jpg'  
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  

image_np = np.array(image)
image_pil = image.copy()

with torch.no_grad():
    prediction = model(image_tensor)

masks = prediction[0]['masks']
scores = prediction[0]['scores']

confidence_threshold = 0.7  
high_confidence_indices = scores > confidence_threshold
masks = masks[high_confidence_indices]

for i in range(len(masks)):
    mask = masks[i, 0] > 0.5  
    mask = mask.byte().cpu().numpy()

    if mask.max() == 0:
        continue

    random_color = np.random.randint(0, 256, size=3)

    mask_overlay = np.zeros_like(image_np, dtype=np.uint8)
    for c in range(3):  
        mask_overlay[..., c] = random_color[c] * mask

    original_rgba = image_pil.convert("RGBA")
    mask_overlay_rgba = Image.fromarray(mask_overlay).convert("RGBA")

    alpha_layer = (mask * 128).astype(np.uint8)  
    mask_overlay_rgba.putalpha(Image.fromarray(alpha_layer))

    image_pil = Image.alpha_composite(original_rgba, mask_overlay_rgba).convert("RGB")

output_path = 'output_image.png'
image_pil.convert("RGB").save(output_path)
image_pil.show()
