from flask import Flask, request, render_template, send_from_directory
from PIL import Image, ImageDraw
import os
import torch
from torchvision import models, transforms
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['OUTPUT_FOLDER'] = 'static/output/'

# Load the pretrained Mask R-CNN model
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transform for image preprocessing
transform = transforms.Compose([transforms.ToTensor()])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file uploaded!", 400

        file = request.files['file']
        if file.filename == '':
            return "No file selected!", 400

        # Save the uploaded file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Load and process the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        image_pil = image.copy()
        image_np = np.array(image)

        # Run the model
        with torch.no_grad():
            prediction = model(image_tensor)

        # Extract predictions
        masks = prediction[0]['masks']
        scores = prediction[0]['scores']

        # Filter masks by confidence
        confidence_threshold = 0.7
        high_confidence_indices = scores > confidence_threshold
        masks = masks[high_confidence_indices]

        # Overlay transparent masks
        for i in range(len(masks)):
            mask = masks[i, 0] > 0.5  # Convert the mask to binary
            mask = mask.byte().cpu().numpy()

            if mask.max() == 0:
                continue

            # Generate a random color
            random_color = np.random.randint(0, 256, size=3)

            # Create the mask overlay
            mask_overlay = np.zeros_like(image_np, dtype=np.uint8)
            for c in range(3):  # Apply the color to each channel
                mask_overlay[..., c] = random_color[c] * mask

            # Convert original image and overlay to RGBA
            original_rgba = image_pil.convert("RGBA")
            mask_overlay_rgba = Image.fromarray(mask_overlay).convert("RGBA")

            # Add transparency to the overlay
            alpha_layer = (mask * 128).astype(np.uint8)  # 50% transparency
            mask_overlay_rgba.putalpha(Image.fromarray(alpha_layer))

            # Composite the overlay on the original image
            image_pil = Image.alpha_composite(original_rgba, mask_overlay_rgba).convert("RGB")

        # Save the output image
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], file.filename)
        image_pil.save(output_path)

        return render_template("index.html", output_image=output_path)

    return render_template("index.html")

@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5009)
