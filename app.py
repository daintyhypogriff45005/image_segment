from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the segmentation pipeline
semantic_segmentation = pipeline("image-segmentation", "facebook/mask2former-swin-large-cityscapes-panoptic")

@app.route('/')
def index():
    return render_template('index.html', selected_imgs=None, segmented_imgs=None)

@app.route('/segment', methods=['POST'])
def segment():
    # Check if image files were uploaded
    if 'file' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('file')

    # If no files are selected, redirect to the index page
    if not files:
        return redirect(request.url)

    selected_imgs = []
    segmented_imgs = []

    for file in files:
        # Read the image file
        img_bytes = file.read()

        # Open the image using PIL
        image = Image.open(io.BytesIO(img_bytes))

        # Perform segmentation
        results = semantic_segmentation(image)

        # Get the selected image
        selected_img_buffered = io.BytesIO()
        image.save(selected_img_buffered, format="PNG")
        selected_img_str = base64.b64encode(selected_img_buffered.getvalue()).decode('utf-8')
        selected_imgs.append(selected_img_str)

        # Get the segmented image
        segmented_img_buffered = io.BytesIO()
        results[-1]["mask"].save(segmented_img_buffered, format="PNG")
        segmented_img_str = base64.b64encode(segmented_img_buffered.getvalue()).decode('utf-8')
        segmented_imgs.append(segmented_img_str)

    # Render index.html with selected and segmented images
    return render_template('index.html', selected_imgs=selected_imgs, segmented_imgs=segmented_imgs)

if __name__ == '__main__':
    app.run(debug=True)
