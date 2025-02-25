from flask import Flask, request, jsonify
import os
from PIL import Image
import pytesseract
import cv2
import numpy as np

app = Flask(__name__)

def preprocess_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Convert PIL Image to OpenCV format for advanced processing
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image.astype(np.uint8)
    
    # 1. Resize to higher resolution for ultra-clear text
    scale_factor = 3  # Increase resolution (e.g., 3x)
    height, width = open_cv_image.shape
    new_width = width * scale_factor
    new_height = height * scale_factor
    resized_image = cv2.resize(open_cv_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # 2. Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)  # 5x5 kernel, sigma=0
    
    # 3. Enhance contrast (similar to normalize and linear adjustment)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted_image = clahe.apply(blurred_image)
    
    # 4. Apply median filter for strong noise reduction, preserving text edges
    noise_reduced_image = cv2.medianBlur(contrasted_image, 7)  # 7x7 median filter
    
    # 5. Apply threshold for binary (black-and-white) output
    _, binary_image = cv2.threshold(noise_reduced_image, 160, 255, cv2.THRESH_BINARY)  # High threshold for stark effect
    
    # 6. Apply edge detection to enhance text boundaries
    edges = cv2.Laplacian(binary_image, cv2.CV_64F, ksize=3)  # Laplacian edge detection
    edges = np.uint8(np.absolute(edges))
    _, edge_binary = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)  # Threshold edges for clarity
    
    # 7. Combine binary and edge-enhanced image for better text detection
    final_image = cv2.bitwise_or(binary_image, edge_binary)
    
    # 8. Sharpen the image for ultra-clear text
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpening kernel
    sharpened_image = cv2.filter2D(final_image, -1, kernel)
    
    # 9. Ensure maximum contrast for highlighted text
    contrasted_final = cv2.convertScaleAbs(sharpened_image, alpha=1.5, beta=-50)  # Boost contrast and darken text
    
    # 10. Re-threshold to ensure binary output
    _, final_binary = cv2.threshold(contrasted_final, 128, 255, cv2.THRESH_BINARY)  # Reapply threshold for clean binary
    
    # Convert back to PIL Image for OCR
    processed_pil = Image.fromarray(final_binary)
    
    return processed_pil

@app.route('/ocr', methods=['POST'])
def ocr():
    # Get the image path from the request
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image file does not exist'}), 400
    
    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        
        # Perform OCR
        text = pytesseract.image_to_string(preprocessed_image, lang='eng')
        
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)