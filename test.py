import requests

# Define the API endpoint and image path
url = 'http://127.0.0.1:5000/ocr'
image_path = '/Users/bs1155/Desktop/projects/newOcrPy/images/first.jpeg' 

# Send a POST request to the API
response = requests.post(url, json={'image_path': image_path})

# Print the response
if response.status_code == 200:
    print('Extracted Text:', response.json().get('text'))
else:
    print('Error:', response.json().get('error'))