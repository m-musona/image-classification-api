# Image Classification Web App

This is a simple image classification web application built using **FastAPI** and **ResNet-50**, a powerful pre-trained convolutional neural network. The application provides an interface for users to upload images and receive classification results.

## Features

- **FastAPI** for web framework and API endpoints.
- **ResNet-50** for robust image classification.
- Supports image uploads and returns predictions in a user-friendly format.

---

## Project Structure

├── classify.py # Contains model initialization, preprocessing, and classification logic 
├── main.py # Contains FastAPI endpoints and app launch code 
└── README.md # Project documentation


### Files Overview

1. **`classify.py`**
   - **Functions**:
     - `init()`: Initializes the ResNet-50 model and image transformation pipeline.
     - `preprocess_image(image_path)`: Opens the uploaded image and applies transformations.
     - `classify_image(image_path)`: Uses the model to classify the image and returns the result in a Python dictionary.

2. **`main.py`**
   - **Functions**:
     - `home()`: Renders the web application's homepage using the **GET** method.
     - `upload_image()`: Handles image uploads using the **POST** method and redirects to the homepage after processing.

---

## Setup and Installation

### Prerequisites
- Python 3.8 or above
- `pip` for Python package management

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install required dependencies:
    ```
    pip install fastapi uvicorn torch torchvision pillow
    ```

3. Run the application:
    ```
    python main.py
    ```

4. Open the app in your browser at:
    ```
    http://127.0.0.1:8000
    ```

## Usage

### Homepage (GET /)

- The homepage provides a user-friendly interface to upload images for classification.

### Upload Image (POST /upload_image)

- Upload an image via the interface or directly via an API client.
- The app will process the image, classify it using ResNet-50, and display the results.

## Code Highlights

### Initializing the App
In `main.py`, the application is launched with:

```
if __name__ == "__main__":
    import uvicorn
    init()
    uvicorn.run(app)
```

## Model Functions in `classify.py`

- init(): Loads ResNet-50 and defines the necessary image transformations.
- preprocess_image(image_path): Prepares the image for classification using the initialized transformations.
- classify_image(image_path): Uses the model to predict the image class and returns a dictionary with results.

## Future Enhancements
- Enable multi-image upload functionality.
- Integrate Docker for containerized deployment.

## Acknowledgements
- FastAPI: For a fast and modern web framework.
- ResNet-50: For robust and accurate image classification.





