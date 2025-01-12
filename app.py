import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image



def main():
    image_ban = Image.open('images/image1.png')
    st.image(image_ban, use_container_width=False)


if __name__ == '__main__':
    main()

# Load the trained model with custom objects
model_path = './icp_prediction_model2.h5'  # Update with the correct path

# Define custom objects for loading the model (e.g., MSE loss function)
custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError()  # Specify the mean squared error loss function
}

# Load the model with the custom objects
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)


# Preprocessing function
def preprocess_image(image):
    """
    Preprocess the input image to match the model's expected input format.
    Resizes the image to 128x128 and normalizes pixel values.
    """
    image = image.resize((128, 128))  # Resize image to 128x128
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    if image_array.shape[-1] != 3:
        raise ValueError("Input image must have 3 color channels (RGB).")
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


# Prediction function-cmH20
def predict_icp(image):
    """
    Predict the ICP value given an uploaded CT image.
    """
    try:
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_icp = prediction[0][0]  # Get the scalar value from the prediction
        return f"Predicted ICP Value: {predicted_icp:.2f} cmH2O"
    except Exception as e:
        return f"Error: {str(e)}"

# Prediction function-mmHg
def predict_icp2(image):
    """
    Predict the ICP value given an uploaded CT image and convert it to mmHg.
    """
    try:
        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(image)

        # Get the prediction (in cmH₂O)
        prediction = model.predict(preprocessed_image)
        predicted_icp_cmH2O = prediction[0][0]  # Get the scalar value from the prediction

        # Convert to mmHg
        predicted_icp_mmHg = predicted_icp_cmH2O * 0.736  # Convert cmH₂O to mmHg

        # Return the predicted ICP value in mmHg
        return f"Predicted ICP Value: {predicted_icp_mmHg:.2f} mmHg"

    except Exception as e:
        return f"Error: {str(e)}"


# Streamlit Interface
st.title("Intracranial Pressure (ICP) Prediction")
st.write("""
Upload a CT scan image, and the trained model will predict the Intracranial Pressure (ICP) value.
Ensure the image is in RGB format and clearly represents the CT scan for accurate predictions.
""")

# Image uploader
uploaded_image = st.file_uploader("Upload a CT scan image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Make prediction when the image is uploaded
    prediction = predict_icp(image)
    prediction2 = predict_icp2(image)
    # Displaying the prediction using st.markdown with HTML formatting
    #st.write(prediction)
    st.title(prediction)
    st.title(prediction2)

