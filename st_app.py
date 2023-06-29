import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# Load the trained model for Devanagari character recognition
model = tf.keras.models.load_model('trained_model.h5')

# Define the labels for the predicted characters
# Creating class labels for 58 classes according to the dataset

# added by prachi 
word_dict = {
    0: 'क',
    1: 'ख',
    2: 'ग',
    3: 'घ',
    4: 'ङ',
    5: 'च',
    6: 'छ',
    7: 'ज',
    8: 'झ',
    9: 'ञ',
    10: 'ट',
    11: 'ठ',
    12: 'ड',
    13: 'ढ',
    14: 'ण',
    15: 'त',
    16: 'थ',
    17: 'द',
    18: 'ध',
    19: 'न',
    20: 'प',
    21: 'फ',
    22: 'ब',
    23: 'भ',
    24: 'म',
    25: 'य',
    26: 'र',
    27: 'ल',
    28: 'व',
    29: 'श',
    30: 'ष',
    31: 'स',
    32: 'ह',
    33: 'क्ष',
    34: 'त्र',
    35: 'ज्ञ',
    36: '0',
    37: '१',
    38: '२',
    39: '३',
    40: '४',
    41: '५',
    42: '६',
    43: '७',
    44: '८',
    45: '९',
    46: 'क्त',
    47: 'क्ष्म',
    48: 'ल्ल',
    49:  'न्य',
    50: 'स्था',
    51: 'ष्ट्र',
    52: 'व्ह',
    53: 'विना',
    54: '३ल',
    55: '५य',
    56: 'रुद्र',
    57: '१२%',
}

# Function to preprocess the input image
def preprocess_image_display(image):
    # Apply adaptive thresholding for better image clarity
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # above added extra 

    # Normalize the pixel values
    normalized_image = thresholded_image / 255.0
    # Resize the image to the required shape
    resized_image = cv2.resize(normalized_image, (32, 32))
    # Convert the image to a numpy array and add an extra dimension
    preprocessed_image_display = np.expand_dims(resized_image, axis=-1)
    return preprocessed_image_display


def preprocess_image(image):

    # Normalize the pixel values
    normalized_image = image / 255.0
    # Resize the image to the required shape
    resized_image = cv2.resize(normalized_image, (32, 32))
    # Convert the image to a numpy array and add an extra dimension
    preprocessed_image = np.expand_dims(resized_image, axis=-1)
    return preprocessed_image

# Streamlit app
def main():
    st.title("Devanagari Handwritten Joint Character Recognition")

    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    col1, col2, col3 = st.columns(3)

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=None, width=150, channels="RGB")

        # Preprocess the image
        preprocessed_image_display = preprocess_image_display(image)

        preprocessed_image = preprocess_image(image)
        # Preprocessed image display
        if st.button("Preprocess Image"):
             with col2:
                st.image(preprocessed_image_display, caption="Preprocessed Image", use_column_width=None, width=150, channels="RGB")


        # Predict the character on button click
        if st.button("Predict"):
            # Make the prediction
            prediction = model.predict(np.array([preprocessed_image]))
            predicted_class_index = np.argmax(prediction)
            predicted_character = word_dict[predicted_class_index]
            
            # Get the prediction accuracy
            prediction_accuracy = np.max(prediction)

            # Display the predicted character
            st.subheader("Predicted Character:")
            st.markdown(f"<h2 style='text-align: center;'>{predicted_character}</h2>", unsafe_allow_html=True)
            st.subheader("Prediction Accuracy:")
            st.write(f"{prediction_accuracy:.2%}")

        # Clean up preprocessed image

    if st.button("Clear"):
        st.empty()
        col2.empty()


if __name__ == "__main__":
    main()
