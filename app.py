import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras

# Load pre-trained models
vgg_model = load_model('vgg_model.h5')
resnet_model = load_model('res50.h5')

# Define image dimensions
image_shape = (224, 224, 3)  

# Prepare test image paths
test_normal_dir = 'test/NORMAL/'
test_pneumonia_dir = 'test/PNEUMONIA/'

test_normal_images = [os.path.join(test_normal_dir, fname) for fname in os.listdir(test_normal_dir) if fname.endswith('.jpeg')]
test_pneumonia_images = [os.path.join(test_pneumonia_dir, fname) for fname in os.listdir(test_pneumonia_dir) if fname.endswith('.jpeg')]

# Create a list of tuples with display names and full paths
test_images_list = [('Normal: ' + os.path.basename(path), path) for path in test_normal_images] + [('Pneumonia: ' + os.path.basename(path), path) for path in test_pneumonia_images]

# Streamlit app title
st.title('Pneumonia Image Classification Dashboard with Grad-CAM and Saliency Map')

# User input selections
selected_option = st.selectbox('Select an image', [option[0] for option in test_images_list])
model_selection = st.selectbox('Select a Model', ('VGG16', 'ResNet50'))

# Get the full path of the selected image
selected_image_path = next(path for label, path in test_images_list if label == selected_option)

# Preprocess the selected image
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(image_shape[0], image_shape[1]), color_mode='rgb')
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, image

# Grad-CAM function
def generate_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = keras.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # Compute the gradient of the top predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_class = tf.argmax(predictions[0])  # Index of the top prediction
        loss = tf.reduce_mean(predictions[:, top_class])
        top_class_grad = tape.gradient(loss, conv_outputs)

    # Compute the channel-wise mean of gradients
    weights = tf.reduce_mean(top_class_grad, axis=(0, 1, 2))
    gradcam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)

    # Weight each channel by corresponding gradient
    for i, w in enumerate(weights):
        gradcam += w * conv_outputs[0, :, :, i]
    
    # Apply ReLU
    gradcam = np.maximum(gradcam, 0)

    # Normalize the heatmap
    gradcam = gradcam / np.max(gradcam)
    return gradcam

# Overlay Grad-CAM on original image
def overlay_gradcam_on_image(image, heatmap, alpha=0.4):
    # Resize heatmap to match the image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay_image = cv2.addWeighted(heatmap, alpha, np.array(image), 1 - alpha, 0)
    return overlay_image

# Saliency Map function
def generate_saliency_map(model, img_array):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        top_class = tf.argmax(predictions[0])
        loss = predictions[0, top_class]
    
    # Compute gradients of the loss with respect to the input image
    gradients = tape.gradient(loss, img_tensor)
    
    # Take the maximum across the color channels
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)
    
    # Normalize the saliency map
    saliency_map = (saliency_map - tf.reduce_min(saliency_map)) / (tf.reduce_max(saliency_map) - tf.reduce_min(saliency_map))
    return saliency_map[0].numpy()

# Load and preprocess the image
img_array, image = load_and_preprocess_image(selected_image_path)

# Select the model based on user choice
if model_selection == 'VGG16':
    model = vgg_model
    last_conv_layer_name = 'block5_conv3'  
elif model_selection == 'ResNet50':
    model = resnet_model
    last_conv_layer_name = 'conv5_block3_out'  

# Display the image
st.image(image, caption='Selected X-ray Image', use_container_width=True)

# Display the prediction probability
prediction = model.predict(img_array)
probability = prediction[0][0] 

st.write('Model:' + model_selection)
# Classify based on probability threshold (e.g., 0.5)
if probability > 0.5:
    st.write('Prediction: Pneumonia')
else:
    st.write('Prediction: Normal')

# Generate Grad-CAM heatmap
heatmap = generate_gradcam_heatmap(model, img_array, last_conv_layer_name)

# Overlay the heatmap on the original image
heatmap_overlay = overlay_gradcam_on_image(np.array(image), heatmap)

# Display Grad-CAM heatmap
st.image(heatmap_overlay, caption='Grad-CAM Heatmap', use_container_width=True)

# Generate Saliency Map
saliency_map = generate_saliency_map(model, img_array)

# Overlay the saliency map on the original image
saliency_overlay = overlay_gradcam_on_image(np.array(image), saliency_map)

# Display Saliency Map
st.image(saliency_map, caption='Saliency Map', use_container_width=True)
st.image(saliency_overlay, caption='Saliency Overlay', use_container_width=True)