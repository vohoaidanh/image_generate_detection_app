import os
import sys

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)

sys.path.append(parent)

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import resnet50, npr

# Load the model
#model = resnet50(num_classes=1)
#model.load_state_dict(torch.load('./weights/ADOF_model_epoch_9.pth', map_location='cpu'), strict=True)
#model.eval()

# Define labels
labels = [
    "Real",
    "Generated"
]

def resize_image(image_path):
    # Mở ảnh từ đường dẫn
    img = Image.open(image_path)
    
    # Lấy kích thước ban đầu của ảnh
    w, h = img.size
    
    # Tính toán tỷ lệ thay đổi kích thước
    if w < h:
        new_w = 256
        new_h = int((256 / w) * h)
    else:
        new_h = 256
        new_w = int((256 / h) * w)
    
    # Thay đổi kích thước hình ảnh giữ tỷ lệ
    img_resized = img.resize((new_w, new_h))
    
    return img_resized

preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224, 224)),  # Thay đổi kích thước
    transforms.ToTensor(),           # Chuyển đổi sang tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
])

# Define the sample images
sample_images = {
    "cat": "./test_images/cat.png",
    "dog": "./test_images/dog.png",
}

# Define the sample images
model_paths = {
    "ADOF_1": "./weights/ADOF_model_epoch_9.pth",
    "ADOF_2": "./weights/adof_2.pth",
    "NPR": "./weights/NPR.pth",
}
model = None

# Define the function to make predictions on an image
def predict(model, image):
    if model is None:
        return []
    try:
        image = preprocess(image).unsqueeze(0)

        # Prediction
        # Make a prediction on the image
        #predictions = []

        with torch.no_grad():
            output = model(image).sigmoid().flatten()
            #print('output is ', output)
            #print(100*'_')
            probabilities = output[0].numpy() #torch.nn.functional.softmax(output[0])
            label_pred = int((probabilities > 0.5)*1.0)
            if label_pred==0:
                probabilities = 1 - probabilities
            return [probabilities, label_pred]
    except Exception as e:
        print(f"Error predicting image: {e}")
        return []

if 'model' not in st.session_state:
    st.session_state.model = None  # Khởi tạo trạng thái mô hình ban đầu là None

# Define the Streamlit app
def app():
    
    st.title("Synthetic Image Detection")
    col1, col2 = st.columns(2)
    file_name = ''
    
# Tạo một ô trống để hiển thị kết quả
    with col1:
            # Add a file uploader
        uploaded_file = st.file_uploader("Upload an image (<=5Mb)", type=["jpg", "jpeg", "png"])
        
    # # Add a selectbox to choose from sample images
        model_name = st.selectbox("Select model:", list(model_paths.keys()))
        
    # Thêm nút để thực hiện dự đoán (nếu cần)
        if st.session_state.model != model_name:
            st.session_state.model = model_name
            
        if model_name != 'NPR':
            st.write('Load Adof model')
            model = resnet50(num_classes=1)
            model.load_state_dict(torch.load(model_paths[model_name], map_location='cpu'), strict=True)
            model.eval()
            st.write(model_paths[st.session_state.model])
        else:
            st.write('Load NPR model')
            model = npr(num_classes=1)
            model.load_state_dict(torch.load(model_paths[model_name], map_location='cpu'), strict=True)
            model.eval()
            st.write(model_paths[st.session_state.model])

        sample = st.selectbox("Or choose from sample images:", list(sample_images.keys()))
        # If an image is uploaded, make a prediction on it
        if uploaded_file is not None:
            file_name = uploaded_file.name
            #image = Image.open(uploaded_file)
            image = resize_image(uploaded_file)

            predictions = predict(model,image)
            #st.image(image, caption="Uploaded Image.", use_column_width=True)


        # If a sample image is chosen, make a prediction on it
        elif sample:
            file_name = sample.capitalize()
            image = Image.open(sample_images[sample])
            #st.image(image, caption=sample.capitalize() + " Image.", use_column_width=True)
            predictions = predict(model,image)

        if predictions:
            st.write("Predictions:")
            st.write(f"This image is {labels[predictions[1]]} ({predictions[0]*100:.2f}%)")
                        # Show progress bar with probabilities
            st.markdown(
                """
                <style>
                .stProgress .st-b8 {
                    background-color: orange;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.progress(int(predictions[0]*100))
        
        else:
            st.write("No predictions.")

    with col2:
        
        st.image(image, caption=file_name , use_column_width=True)

# Run the app
if __name__ == "__main__":
    app()