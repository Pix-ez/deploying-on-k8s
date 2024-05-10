from flask import Flask, request, jsonify
import os
from PIL import Image
from flask_cors import CORS

import torch
from torch import nn
import logging
from PIL import ImageOps
import cv2
import numpy as np

app = Flask(__name__)
CORS(app , origins="*")  # Enable CORS for the entire app


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using device: {device}")


# Create a convolutional neural network 
class CNN_MNIST(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x




    

MODEL_SAVE_PATH = './mnist_CNN.pth'

model = CNN_MNIST(input_shape=1, 
    hidden_units=10, 
    output_shape=10).to(device)

print(f"Loading the model from: {MODEL_SAVE_PATH}")
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=torch.device('cpu')))


# next(model.parameters()).device

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters() ,lr=0.001 ,)



# Configure the directory where images will be saved
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': "Hello from server"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        if image_file:
            save_folder = './uploads'
        
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            img= cv2.imread(image_path)
      
            

         


            jpg_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
           
        

            # res_img = cv2.resize(jpg_img ,(28,28))
            # Resize the image to a smaller size with anti-aliasing
            width, height = 28, 28
            res_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            gray_img = cv2.cvtColor(res_img ,cv2.COLOR_RGB2GRAY)

            save_path = os.path.join(save_folder, 'processed_' + image_file.filename)
            cv2.imwrite(save_path, gray_img)

         

            # gray_img = cv2.cvtColor(image_file ,cv2.COLOR_RGB2GRAY)

            # new_img = Image.open(jpg_img)

            # np_array = np(new_img)

            print(gray_img.shape)

            img_tensor = torch.tensor(gray_img)
            
            # flattened_image = img_tensor.reshape(1, -1)
            # flattened_image = flattened_image.unsqueeze(0)

            print(img_tensor.shape)
            tensor = img_tensor.to(torch.float32)

            tensor = tensor.to(device)
            print(tensor.dtype)



          

           

            # Convert the transformed image to a PyTorch tensor
            # tensor_image = torch.unsqueeze(preprocessed_image_path, 0).to(device)

            # print("Tensor image shape:", tensor_image.shape)
            model.eval()
            with torch.inference_mode():
              
                image = tensor.unsqueeze(0)
                image = image.unsqueeze(0) 
                logits = model(image)

            
            ans =logits.argmax().tolist()

            print(logits.tolist())
            print(ans)


            # with torch.inference_mode():
            #     logits = model(tensor)
            
            # ans = str(logits.argmax())

            # logits_list = logits.tolist()

            # return jsonify({'logits': logits_list, 'ans': ans}), 200
            return jsonify({'logits': 'ffs' , 'ans':ans}), 200
        
        else:
            return jsonify({'error': 'Invalid image format'}), 400

    except Exception as e:
        logging.exception(f'An error occurred during image processing: {e}')
        return jsonify({'error': 'An error occurred'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002, host="0.0.0.0")
