#importing neccessary libraries
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template

#creating a new instance of the Flask web application.
app = Flask(__name__)

#loading the model architecture
model = torch.load('lcd.pth')

#Processing device (cuda or cpu)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Initializing the Model to the resource hardware
model = model.to(device)

#entering into eval mode to avoid gradient tracking
model.eval()

"""compiling transforms like -
	Resizing to 256
	Cropping it to 224px
	PIL Image to tensor
	adding extra dimensions for considering as a batch"""

def transform_image(image_bytes):
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])
    image = Image.open(io.BytesIO(image_bytes))
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mean_value = cv2.mean(image_array)[0]
    _, thresholded = cv2.threshold(image_array, mean_value, 255, cv2.THRESH_BINARY)
    thresholded_image = Image.fromarray(cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB))
    t = transform(thresholded_image)
    return t.unsqueeze(0)

#getting prediction by forwarding it to the model and taking the maximum elements index 
def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        tensor = tensor.to(device)
        outputs = model.forward(tensor)
        l = outputs.logits[0]
        outt = torch.tensor([l[2],l[6],l[14]]).argmax()
        if outt.item() == 0:
            return "Email"
        elif outt.item() == 1:
            return "Scientific Publication"
        else:
            return "Resume"
    except:
        print("Some Error Occured")
        return "Error Occured"

#returns index.html template 
@app.route('/')
def hello():
    return render_template("index.html")

#returns result.html
@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['image']
    t = file.read()
    res = get_prediction(t)
    return render_template("result.html",message=res,ico="images/"+res+".ico")

#starts the flask development server
if __name__ == '__main__':
    app.run()
