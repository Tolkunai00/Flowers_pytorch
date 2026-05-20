from fastapi import UploadFile, File, HTTPException, FastAPI
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import uvicorn


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class FlowersVGG(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
  )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 8 * 8, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 5)
    )

  def forward(self, flower):
      flower = self.first(flower)
      flower = self.second(flower)
      return flower


flower_app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FlowersVGG()
model.load_state_dict(torch.load('flowers_model.pth', map_location=device))
model.to(device)
model.eval()


classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


@flower_app.post('/predict')
async def flower_image(image: UploadFile = File(...)):
    try:
        image_data = await image.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='Файл табылган жок')

        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred =model(img_tensor)
            pred = y_pred.argmax(dim=1).item()

        prediction = classes[pred]

        return {'predict': prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(flower_app, host='127.0.0.1', port=8003)

