import os
import sys
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import torch
from PIL import Image
import io
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src import Generator
from torchvision.transforms import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

transform = transforms.Compose([
    transforms.ToTensor(),
])


# Load your model
generator =  Generator().to(device)
generator.load_state_dict(torch.load('SRGAN.pth', map_location=device))
generator.eval()

def process_image(image: Image.Image) -> Image.Image:
    image = transform(image).unsqueeze(0).to(device)
    start = time.time()
    with torch.no_grad():
        generated = generator(image)
    generated = generated.squeeze(0).cpu().detach().clamp(0, 1)
    pil_image = transforms.ToPILImage()(generated)
    end = time.time()
    print(f'Generated in {end-start} seconds | using {str(device).upper()}')
    return pil_image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Process the image using the model
    output_image = process_image(image)
    
    # Convert the output image to bytes
    buffer = io.BytesIO()
    output_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Return the output image as a response
    return StreamingResponse(buffer, media_type="image/png")

# Serve the static index.html file
@app.get("/")
def read_root():
    return FileResponse('api/index.html')