from fastapi import FastAPI, Request
import os

# os.chdir('/home/hail/Documents/hailLAB_luna/240121_FastAPI_practice/Face_emotion_detection_fastapi')
# import sys
# print(sys.path.append('/home/hail/Documents/hailLAB_luna/240121_FastAPI_practice/Face_emotion_detection_fastapi'))
# print(sys.path)
# print(os.getcwd())

# from .api.api import main_router
# from api import api
from api import api
import onnxruntime as rt
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


# main_router = api.main_router
app = FastAPI(project_name='Emotion_detection')
app.include_router(api.main_router)

providers = ['CPUExecutionProvider']
model = rt.InferenceSession('service/core/logic/vit_quantized.onnx', providers=providers)

# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "name": "Rahul"})