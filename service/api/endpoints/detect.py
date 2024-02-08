from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
# import cv2
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from service.core.logic.onnx_inference import emotions_detector, emotion_detection_result
from service.core.schemas.output import APIOutput
from fastapi import Request
from service.core.logic.system_monitor import monitor_memory_usage
import psutil

detect_router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Get CPU usage as a percentage
cpu_usage = psutil.cpu_percent(interval=1)  # Check every 1 second
print(f"CPU Usage 1: {cpu_usage}%")
monitor_memory_usage()

@detect_router.post('/detect', response_class=HTMLResponse, response_model=APIOutput)
async def detect(request: Request, vid: UploadFile):
    print(f"CPU Usage 2: {cpu_usage}%")
    print(vid.filename)
    monitor_memory_usage()

    if vid.filename.split('.')[-1] in ('mp4', 'avi', 'mov'):  # jpg, jpeg, png
    # if vid.content_type.startswith('video/'):
        # print(vid.content_type)
        pass

    else:
        raise HTTPException(
            status_code=415, detail='Not an Video'
        )

    # Read the uploaded image ######################## edit code for 'video'
    # video = Image.open(BytesIO(await vid.read())) #############################
    # fps = int(video.get(cv2.CAP_PROP_FPS))  # fps == Frames Per Second
    # frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"CPU Usage 3: {cpu_usage}%")
    monitor_memory_usage()
    # image = np.array(image)

    emotion_list = emotions_detector(vid)
    result_data = emotion_detection_result(emotion_list)
    print(result_data)
    print(f"CPU Usage 6: {cpu_usage}%")
    monitor_memory_usage()
    print(result_data)

    return templates.TemplateResponse("result.html", {"request": request, "result_data": result_data})

