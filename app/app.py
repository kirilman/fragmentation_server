# server

from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import base64
import json
import io
# from Image_analysis import *
import torch
import gc
from model_predict import BoxModel, draw_boxes

YOLO_BOX = BoxModel('./app/box.pt')
YOLO_BOX = BoxModel('./app/obb.pt')
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello", "Start server image fragmentation!"}

@app.post("/box")
async def process(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_dimensions = str(img.shape)
    anno = YOLO_BOX.predict_box(img)
    draw_boxes(img, anno[0])
    gc.collect()
    torch.cuda.empty_cache()
    return {"bbox": len(anno[0])}


@app.post("/model_inferences")
async def process(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_dimensions = str(img.shape)
    image_with_bbox, predcited_labels = analise(img)

    _, image_with_bbox = cv2.imencode('.JPEG', image_with_bbox)
    image_with_bbox = base64.b64encode(image_with_bbox)

    gc.collect()
    torch.cuda.empty_cache()
    return {"image_with_bbox":image_with_bbox,
            'image_dimensions': img_dimensions,
            "predcited_labels": predcited_labels}