import base64
import io

import cv2
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from tensorflow.keras.models import load_model
from starlette.requests import Request

from using_modal_test_server import using

my_model = load_model("model_phu_tung.h5")


def readb64(base64_string):
    path = 'cache/predict.jpg'
    with open(path, 'wb') as f_output:
        f_output.write(base64.b64decode(base64_string))
    return path


app = FastAPI()


class BodyObjectPredict(BaseModel):
    image: str


@app.post("/object_predict")
async def root(req: Request, image: BodyObjectPredict):
    path = readb64(image.image)
    host = req.url.hostname
    port = req.url.port
    res = using(path, my_model, host, port)
    return res


@app.get("/image")
async def get_image(file: str = 'out.jpg'):
    cv2img = cv2.imread('cache/' + file)
    res, im_png = cv2.imencode(".png", cv2img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
