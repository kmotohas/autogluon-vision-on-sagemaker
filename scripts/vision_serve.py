import os
import json
#import tarfile
#from typing import Any

#import numpy as np
#import autogluon.core as ag
from sagemaker_inference import (
    content_types,
    decoder,
    default_inference_handler,
    encoder,
    errors,
)
from autogluon.vision import (
    ImageDataset,
    ImagePredictor,
)


def model_fn(model_dir: str) -> ImagePredictor:
    print('*****loading model*****')
    os.system(f'ls {model_dir}')
    predictor = ImagePredictor.load(model_dir)
    print('*****loaded model*****')
    return predictor


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):

    data = decoder.decode(request_body, input_content_type)
    #print(data)
    from PIL import Image
    im = Image.fromarray(data)
    im.save('0.jpg')
    import pandas as pd
    df = pd.DataFrame({"image": ['0.jpg']})
    
    prediction = model.predict(df)

    return json.dumps(prediction.tolist()), output_content_type
