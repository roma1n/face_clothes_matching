import mediapipe as mp
from skimage import (
    transform,
    util,
)

import config
from lib.models.abstract_model import AbstractModel


def upscale_relative_bb(bb, alpha=0.75):
    h, w = bb.height, bb.width
    bb.xmin = max(0.0, bb.xmin - alpha * w / 2)
    bb.ymin = max(0.0, bb.ymin - alpha * h / 2)
    bb.width = min (1.0 - bb.xmin, bb.width * (1 + alpha))
    bb.height = min (1.0 - bb.ymin, bb.height * (1 + alpha))
    
    return bb


def crop_image_by_bb(image, bb):
    assert bb is not None, 'Bounding box is not provided'
    h, w, _ = image.shape
    cropped = util.crop(
        image,
        (
            (int(h * bb.ymin), int(h * (1 - bb.ymin - bb.height))),
            (int(w * bb.xmin), int(w * (1 - bb.xmin - bb.width))),
            (0,0)
        ),
        copy=True
    )
    return cropped


class FaceCropper(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)

        self.mp_face_detection = mp.solutions.face_detection

    def apply(self, input):
        with self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        ) as face_detection:
            results = face_detection.process(input)

        bb = None
        
        for detection in results.detections:
            bb = upscale_relative_bb(detection.location_data.relative_bounding_box)

        if bb is None:
            return None

        cropped = crop_image_by_bb(input, bb)
        cropped = transform.resize(cropped, config.EMBEDDER_INPUT_SHAPE)

        return cropped
