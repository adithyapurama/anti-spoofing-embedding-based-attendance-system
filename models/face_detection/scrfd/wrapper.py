# face_detection/scrfd/wrapper.py
import os
import numpy as np
from pathlib import Path
from .detector import SCRFD


class SCRFDWrapper:
    """
    Thin adapter so we can call `retina.detect(frame)` like old RetinaFace.
    Output format:
    [
        {
            "bbox": [x1, y1, x2, y2],
            "landmarks": [[x,y], ... 5 points],
            "score": float
        }
    ]
    """
    def __init__(self, model_path: str = None, input_size=(640, 480), score_thresh=0.5):
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent / "weights" / "scrfd_10g.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SCRFD model not found: {model_path}")

        self.detector = SCRFD(model_file=model_path)
        self.detector.prepare(ctx_id=-1, input_size=input_size)  # force CPU runtime
        self.input_size = input_size
        self.score_thresh = score_thresh

    def detect(self, frame):
        """Returns list of faces: bbox + landmarks(5x2)."""
        if frame is None:
            return []

        bboxes, kps = self.detector.detect(
            frame,
            thresh=self.score_thresh,
            input_size=self.input_size,
            max_num=10
        )

        faces = []
        if bboxes is None or len(bboxes) == 0:
            return faces

        bboxes = np.asarray(bboxes)

        for i in range(bboxes.shape[0]):
            row = bboxes[i]

            # Handle score present or absent
            if row.shape[0] == 5:
                x1, y1, x2, y2, score = map(float, row)
                score = float(score)
            else:
                x1, y1, x2, y2 = map(float, row)
                score = None

            face = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            }

            # Convert keypoints to 5×2 format
            if kps is not None and len(kps) > i:
                # kps[i] is 5×2
                kp_5x2 = np.array(kps[i]).astype(float).tolist()
                face["landmarks"] = kp_5x2
            else:
                face["landmarks"] = []

            if score is not None:
                face["score"] = score

            faces.append(face)

        return faces
