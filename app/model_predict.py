from ultralytics import YOLO
import cv2
from asbestutills.plotter.plotting import plot_bboxs, plot_obounding_box
from datetime import datetime

#curl --location 'http://127.0.0.1:8787/box' -X POST -F 'file=@"received_image.jpeg"

class BoxModel:
    def __init__(self, path2weight):
        self.model = YOLO(path2weight)

    def predict_box(self, image):
        results = self.model.predict(image, max_det=1500, device="cpu", conf=0.3)
        return results


def draw_boxes(image, anno):
    """
    anno [List[torch.tensor]]: yolo anno
    """
    size = max(image.shape)
    t = int(size * 0.0015)
    if anno.boxes:
        if len(anno.boxes.xyxyn) > 0:
            image_with_bbox = plot_bboxs(image, anno.boxes.xyxyn, t)
    elif anno.obb:
        if len(anno.obb.xyxyxyxyn) > 0:
            N = anno.obb.xyxyxyxyn.shape[0]
            image_with_bbox = plot_obounding_box(
                image, anno.obb.xyxyxyxyn.reshape(N, -1).detach().numpy(), t
            )
    now = datetime.now()
    cv2.imwrite(f"./result/result_{now}.jpeg", image_with_bbox)
