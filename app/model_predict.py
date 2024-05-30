from ultralytics import YOLO
import cv2
from asbestutills.plotter.plotting import plot_bboxs, plot_obounding_box
from datetime import datetime


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
    # predcited_labels = results[0].names[int(results[0].boxes.cls)]


# image = cv2.imread(
#     "/media/kirilman/Z/dataset/rocks/SAM-Rock-Fragmentation-main/images/103.JPG"
# )
# anno = predict_box(image)
# draw_anno(image,anno[0])
