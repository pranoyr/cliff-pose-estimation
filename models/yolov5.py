import torch


class YOLO():
	""" YOLO model for person detection
	"""
	def __init__(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
	def detect_person(self, frame):
		results = self.model(frame)
		final_results = results.pandas().xyxy[0].values.tolist()
		for result in final_results:
			x1, y1, x2, y2, conf, _,  cls = result
			if cls == "person":
				return torch.tensor([x1, y1, x2, y2])
				


