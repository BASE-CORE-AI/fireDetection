from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import numpy as np
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, img = cap.read()
    
    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results = model.predict(img)

    for r in results:
        
        annotator = Annotator(img)
        
        boxes = r.boxes.numpy()
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.xywh[0]
            x=c[0]
            y=c[1]
            x = np.interp(x,[0,640],[-320,320])
            y = np.interp(y,[0,480],[-240,240])
            coord = str(x) + ' , '+str(y)
            annotator.box_label(b, coord)
    
    

    img = annotator.result()  
    cv2.imshow('YOLO V8 Detection', img)     
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()