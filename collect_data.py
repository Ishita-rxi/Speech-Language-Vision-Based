import cv2
import os
import time

# List all 5 signs + NONE class
ACTIONS = ['GOOD', 'BAD', 'OKAY', 'PEACE', 'NONE']
DATA_PATH = "dataset"
SAMPLES_PER_CLASS = 100 
x1, y1, x2, y2 = 50, 50, 450, 450 

cap = cv2.VideoCapture(0)

for action in ACTIONS:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)
    count = 0
    is_capturing = False
    print(f"--- NEXT CLASS: {action} ---")
    
    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[y1:y2, x1:x2]
        
        # Color Masking (No Grayscale)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 30, 60), (20, 255, 255))
        res = cv2.bitwise_and(roi, roi, mask=mask)
        
        color = (0, 255, 0) if is_capturing else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"COLLECTING: {action} ({count}/{SAMPLES_PER_CLASS})", (x1, y1-10), 0, 0.7, color, 2)
        
        if not is_capturing:
            cv2.putText(frame, "Press 'S' to Start Burst", (120, 250), 0, 0.8, (255, 255, 255), 2)

        cv2.imshow('Data Collector', frame)
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('s'): 
            is_capturing = True
            time.sleep(0.5) # Time to pose
        
        if is_capturing:
            cv2.imwrite(os.path.join(DATA_PATH, action, f"{count}.jpg"), res)
            count += 1
            time.sleep(0.05) # Delay to capture variations in motion
            
        if key & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()