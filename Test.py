import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from threading import Thread

# Replace 'your_video_file.mp4' with the path to your video file
video_file_path = 'Test1.mp4'
cap = cv2.VideoCapture(video_file_path)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize variables
car_count = 0
car_count_threshold = 15
traffic_light_color = "green"
red_light_duration = 10
yellow_light_duration = 2
green_light_duration = 10
light_timer = 0
simulation_running = False

def control_traffic_light(color):
    global traffic_light_color  # Declare traffic_light_color as a global variable
    print(f"Traffic light turned {color}")

def draw_green_boxes(frame, detections):
    for detection in detections:
        x, y, w, h = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def start_simulation():
    global simulation_running
    simulation_running = True
    simulation_thread = Thread(target=run_simulation)
    simulation_thread.start()

def stop_simulation():
    global simulation_running
    simulation_running = False

def run_simulation():
    global light_timer, traffic_light_color  # Declare light_timer and traffic_light_color as global variables
    while simulation_running:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        car_count = 0
        car_detections = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.95 and class_id == 2:
                    car_count += 1
                    car_detections.append(detection)

        if light_timer == 0:
            if car_count >= car_count_threshold:
                if traffic_light_color == "green":
                    control_traffic_light("red")
                    traffic_light_color = "red"
                    light_timer = red_light_duration
                elif traffic_light_color == "red":
                    control_traffic_light("yellow")
                    traffic_light_color = "yellow"
                    light_timer = yellow_light_duration
                else:
                    control_traffic_light("green")
                    traffic_light_color = "green"
                    light_timer = green_light_duration

        light_timer = max(0, light_timer - 1)

        car_count_var.set(f"Car Count: {car_count}")
        traffic_light_var.set(f"Traffic Light: {traffic_light_color}")

        draw_green_boxes(frame, car_detections)

        cv2.putText(frame, f"Car Count: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Traffic Light: {traffic_light_color}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Traffic Management", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Traffic Management Simulation")
root.geometry("400x200")

car_count_var = tk.StringVar()
car_count_var.set("Car Count: 0")

traffic_light_var = tk.StringVar()
traffic_light_var.set("Traffic Light: Green")

car_count_label = ttk.Label(root, textvariable=car_count_var)
car_count_label.pack(pady=20)

traffic_light_label = ttk.Label(root, textvariable=traffic_light_var)
traffic_light_label.pack()

start_button = ttk.Button(root, text="Start Simulation", command=start_simulation)
start_button.pack(pady=20)

stop_button = ttk.Button(root, text="Stop Simulation", command=stop_simulation)
stop_button.pack()

root.mainloop()
