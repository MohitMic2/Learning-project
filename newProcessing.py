#
#
# import socketio
# import cv2
# import numpy as np
# import base64
# import eventlet
# from datetime import datetime
# from engineio.payload import Payload
# from ultralytics import YOLO
#
# def log_with_timestamp(message):
#     """Add timestamp to log messages"""
#     current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
#     print(f"[{current_time}] {message}")
#
# # Increase payload size limit
# Payload.max_decode_packets = 500
#
# # Create Socket.IO server
# sio = socketio.Server(cors_allowed_origins='*')
# app = socketio.WSGIApp(sio)
#
# # Load YOLO model once at startup
# try:
#     model = YOLO("D:\\live attendance\\best(attendance).pt")
#     log_with_timestamp("✓ YOLO model loaded successfully")
# except Exception as e:
#     log_with_timestamp(f"❌ Error loading YOLO model: {str(e)}")
#     raise
#
# @sio.event
# def connect(sid, environ):
#     log_with_timestamp(f"Client connected: {sid}")
#
# @sio.event
# def disconnect(sid):
#     log_with_timestamp(f"Client disconnected: {sid}")
#
# def process_image(frame):   #function for image processing
#     """
#     Process the image here. Add your face detection/recognition code.
#     This is a placeholder function.
#     """
#     # Add a text overlay to show processing
#     processed = frame.copy()
#     cv2.putText(processed,
#                 'Processed Frame',
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 255, 0),
#                 2)
#     return processed
#
# @sio.event
# def send_frame(sid, frame_data):
#     try:
#         # Log image received
#         print(frame_data)
#         # object_info=frame_data.frameData
#         object_info1=frame_data['frameData']
#         log_with_timestamp("→ Image received from client")
#         # print('1:',object_info)
#         print('2:',object_info1)
#         # Decode image
#         img_data = base64.b64decode(object_info1)
#         img_array = np.frombuffer(img_data, dtype=np.uint8)
#         frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#
#         log_with_timestamp("⚙ Processing image...")
#
#         # Process with YOLO model
#         results = model(frame)
#
#         # Extract detections
#         detections = []
#         processed_frame = frame.copy()
#
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 # Get detection details
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                 class_id = int(box.cls[0])
#                 confidence = float(box.conf[0])
#                 class_name = model.names[class_id]
#
#                 # Draw bounding box
#                 cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(processed_frame, f"{class_name} {confidence:.2f}",
#                             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#                 detections.append({
#                     'class_name': class_name,
#                     'confidence': confidence
#                 })
#
#         log_with_timestamp("✓ Image processing completed")
#
#         # Encode processed frame
#         _, buffer = cv2.imencode('.jpg', processed_frame)
#         processed_frame_data = base64.b64encode(buffer).decode()
#
#         # Send processing status
#         sio.emit('processing_status', {
#             'message': 'Image processed successfully'
#         }, room=sid)
#
#         # Send processed image and results
#         result_data = {
#             'image': processed_frame_data,
#             'result': f"Detected {len(detections)} objects",
#             'timestamp': datetime.now().strftime("%H:%M:%S")
#         }
#
#         log_with_timestamp("← Sending processed image to client")
#         sio.emit('processed_image', result_data, room=sid)
#
#         # Send recognition result
#         if detections:
#             recognition_data = {
#                 'faces_detected': len(detections),
#                 'names': [d['class_name'] for d in detections],
#                 'confidence': max(d['confidence'] for d in detections)
#             }
#             sio.emit('recognition_result', recognition_data, room=sid)
#
#             # Send attendance update for each detection
#             for detection in detections:
#                 attendance_data = {
#                     'class_name': detection['class_name'],
#                     'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     'confidence': detection['confidence']
#                 }
#                 sio.emit('attendance_update', attendance_data, room=sid)
#
#         log_with_timestamp("← Sending recognition results to client")
#
#     except Exception as e:
#         error_msg = f"Error processing frame: {str(e)}"
#         log_with_timestamp(f"❌ {error_msg}")
#         sio.emit('processing_status', {'message': error_msg}, room=sid)
#
# @sio.event
# def exit(sid):
#     log_with_timestamp(f"Client exiting: {sid}")
#
# if __name__ == '__main__':
#     try:
#         port = 4000
#         log_with_timestamp(f"Starting server on port {port}...")
#         eventlet.wsgi.server(eventlet.listen(('', port)), app)
#     except Exception as e:
#         log_with_timestamp(f"❌ Server error: {str(e)}")


import socketio
import cv2
import numpy as np
import base64
import eventlet
from datetime import datetime
from engineio.payload import Payload
from ultralytics import YOLO

def log_with_timestamp(message):
    """Add timestamp to log messages"""
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{current_time}] {message}")

# Increase payload size limit
Payload.max_decode_packets = 500

# Create Socket.IO server
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

# Load YOLO model once at startup
try:
    model = YOLO("D:\\live attendance\\best(attendance).pt")
    log_with_timestamp("✓ YOLO model loaded successfully")
except Exception as e:
    log_with_timestamp(f"❌ Error loading YOLO model: {str(e)}")
    raise

@sio.event
def connect(sid, environ):
    log_with_timestamp(f"Client connected: {sid}")

@sio.event
def disconnect(sid):
    log_with_timestamp(f"Client disconnected: {sid}")

def process_image(frame):
    """
    Process the image here. Add your face detection/recognition code.
    This is a placeholder function.
    """
    # Add a text overlay to show processing
    processed = frame.copy()
    cv2.putText(processed,
                'Processed Frame',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)
    return processed

@sio.event
def send_frame(sid, frame_data):
    try:
        # Log image received
        print(frame_data)
        object_info1=frame_data['frameData']
        log_with_timestamp("→ Image received from client")
        print('2:',object_info1)
        # Decode image
        img_data = base64.b64decode(object_info1)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Display received image in a window
        cv2.imshow('Received Image', frame)
        cv2.waitKey(1)  # Update window and wait 1ms

        log_with_timestamp("⚙ Processing image...")

        # Process with YOLO model
        results = model(frame)

        # Extract detections
        detections = []
        processed_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get detection details
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                # Draw bounding box
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"{class_name} {confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detections.append({
                    'class_name': class_name,
                    'confidence': confidence
                })

        log_with_timestamp("✓ Image processing completed")

        # Encode processed frame
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_data = base64.b64encode(buffer).decode()

        # Send processing status
        sio.emit('processing_status', {
            'message': 'Image processed successfully'
        }, room=sid)

        # Send processed image and results
        result_data = {
            'image': processed_frame_data,
            'result': f"Detected {len(detections)} objects",
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }

        log_with_timestamp("← Sending processed image to client")
        sio.emit('processed_image', result_data, room=sid)

        # Send recognition result
        if detections:
            recognition_data = {
                'faces_detected': len(detections),
                'names': [d['class_name'] for d in detections],
                'confidence': max(d['confidence'] for d in detections)
            }
            sio.emit('recognition_result', recognition_data, room=sid)

            # Send attendance update for each detection
            for detection in detections:
                attendance_data = {
                    'class_name': detection['class_name'],
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'confidence': detection['confidence']
                }
                sio.emit('attendance_update', attendance_data, room=sid)

        log_with_timestamp("← Sending recognition results to client")

    except Exception as e:
        error_msg = f"Error processing frame: {str(e)}"
        log_with_timestamp(f"❌ {error_msg}")
        sio.emit('processing_status', {'message': error_msg}, room=sid)

@sio.event
def exit(sid):
    log_with_timestamp(f"Client exiting: {sid}")
    cv2.destroyAllWindows()  # Clean up windows when client exits

if __name__ == '__main__':
    try:
        port = 4000
        log_with_timestamp(f"Starting server on port {port}...")
        eventlet.wsgi.server(eventlet.listen(('', port)), app)
    except Exception as e:
        log_with_timestamp(f"❌ Server error: {str(e)}")