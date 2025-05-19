#!/usr/bin/env python3
"""Test MediaPipe hand detection format to understand the output."""

import time
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe hands directly
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create a simple test image with a hand (you can use webcam instead)
cap = cv2.VideoCapture(0)  # Use webcam

print("MediaPipe format test - Press 'q' to quit")
print("This will show the raw MediaPipe output format")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    
    frame_count = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read from camera")
            continue
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = hands.process(image_rgb)
        
        # Debug output every second
        if frame_count % 30 == 0:
            print(f"\nFrame {frame_count}:")
            
            if results.multi_hand_landmarks:
                print(f"  Detected {len(results.multi_hand_landmarks)} hands")
                
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    print(f"\n  Hand {i}:")
                    # Check first landmark (wrist)
                    wrist = hand_landmarks.landmark[0]
                    print(f"    Wrist - x: {wrist.x:.3f}, y: {wrist.y:.3f}, z: {wrist.z:.3f}")
                    print(f"    Type: {type(wrist.x)}")
                    
                    # Check if handedness is available
                    if results.multi_handedness:
                        handedness = results.multi_handedness[i].classification[0]
                        print(f"    Handedness: {handedness.label} (score: {handedness.score:.3f})")
                    
                    # Count landmarks
                    print(f"    Total landmarks: {len(hand_landmarks.landmark)}")
            else:
                print("  No hands detected")
        
        # Draw the hand annotations on the image
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
        
        # Display info
        cv2.putText(image_bgr, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('MediaPipe Format Test', image_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("\nTest finished!")