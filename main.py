# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 08:06:36 2025

@author: canoz
"""

import cv2
import mediapipe as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from multiprocessing import Process

def process_participant(
        camera_index, 
        model_path, 
        stream_name, 
        window_name, 
        lsl_source_id):

    # LSL Stream setup
    landmark_channel_count = 52 + 132  
    fps = 30

    info = StreamInfo(
        name=stream_name, 
        type='Landmark', 
        channel_count=landmark_channel_count, 
        nominal_srate=fps, 
        channel_format='float32', 
        source_id=lsl_source_id
    )
    outlet = StreamOutlet(info)

    # Mediapipe setup
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic       = mp.solutions.holistic

    with open(model_path, "rb") as f:
        model_bytes = f.read()
    base_options = python.BaseOptions(model_asset_buffer=model_bytes)
    blendshape_options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.IMAGE
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(blendshape_options)

    cap = cv2.VideoCapture(camera_index)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True
    ) as holistic:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Holistic
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hol_res = holistic.process(rgb)

            # Blendshapes
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            blend_res  = face_landmarker.detect(mp_image)
            blend_scores = []
            blend_names  = []

            if blend_res.face_blendshapes:
                for cat in blend_res.face_blendshapes[0]:
                    blend_scores.append(cat.score)
                    blend_names.append(cat.category_name)
            else:
                # no detections → all zeros + empty names
                blend_scores = [0.0]*52
                blend_names  = ['']*52

            # In case some categories are missing (shouldn't happen), pad
            while len(blend_scores) < 52:
                blend_scores.append(0.0)
                blend_names.append('')

            # Build LSL sample
            sample = blend_scores.copy()
            if hol_res.pose_landmarks:
                for lm in hol_res.pose_landmarks.landmark:
                    sample.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                sample.extend([0.0] * (33 * 4))

            outlet.push_sample(sample)

            # Visualization + barplot
            frame.flags.writeable = True
            disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # draw meshes
            mp_drawing.draw_landmarks(
                disp,
                hol_res.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                disp,
                hol_res.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style()
            )

            # barplot on right
            h, w, _ = disp.shape
            plot_w   = 300
            x0       = w - plot_w
            bar_h    = int(h / 52)
                        

            for i, (name, score) in enumerate(zip(blend_names, blend_scores)):
                y0 = i * bar_h
                y1 = y0 + bar_h
                length = int(score * (plot_w - 100))
                cv2.rectangle(disp, (x0, y0), (x0 + length, y1), (0, 255, 0), -1)
                cv2.putText(
                    disp,
                    name,
                    (x0 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            cv2.imshow(window_name, disp)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = r"D:\Projects\MovieSynchrony\face_landmarker.task"
    # Participant 1
    p1 = Process(target=process_participant, args=(
        0, model_path, "Participant1_Holistic", "Webcam1", "holistic1_uid"))
    # Participant 2
    p2 = Process(target=process_participant, args=(
        1, model_path, "Participant2_Holistic", "Webcam2", "holistic2_uid"))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
