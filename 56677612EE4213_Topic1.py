import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
import pyautogui as pg

# init
reset_ctl = False
player_ctl = -1
timer_ctl = 0
hand_pos = -1
hand_land_marks = -1

# gesture recognizer
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def get_finger_num():
    global hand_land_marks
    ret = 0
    threshold = (hand_land_marks.landmark[0].y*100 - hand_land_marks.landmark[9].y*100)/2
    if (hand_land_marks.landmark[5].y*100 - hand_land_marks.landmark[8].y*100) > threshold:
        ret += 1
    if (hand_land_marks.landmark[9].y*100 - hand_land_marks.landmark[12].y*100) > threshold:
        ret += 1
    if (hand_land_marks.landmark[13].y*100 - hand_land_marks.landmark[16].y*100) > threshold:
        ret += 1
    if (hand_land_marks.landmark[17].y*100 - hand_land_marks.landmark[20].y*100) > threshold:
        ret += 1
    if (hand_land_marks.landmark[5].x*100 - hand_land_marks.landmark[4].x*100) > 6:
        ret += 1

    return ret

def player_control():
    global reset_ctl, player_ctl
    if reset_ctl:
        match player_ctl:
            case 1:
                pg.press('space')
            case 2:
                pg.press('down')
            case 3:
                pg.press('up')
            case 4:
                pg.moveTo((1-hand_pos.x)*1920, hand_pos.y*1080)
            case 5:
                pg.mouseDown()
                pg.moveTo((1-hand_pos.x)*1920, hand_pos.y*1080)
            case 6:
                pg.click()
        if player_ctl != -1 and player_ctl != 4 and player_ctl != 5:
            reset_ctl = False
            player_ctl = -1
        if player_ctl == -1:
            pg.mouseUp()

def callback_gesture(results: GestureRecognizerResult, out_image: mp.Image, timestamp_ms: int):
    global reset_ctl, player_ctl, timer_ctl
    if(len(results.gestures) and timestamp_ms - timer_ctl > 300):
        print(results.gestures[0][0].category_name)
        match results.gestures[0][0].category_name:
            case 'Closed_Fist':
                reset_ctl = True
                timer_ctl = time.time()*1000
                player_ctl = -1
            case 'Open_Palm':
                player_ctl = 1
            case 'Thumb_Down':
                player_ctl = 2
            case 'Thumb_Up':
                player_ctl = 3
            case 'Pointing_Up':
                player_ctl = 4
                timer_ctl = time.time()*1000-100
            case 'Victory':
                player_ctl = 5
                timer_ctl = time.time()*1000-100
            case _:
                if get_finger_num() == 3:
                    player_ctl = 6
                else:
                    reset_ctl = False
                    player_ctl = -1

        # action
        player_control()

model_path = 'gesture_recognizer.task'
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=callback_gesture)

# hand landmarker
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# main
def main():
    cap = cv2.VideoCapture(0)
    # get size of frame in video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with (
        GestureRecognizer.create_from_options(options) as recognizer,
        mp_hands.Hands(model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as landmarker ):
        while(cap.isOpened()):
            ret, frame = cap.read()
            if(not(ret)):
                continue

            frame.flags.writeable = False
            # frame for cv2 graphics
            # np_frame for processing
            np_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_frame)
            hand_results = landmarker.process(np_frame)
            if(hand_results.multi_hand_landmarks):
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                # index finger tip
                global hand_pos, hand_land_marks
                hand_pos = hand_results.multi_hand_landmarks[0].landmark[8]
                hand_land_marks = hand_results.multi_hand_landmarks[0]
            recognizer.recognize_async(mp_frame, int(time.time()*1000))

            cv2.imshow('Gesture Media Player Controller', cv2.flip(frame, 1))
            if(cv2.waitKey(1) & 0xFF == 27):
                break

    # end
    cap.release()

if __name__ == '__main__':
    main()
