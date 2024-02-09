import time
import random
import numpy as np
import cv2

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    new_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    new_width = int(new_frame.shape[1] * 100 / percent)
    new_height = int(new_frame.shape[0] * 100 / percent)
    new_dim = (new_width, new_height)
    return cv2.resize(new_frame, new_dim, interpolation = cv2.INTER_AREA)

max_change = 15
per = 10
prepare_time = 4
end_time = 0

frame_color_size = 150
white_color  = (255, 255, 255)
red_color = (0, 0, 200)
green_color = (0, 150, 0)
what_color = False
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
background_color = (0, 0, 0)


cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame_old = cap.read()
ret, frame_new = cap.read()

frame_old_low = rescale_frame(frame_old, percent=per)
frame_new_low = rescale_frame(frame_new, percent=per)
frame_color = np.zeros((frame_color_size, frame_color_size, 3), dtype=np.uint8)

def some_text_style(text):
  text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
  text_x = text_size[1] // 2 - 1
  text_y = frame_height - text_size[1] // 2
  background_x = text_x - 10
  background_y = text_y - 30
  background_width = text_size[0] + 20
  background_height = text_size[1] + 20
  return ([text_x, text_y, background_x, background_y, background_width, background_height])

while True:
  key = 0
  fl = False
  start_time = time.time()
  start_interval_time = 0
  interval_time = 0

  while True:
    key = cv2.waitKey(20) & 0xff
    current_time = time.time() - start_time
    diff = cv2.absdiff(frame_old_low, frame_new_low)
    mask = np.abs(diff) <= max_change
    img_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame_old, contours, -1, (0, 0, 255))
    frame_old = frame_new
    frame_old_low = frame_new_low
    ret, frame_new = cap.read()
    frame_new_low = rescale_frame(frame_new, percent=per)
    if fl:
      if current_time < end_time:
        text = 'You Lose!'
        text_style = some_text_style(text)
        cv2.rectangle(frame_color, (0, 0), (frame_color_size - 1, frame_color_size - 1), white_color, -1)
        cv2.rectangle(frame_old, (text_num_style[2], text_style[3]), (text_style[2] + text_style[4], text_style[3] + text_style[5]), background_color, -1)
        cv2.putText(frame_old, text, (text_style[0], text_style[1]), font, font_scale, white_color, font_thickness)
      else:
        break
    else:
      if current_time <= prepare_time:
        cv2.rectangle(frame_color, (0, 0), (frame_color_size - 1, frame_color_size - 1), white_color, -1)
        text_num = 'Game start in: ' + str(int(prepare_time - current_time)) + ' sec'
        text_num_style = some_text_style(text_num)
        cv2.rectangle(frame_old, (text_num_style[2], text_num_style[3]), (text_num_style[2] + text_num_style[4], text_num_style[3] + text_num_style[5]), background_color, -1)
        cv2.putText(frame_old, text_num, (text_num_style[0], text_num_style[1]), font, font_scale, white_color, font_thickness)
      else:
        if start_interval_time + interval_time <= current_time:
          interval_time = random.randint(1, 8)
          start_interval_time = current_time
          what_color = not what_color
        text = ''
        font_color = (0, 0, 0)
        if what_color:
          text = 'Green Light'
          font_color = green_color
        else:
          text = 'Red Light'
          font_color = red_color
        text_style = some_text_style(text)
        cv2.rectangle(frame_old, (text_num_style[2], text_style[3]), (text_style[2] + text_style[4], text_style[3] + text_style[5]), background_color, -1)
        cv2.putText(frame_old, text, (text_style[0], text_style[1]), font, font_scale, font_color, font_thickness)

        if np.all(mask):
          cv2.rectangle(frame_color, (0, 0), (frame_color_size - 1, frame_color_size - 1), green_color, -1)
        else:
          cv2.rectangle(frame_color, (0, 0), (frame_color_size - 1, frame_color_size - 1), red_color, -1)
          if not what_color:
            fl = True
            end_time = current_time + prepare_time


    cv2.imshow('COLOR', frame_color)
    cv2.imshow('FRAME_CAM', frame_old)
    if key == 13:
      cv2.destroyWindow('FRAME_CAM')
      cv2.destroyWindow('FRAME_COLOR')
      break
  if key == 13:
    break

cap.release()