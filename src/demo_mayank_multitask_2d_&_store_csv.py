from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from src.lib.opts_mayank_mutlitask import opts
from src.lib.detector import Detector
score_thres=0.2
saving_path='/home/mayank_s/Desktop/centertrack/demo'
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']
bblabel=[]
class_name = [
    'car', 'truck', 'bus', 'trailer',
    'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
    'traffic_cone', 'barrier']
def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    is_video = True
    # demo on video stream
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
  else:
    is_video = False
    # Demo on images sequences
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

  # Initialize output video
  out = None
  out_name = opt.demo[opt.demo.rfind('/') + 1:]
  print('out_name', out_name)
  if opt.save_video:
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('../results/{}.mp4'.format(
      opt.exp_id + '_' + out_name),fourcc, opt.save_framerate, (
        opt.video_w, opt.video_h))
  
  if opt.debug < 5:
    detector.pause = False
  cnt = 0
  results = {}

  while True:
      if is_video:
        _, img = cam.read()
        if img is None:
          save_and_exit(opt, out, results, out_name)
      else:
        if cnt < len(image_names):
          img = cv2.imread(image_names[cnt])
        else:
          save_and_exit(opt, out, results, out_name)
      cnt += 1

      # resize the original video for saving video results
      if opt.resize_video:
        img = cv2.resize(img, (opt.video_w, opt.video_h))

      # skip the first X frames of the video
      if cnt < opt.skip_first:
        continue
      
      cv2.imshow('input', img)

      # track or detect the image.
      ret = detector.run(img)

      ############################33333
      resu = ret['results']
      for data in resu:
          # print(resu[data])
          arr=data['bbox']
          # $$$$$$$$$$$$$$$$$$$$$$
          xmin = int(arr[0])
          ymin = int(arr[1])
          xmax = int(arr[2])
          ymax = int(arr[3])
          score = float(data["score"])
          cls = int(data["class"])
          width = 1
          height = 1
          if score<score_thres:
              continue

          object_name=class_name[cls-1]
          image_name="cool"

          data_label = [image_name, width, height, object_name, xmin, ymin, xmax, ymax, score]
          # data_label = [file_name, width, height, object_name, xmin, ymin, xmax, ymax]
          if not ((xmin == xmax) and (ymin == ymax)):
              bblabel.append(data_label)
              # print(file_name)
              # print()

          top = (xmin,ymax)
          bottom = (xmax, ymin)
          cv2.rectangle(img, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
          # cv2.putText(image_scale, y[read_index][4], ((y[read_index][0]+y[read_index][2])/2, y[read_index][1]), cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 255)
          cv2.putText(img, str(object_name), (int((xmin +xmax) / 2), ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), lineType=cv2.LINE_AA)
          cv2.putText(img, (str(round(score,2))), (int((xmin +xmax) / 2), ymin - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), lineType=cv2.LINE_AA)
      # else:
      #     1

      cv2.imshow('streched_image', img)
      ch = cv2.waitKey(10)
      if ch & 0XFF == ord('q'):
          cv2.destroyAllWindows()
      # cv2.waitKey(1)
      # cv2.destroyAllWindows()
      # output_path = saving_path + ((image_names[cnt].split("/")[-1])
      image_name= (image_names[cnt].split("/")[-1])
      output_path = os.path.join(saving_path, image_name)
      cv2.imwrite(output_path, img)


      #######################################
  #
  #     # log run time
  #     time_str = 'frame {} |'.format(cnt)
  #     for stat in time_stats:
  #       time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
  #     print(time_str)
  #
  #     # results[cnt] is a list of dicts:
  #     #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
  #     results[cnt] = ret['results']
  #
  #     # save debug image to video
  #     if opt.save_video:
  #       out.write(ret['generic'])
  #       if not is_video:
  #         cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])
  #
  #     # esc to quit and finish saving video
  #     if cv2.waitKey(1) == 27:
  #       save_and_exit(opt, out, results, out_name)
  #       return
  # save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name=''):
  if opt.save_results and (results is not None):
    save_dir =  '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_dir, 'w'))
  if opt.save_video and out is not None:
    out.release()
  sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  # opt = opts().parse()
  demo(opt)
