# YOLOv3 + Deep_SORT
YOLOv3 + Deep_SORT 实现多类多目标检测(计数)
<img src="https://github.com/xiaoxiong74/Object-Detection-and-Tracking/blob/master/output/result.png" width="80%" height="80%"> 
<img src="https://github.com/xiaoxiong74/Object-Detection-and-Tracking/blob/master/output/st1_vedio_person_output.gif" width="80%" height="80%"> 

## Requirement
* OpenCV
* keras
* NumPy
* sklean
* Pillow
* tensorflow-gpu 1.10.0 
***

It uses:

* __Detection__: [YOLOv3](https://github.com/qqwweee/keras-yolo3) to detect objects on each of the video frames. - 用自己的数据训练YOLOv3模型

* __Tracking__: [Deep_SORT](https://github.com/nwojke/deep_sort) to track those objects over different frames.

*This repository contains code for Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT). We extend the original SORT algorithm to integrate appearance information based on a deep appearance descriptor. See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.*

## Quick Start

__0.Requirements__

    pip install -r requirements.txt
    
__1. Download the code to your computer.__
    
    git clone https://github.com/xiaoxiong74/Object-Detection-and-Tracking.git
    
__2. Download [[yolov3.weights]](https://pjreddie.com/media/files/yolov3.weights)__ and place it in `deep_sort_yolov3/model_data/`

*Here you can download my trained [[yolo-spp.h5]](https://pan.baidu.com/s/1DoiifwXrss1QgSQBp2vv8w&shfl=shareset) - `t13k` weights for detecting person/car/bicycle,etc.*

__3. Convert the Darknet YOLO model to a Keras model:__
```
$ python convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo.h5
``` 
__4. Run the YOLO_DEEP_SORT:__

```
$ python main.py -c [CLASS NAME] -i [INPUT VIDEO PATH]

$ python main.py -c person -i ./test_video/testvideo.avi
```

__5. Can change [yolo.py] `__Line 129__` to your tracking object__

```
       if predicted_class != 'person' and predicted_class != 'bicycle':
           print(predicted_class)
           continue
```
and change [main.py] `__Line 108__` and  `__Line 123__` to your tracking object__
```
            # __Line 108__`分别保存每个类别的track_id
            if class_name == ['person']:
                counter1.append(int(track.track_id))
            if class_name == ['bicycle']:
                counter2.append(int(track.track_id))
                
            # __Line 123__当前画面中的每个类别单独计数             
            if class_name == ['person']:
                i1 = i1 +1
            else:
                i2 = i2 +1
            
```
and change some desciption in [main.py] `__Line 146__` and `__Line 175__` 


## Train on Market1501 & MARS
*People Re-identification model*

[cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning) for training a metric feature representation to be used with the deep_sort tracker.

## Citation

### YOLOv3 :

    @article{yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Redmon, Joseph and Farhadi, Ali},
    journal = {arXiv},
    year={2018}
    }

### Deep_SORT :

    @inproceedings{Wojke2017simple,
    title={Simple Online and Realtime Tracking with a Deep Association Metric},
    author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
    booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
    year={2017},
    pages={3645--3649},
    organization={IEEE},
    doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
    title={Deep Cosine Metric Learning for Person Re-identification},
    author={Wojke, Nicolai and Bewley, Alex},
    booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
    year={2018},
    pages={748--756},
    organization={IEEE},
    doi={10.1109/WACV.2018.00087}
    }
    
## Reference
#### Github:deep_sort@[Nicolai Wojke nwojke](https://github.com/nwojke/deep_sort)
#### Github:deep_sort_yolov3@[Qidian213 ](https://github.com/Qidian213/deep_sort_yolov3)



