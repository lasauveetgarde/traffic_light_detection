# traffic_light_detection

Перед началом работы с проектом необходимо склонировать к себе репозиторий

```
git clone https://github.com/lasauveetgarde/traffic_light_detection.git
```

## Настройка виртуального окружения

Для начала рекомендуется настроить виртуальное окружение командой `python3.8 -m venv venv38` и активировать `source ./venv38/bin/activate`

## Установка пакетов

Для установки пакетов в виртуальное окружение следует выполнить:

```
pip install -r requirements.txt
```

## Dataset

[Original source](https://github.com/Thinklab-SJTU/S2TLD)

Уменьшенную версию датасета Sall Traffic Light Dataset (S2TLD) можно выгрузить по [ссылке](https://www.kaggle.com/datasets/sovitrath/small-traffic-light-dataset-xml-format)

Текущий набор данных разделен на набор для обучения и проверки.

- Annotations: Исходные XML-аннотации набора данных.
- JPEGImages: Оригинальные изображения
- train_images: Обучающие изображения JPG, 1022 шт.
- train_annoatations: Обучающие XML-аннотации, 1022, шт.
- valid_images: Проверочные изображения JPG, 200 шт.
- valid_annoatation: XML-аннотации для проверки, 200 шт.

Классы: `red`, `yellow`, `green`, `off`,`wait_on`

Более полный датасет можно найти по [ссылке](https://www.kaggle.com/datasets/sovitrath/s2tld-720x1280-traffic-light-detection-xml-format/data)

## Обучение  модели

```
python train.py --model fasterrcnn_mobilenetv3_large_fpn --epochs 30 --imgsz 800 --data data_configs/s2tld.yaml --project-dir outputs/training/traffic_large
```
```
python train.py --dataset <path to dataset> --model <model architecture> --save-dir <directory for saving model weights>
```

Модели, которые можно использовать и находятся в папке `models`

```
[
    'fasterrcnn_convnext_small',
    'fasterrcnn_convnext_tiny',
    'fasterrcnn_custom_resnet', 
    'fasterrcnn_darknet',
    'fasterrcnn_efficientnet_b0',
    'fasterrcnn_efficientnet_b4',
    'fasterrcnn_mbv3_small_nano_head',
    'fasterrcnn_mbv3_large',
    'fasterrcnn_mini_darknet_nano_head',
    'fasterrcnn_mini_darknet',
    'fasterrcnn_mini_squeezenet1_1_small_head',
    'fasterrcnn_mini_squeezenet1_1_tiny_head',
    'fasterrcnn_mobilenetv3_large_320_fpn', # Torchvision COCO pretrained
    'fasterrcnn_mobilenetv3_large_fpn', # Torchvision COCO pretrained
    'fasterrcnn_nano',
    'fasterrcnn_resnet18',
    'fasterrcnn_resnet50_fpn_v2', # Torchvision COCO pretrained
    'fasterrcnn_resnet50_fpn',  # Torchvision COCO pretrained
    'fasterrcnn_resnet101',
    'fasterrcnn_resnet152',
    'fasterrcnn_squeezenet1_0',
    'fasterrcnn_squeezenet1_1_small_head',
    'fasterrcnn_squeezenet1_1',
    'fasterrcnn_vitdet',
    'fasterrcnn_vitdet_tiny',
    'fasterrcnn_mobilevit_xxs',
    'fasterrcnn_regnet_y_400mf'
]
```

После начала обучения вывод в терминал должен быть примерно следующим:

```
============================================================================================================================================
Total params: 18,950,729
Trainable params: 18,891,833
Non-trainable params: 58,896
Total mult-adds (G): 17.98
============================================================================================================================================
Input size (MB): 30.72
Forward/backward pass size (MB): 1831.51
Params size (MB): 75.80
Estimated Total Size (MB): 1938.04
============================================================================================================================================
18,950,729 total parameters.
18,891,833 training parameters.
Epoch: [0]  [  0/256]  eta: 0:04:13  lr: 0.000005  loss: 2.1016 (2.1016)  loss_classifier: 1.9518 (1.9518)  loss_box_reg: 0.0690 (0.0690)  loss_objectness: 0.0693 (0.0693)  loss_rpn_box_reg: 0.0115 (0.0115)  time: 0.9913  data: 0.4543  max mem: 1391
Epoch: [0]  [100/256]  eta: 0:00:28  lr: 0.000397  loss: 0.1670 (0.6169)  loss_classifier: 0.0815 (0.4989)  loss_box_reg: 0.0365 (0.0426)  loss_objectness: 0.0388 (0.0616)  loss_rpn_box_reg: 0.0128 (0.0138)  time: 0.1709  data: 0.0027  max mem: 1511
```


## Запуск примеров 

Для запуска примера распознавания дорожных знаков нужно воспользоваться скриптом `inference_video.py`

```
python inference_video.py --input input/inference_videos/1.avi --show --weights outputs/training/traffic_large/best_model.pth
```




