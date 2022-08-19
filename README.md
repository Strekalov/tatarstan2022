<h1 align="center">Решение N-го места для соревнования <a href="https://hacks-ai.ru/championships/758258">Региональный чемпионат республики Татарстан</a> 

## Для воспроизведения результатов выполните следующие действия:

#### Конвертим разметку в формат yolov5:
```
python convert_bad_labels.py
```

#### Запускаем обучение для каждой модели
```
python train.py --img 1600 --epoch 100 --batch 2 --data custom.yaml --hyp hyp_evolve.yaml --weights yolov5x6.pt --name 1600 --cache disk 

python train.py --img 1900 --epoch 100 --batch 2 --data custom.yaml --hyp hyp_evolve.yaml --weights yolov5x6.pt --name 1900 --cache disk 

python train.py --img 2200 --epoch 100 --batch 2 --data custom.yaml --hyp hyp_evolve.yaml --weights yolov5x6.pt --name 2200 --cache disk 
```

#### Генерируем предсказанные лейблы 

```
python detect.py --weights runs/train/1600/weights/best.py runs/train/1900/weights/best.py runs/train/2200/weights/best.py --img 2200 --augment --save-txt --data custom.yaml --conf-thres 0.02 --iou-thres 0.4 --name ensemble 
```


#### Создаём сабмишн
```
python create_submission.py
```