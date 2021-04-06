# 자율주행 인지 Report

# **0. 들어가기 전**

기계학습(Machine Learning)에 있어서, 자율주행 인지에 있어서 데이터는 굉장히 중요한 요소입니다. 그렇기에 데이터를 수집하기 위해서 많은 기업들이 노력을 들이고 있습니다. 알고리즘을 설계하고 테스트하는데 있어서 데이터를 활용할 수 있을 것입니다. 하지만 시간대별, 위치 등 다양한 정보를 포함해야하기 때문에 이를 수집하기 위해 들어가는 돈은 천문학적입니다. 기술의 발전을 위해서 공개된 데이터셋이 존재합니다. 이러한 데이터 셋에 대해 알아봅시다.

# **1. 자율주행 인지 관련 데이터 셋 (Data Set)**

## **1.1 BDD100K**

bdd100k 공식 사이트 : [https://www.bdd100k.com/](https://www.bdd100k.com/)

깃허브 주소 : [https://github.com/bdd100k/bdd100k](https://github.com/bdd100k/bdd100k)

### 세부 요소 및 예시

Berkeley DeepDrive Industry Consortium 이 주최하고 후원한 프로젝트인 BDD100K는 Computer Vision 연구를 위한 다양한 개방형 주행 비디오 데이터 셋(Data Set)입니다.

100k, 즉 100,000개의 동영상으로 구성된 데이터 셋으로 각 비디오는 40초 길이, 720p 및 30fps 로 제공됩니다.

이미지에 쉽게 주석처리하기 위해서 수직 차선은 빨간색, 평행 차선은 파란색으로 표시되어 있습니다. 이 데이터 셋은 버스, 신호등, 교통 표지, 사람 등 100,000개의 주석이 달린  2D Bounding Box가 포함되어 있습니다.

또한, 각 영상에서 10초마다 프레임에 대한 주석을 제공하고 있습니다. 이 주석은 이미지 태그, 운전 가능 지역, 차선 표시 및 여러 방법으로 라벨링이 되어있습니다. 이러한 주석을 통해 데이터 및 객체의 통계에 대한 이해를 도울 수 있습니다.

![%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B2%E1%86%AF%E1%84%8C%E1%85%AE%E1%84%92%E1%85%A2%E1%86%BC%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8C%E1%85%B5%20Report%20ba9e7abc14cb43d1977b6992de82c659/Untitled.png](assets/Untitled.png)



![%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B2%E1%86%AF%E1%84%8C%E1%85%AE%E1%84%92%E1%85%A2%E1%86%BC%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8C%E1%85%B5%20Report%20ba9e7abc14cb43d1977b6992de82c659/Untitled%201.png](assets/Untitled%201.png)



### **활용 예**

1. **도로 물체 라벨링을 통한 보행자 인식**

    위 데이터를 통해서 도로 물체 감지, 운전 가능 지역 확인, 차선 구분 등 인지 작업을 진행할 수 있습니다. 그렇기에 특정 영역(Domain)을 연구하기 위해서 사용할 수 있습니다. 이 데이터 셋을 통하여 자율주행 차량의 거리에서 보행자를 감지하고 회피하는 알고리즘을 연구하는 데에 관심이 있다면 이 데이터 셋을 통하여 연구를 진행할 수 있습니다.

2. **차선 표시 확인을 통한 차선 인식 알고리즘** 

    차선 표시는 운전자에게 있어 가장 중요한 요소 중 하나입니다. 자율주행 차량이 GPS 데이터를 이용할 수도 있지만, 카메라 정보를 통해 얻는 차선 정보를 사용하기도 합니다. 이 데이터 셋에서 수직 차선과 수평 차선을 서로 다른 색상으로 구분하여 표시하고 있습니다. 그렇기에 차선 인식 알고리즘 개선에서 사용할 수 있습니다.

3. **운전 가능 영역 확인**

    차선 표시 뿐만 아니라 실제로 주행이 가능한 영역인지 확인을 해야합니다. 이 데이터 셋에서 운전 가능 영역을 2가지로 나누어 표시합니다. 빨간색으로 표시된 영역은 현재 차량이 도로 우선 순위를 갖는 영역이지만, 파란색으로 표시된 영역은 우선 순위가 다른 차량이 더 높은 영역임을 확인할 수 있습니다.

## 1.2 nuScenes

nuScenes 공식 사이트 : [https://www.nuscenes.org/](https://www.nuscenes.org/)

이 데이터셋은 3D 객체 주석이 포함된 대규모 자율주행 차량의 데이터셋입니다. 자율주행 차량의 센서 제품군을 모두 갖추어 데이터를 제공하는데, 장착된 센서의 종류 및 상세 내용은 다음과 같습니다.

### **센서 종류 및 데이터 특징**

**센서 종류**

- LiDAR 1개
    - 390,000 번의 라이다 sweep 데이터가 있습니다.
- RADAR 5개
- 카메라 6개
    - 20초 길이의 장면 1000개가 포함되어 있으며 1,400,000 개의 카메라 이미지가 포함되어 있습니다.
- IMU
- GPS

**데이터 특징**

이 데이터셋은 가시성(Visibility), 활동(activity) 등에 대한 특징들이 포함되어 있습니다.

- 보스턴과 싱가포르에서 운행된 데이터입니다.
- 자세한 지도 정보가 포함되어 있습니다.
- 23종류의 객체에 대해서 1,400,000개의 데이터가 라벨링되어 있습니다.

### **데이터 예시**

![%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B2%E1%86%AF%E1%84%8C%E1%85%AE%E1%84%92%E1%85%A2%E1%86%BC%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8C%E1%85%B5%20Report%20ba9e7abc14cb43d1977b6992de82c659/Untitled%202.png](assets/Untitled%202.png)

nuScenes 샘플 데이터 - 좌: 라이다, 우 : 카메라

샘플 데이터로 보는 라이다 센서를 통해 얻어오는 결과 값과 카메라를 통해 라벨링된 결과가 포함된 이미지 입니다. 실제 물체를 검출하고 있음을 알 수 있습니다.

실제 센서에서 다음과 같은 형태로 데이터를 받아오고 있음을 확인할 수 있습니다.

```json
sensor {
   "token":                   <str> -- Unique record identifier.
   "channel":                 <str> -- Sensor channel name.
   "modality":                <str> {camera, lidar, radar} -- Sensor modality. Supports category(ies) in brackets.
}
```

## 1.3 Waymo 오픈 데이터셋

공식 사이트 : [https://waymo.com/open](https://waymo.com/open)

github 주소 : [https://github.com/waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset)

실제 waymo 자율주행 차량이 수집한 데이터셋입니다. 도심부터 교외, 시간대와 날씨에 따라 다양하게 수집된 결과값을 포함하고 있습니다.

### 센서 및 데이터 특징

**센서 데이터**

- 중거리 LIDAR 1개
- 단거리 LIDAR 4개
- 카메라 5개(front, sides)
- 동기화된 라이다와 카메라 데이터

등이 있습니다. 

**라벨링된 데이터**

- 20,000,000 개 이상의 프레임, 574 시간의 촬영된 데이터가 포함되어 있습니다.
- 다양한 장소에서 지도 데이터를 수집했습니다.
    - 샌프란시스코, 피닉스, LA, 디트로이트, 시애틀 등 여러 장소의 영상이 포함되어 있습니다.
- 각 객체의 라벨링(3D bounding boxes)가 있습니다.
    - 차량, 보행자, 자전거, 표지판을 객체 등급을 4단계로 나누어 라벨링을 했습니다.
- 1,200개 세그먼트의 LIDAR 데이터에 대한 라벨링이 되어있습니다.

### 데이터 예시

웨이모에서 제공하는 실제 샘플 데이터입니다.

![%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B2%E1%86%AF%E1%84%8C%E1%85%AE%E1%84%92%E1%85%A2%E1%86%BC%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8C%E1%85%B5%20Report%20ba9e7abc14cb43d1977b6992de82c659/Untitled%203.png](assets/Untitled%203.png)

2D 라벨링

![%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B2%E1%86%AF%E1%84%8C%E1%85%AE%E1%84%92%E1%85%A2%E1%86%BC%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8C%E1%85%B5%20Report%20ba9e7abc14cb43d1977b6992de82c659/Untitled%204.png](assets/Untitled%204.png)

3D 라벨링

위와 같은 박스에는 다음과 같은 내용이 포함됩니다.

- 물체가 LIDAR 데이터 또는 카메라 이미지에서 차량으로 인식 될 수있는 경우 생성됩니다.
- 기차와 트램(기차의 일종. 영국 등에서 운행하는 운송 수단)은 차량으로 간주되지 않으며 라벨이 부착되어 있지 않습니다.
- 오토바이와 오토바이 운전자는 차량으로 표시됩니다.

더 자세한 사항은 아래 링크를 참조 할 수 있습니다.

[https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md)

# **2. 자율 주행 인지 관련 오픈 소스(Open Source)**

코드 설명, 구성, 활용 및 결과 등 코드를 이해하는데 필요한 정보

## 2.1 LaneNet

github 링크 : [https://github.com/MaybeShewill-CV/lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection)

Tensorflow를 사용하여 차선을 예측하도록 학습된 심층신경망(DNN: Deep Neural Network) 입니다.

### 구성

![%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B2%E1%86%AF%E1%84%8C%E1%85%AE%E1%84%92%E1%85%A2%E1%86%BC%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8C%E1%85%B5%20Report%20ba9e7abc14cb43d1977b6992de82c659/Untitled%205.png](assets/Untitled%205.png)

LaneNet의 구조는 2가지로 구성됩니다. 사진 하단의 Segmentation Branch는 Binary Lane Mask(이진 차선 마스크)를 생성하도록 훈련되어 있습니다. 사진 상단의 Embedding Branch는 차선 픽셀당 N 차원의 Embedding을 생성하여 같은 차선은 가까이 뭉쳐있게 되는 반면 서로 다른 차선은 멀리 떨어지게 됩니다. 위 사진에서 컬러 맵으로 시각화되어 있는데, Segmentation Branch를 통해 배경을 제거하면 파란 점들이 클러스터링된 것을 확인할 수 있고 붉은 점으로 그 중심을 확인할 수 있습니다. 이를 통해 LaneNet의 차선 검출이 이뤄집니다.

### 코드 설명

GPU를 하나 사용한다는 가정 하에 핵심이 되는 모델의 학습 코드를 보겠습니다. 다음 스크립트를 통해 모델을 학습시킬 수 있습니다.

```bash
python tools/train_lanenet_tusimple.py
```

```python
# tools/train_lanenet_tusimple.py

# ...

def train_model():
    LOG.info('Using single gpu trainner ...')
    worker = single_gpu_trainner.LaneNetTusimpleTrainer(cfg=CFG)

    worker.train()
    return
```

해당 학습은 `trainner/tusimple_lanenet_single_gpu_trainner.py` 에서 진행됩니다. train 함수만 보겠습니다.

```python
def train(self):
        """

        :return:
        """
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE:
            try:
                LOG.info('=> Restoring weights from: {:s} ... '.format(self._initial_weight))
                self._loader.restore(self._sess, self._initial_weight)
                global_step_value = self._sess.run(self._global_step)
                remain_epoch_nums = self._train_epoch_nums - math.floor(global_step_value / self._steps_per_epoch)
                epoch_start_pt = self._train_epoch_nums - remain_epoch_nums
            except OSError as e:
                LOG.error(e)
                LOG.info('=> {:s} does not exist !!!'.format(self._initial_weight))
                LOG.info('=> Now it starts to train LaneNet from scratch ...')
                epoch_start_pt = 1
            except Exception as e:
                LOG.error(e)
                LOG.info('=> Can not load pretrained model weights: {:s}'.format(self._initial_weight))
                LOG.info('=> Now it starts to train LaneNet from scratch ...')
                epoch_start_pt = 1
        else:
            LOG.info('=> Starts to train LaneNet from scratch ...')
            epoch_start_pt = 1

        for epoch in range(epoch_start_pt, self._train_epoch_nums):
            train_epoch_losses = []
            train_epoch_mious = []
            traindataset_pbar = tqdm.tqdm(range(1, self._steps_per_epoch))

            for _ in traindataset_pbar:

                if self._enable_miou and epoch % self._record_miou_epoch == 0:
                    _, _, summary, train_step_loss, train_step_binary_loss, \
                        train_step_instance_loss, global_step_val = \
                        self._sess.run(
                            fetches=[
                                self._train_op, self._miou_update_op,
                                self._write_summary_op_with_miou,
                                self._loss, self._binary_seg_loss, self._disc_loss,
                                self._global_step
                            ]
                        )
                    train_step_miou = self._sess.run(
                        fetches=self._miou
                    )
                    train_epoch_losses.append(train_step_loss)
                    train_epoch_mious.append(train_step_miou)
                    self._summary_writer.add_summary(summary, global_step=global_step_val)
                    traindataset_pbar.set_description(
                        'train loss: {:.5f}, b_loss: {:.5f}, i_loss: {:.5f}, miou: {:.5f}'.format(
                            train_step_loss, train_step_binary_loss, train_step_instance_loss, train_step_miou
                        )
                    )
                else:
                    _, summary, train_step_loss, train_step_binary_loss, \
                        train_step_instance_loss, global_step_val = self._sess.run(
                            fetches=[
                                self._train_op, self._write_summary_op,
                                self._loss, self._binary_seg_loss, self._disc_loss,
                                self._global_step
                            ]
                    )
                    train_epoch_losses.append(train_step_loss)
                    self._summary_writer.add_summary(summary, global_step=global_step_val)
                    traindataset_pbar.set_description(
                        'train loss: {:.5f}, b_loss: {:.5f}, i_loss: {:.5f}'.format(
                            train_step_loss, train_step_binary_loss, train_step_instance_loss
                        )
                    )

            train_epoch_losses = np.mean(train_epoch_losses)
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                train_epoch_mious = np.mean(train_epoch_mious)

            if epoch % self._snapshot_epoch == 0:
                if self._enable_miou:
                    snapshot_model_name = 'tusimple_train_miou={:.4f}.ckpt'.format(train_epoch_mious)
                    snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                    os.makedirs(self._model_save_dir, exist_ok=True)
                    self._saver.save(self._sess, snapshot_model_path, global_step=epoch)
                else:
                    snapshot_model_name = 'tusimple_train_loss={:.4f}.ckpt'.format(train_epoch_losses)
                    snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                    os.makedirs(self._model_save_dir, exist_ok=True)
                    self._saver.save(self._sess, snapshot_model_path, global_step=epoch)

            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                LOG.info(
                    '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} '
                    'Train miou: {:.5f} ...'.format(
                        epoch, log_time,
                        train_epoch_losses,
                        train_epoch_mious,
                    )
                )
            else:
                LOG.info(
                    '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} ...'.format(
                        epoch, log_time,
                        train_epoch_losses,
                    )
                )
        LOG.info('Complete training process good luck!!')

        return
```

코드의 흐름은 다음과 같습니다.

1. tensorflow 의 session 을 지정합니다.
    - session 은 받아온 데이터를 통해 연산의 흐름이 이루어지도록 만드는 동작입니다.
2. epoch 값에 따라 훈련을 진행하게 됩니다. 
    - 변수는 `global_configuratio/config.py` 에 정의되어 있습니다.
    - 단계별로 loss 정도를 확인하면서 개선해나가게 됩니다.
3. 학습 과정에서 스냅샷 모델을 만들어두면서 학습을 진행해나갑니다.

### 활용 및 결과

딥러닝을 이용한 오픈 소스 네트워크이기 때문에 이 모델을 그대로 사용할 수도 있습니다. 하지만 프로젝트마다 원하는 출력이 다르기 때문에 학습데이터를 LaneNet에 맞는 데이터 형식에 맞춰주고 epoch와 배치 크기 등을 조절하여 모델의 학습을 진행해야합니다. 적절한 학습이 진행되어 정확도가 원하는 만큼 높아진 모델을 가지고 실제 차선 검출 등에 활용할 수 있을 것입니다.

## 2.2 YOLO: You Only Look Once

기존 Object Detection은 sliding window 를 통한 검출을 하여 실행 속도가 느릴 뿐만 아니라 한 윈도우에 하나의 bound만 지정할 수 있기에 2가지 이상의 물체는 표시할 수 없는 문제가 있었습니다. 이에 grid 와 anchor box를 도입하여 문제를 해결한 것이 yolo 입니다.

grid 방식에서는 단순히 이미지를 구역별로 나눠 계산을 진행해 정확도가 조금 낮아지는 대신, 픽셀당 연산량을 대폭 줄입니다.

### 구성

구성(아키텍처)은 다음 이미지와 같습니다. Yolo도 CNN의 일종이기에 Convolution layer를 어떻게 이용할지 선언해줘야 합니다.

![%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B2%E1%86%AF%E1%84%8C%E1%85%AE%E1%84%92%E1%85%A2%E1%86%BC%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8C%E1%85%B5%20Report%20ba9e7abc14cb43d1977b6992de82c659/Untitled%206.png](%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B2%E1%86%AF%E1%84%8C%E1%85%AE%E1%84%92%E1%85%A2%E1%86%BC%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8C%E1%85%B5%20Report%20ba9e7abc14cb43d1977b6992de82c659/Untitled%206.png)

### 코드

yolo를 통해 만든 darknet 이라는 프레임워크 기준으로 보겠습니다.

디렉토리 구조는 다음과 같습니다.

![%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B2%E1%86%AF%E1%84%8C%E1%85%AE%E1%84%92%E1%85%A2%E1%86%BC%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8C%E1%85%B5%20Report%20ba9e7abc14cb43d1977b6992de82c659/Untitled%207.png](assets/Untitled%207.png)

**cfg/**

cfg파일의 변경 혹은 변형을 통해서, 본인만의 모델을 만들어 데이터 학습을 시킬 수 있습니다.

**src/**

src디렉토리는 YOLO사용에 필요한 C 또는 header파일이 있습니다. 예를 들어 Object에 그려지는 Bounding Box를 어디에 그릴지에 관한좌표에 대한 코드들이 있습니다.

**Makefile**

Makefile은 주로 GPU, CUDENN, OPENCV, OPENMP, DEBUG를 사용할 것인가에 관한 정보를 다루고 있습니다.

실행은 아래 코드에서 main 함수 내에서 받아오는 옵션에 따라 파싱을 진행하여 위와 같이 원하는 함수로 분기하게 됩니다. 

```c
// examples/darknet.c

if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "yolo")){
        run_yolo(argc, argv);
    } else if (0 == strcmp(argv[1], "super")){
        run_super(argc, argv);
    } else if (0 == strcmp(argv[1], "lsd")){
        run_lsd(argc, argv);
    } else if (0 == strcmp(argv[1], "detector")){
        run_detector(argc, argv);
    } else if (0 == strcmp(argv[1], "detect")){
        float thresh = find_float_arg(argc, argv, "-thresh", .5);
        char *filename = (argc > 4) ? argv[4]: 0;
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        int fullscreen = find_arg(argc, argv, "-fullscreen");
        test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
    } else if (0 == strcmp(argv[1], "cifar")){
        run_cifar(argc, argv);
    } else if (0 == strcmp(argv[1], "go")){
        run_go(argc, argv);
    } else if (0 == strcmp(argv[1], "rnn")){
        run_char_rnn(argc, argv);
    } else if (0 == strcmp(argv[1], "coco")){
        run_coco(argc, argv);
    } else if (0 == strcmp(argv[1], "classify")){
        predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
    } else if (0 == strcmp(argv[1], "classifier")){
        run_classifier(argc, argv);
    } else if (0 == strcmp(argv[1], "regressor")){
        run_regressor(argc, argv);
    } else if (0 == strcmp(argv[1], "isegmenter")){
        run_isegmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "segmenter")){
        run_segmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "art")){
        run_art(argc, argv);
    } else if (0 == strcmp(argv[1], "tag")){
        run_tag(argc, argv);
    } else if (0 == strcmp(argv[1], "3d")){
        composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
    } else if (0 == strcmp(argv[1], "test")){
        test_resize(argv[2]);
    } else if (0 == strcmp(argv[1], "nightmare")){
        run_nightmare(argc, argv);
    } else if (0 == strcmp(argv[1], "rgbgr")){
        rgbgr_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "reset")){
        reset_normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "denormalize")){
        denormalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "statistics")){
        statistics_net(argv[2], argv[3]);
    } else if (0 == strcmp(argv[1], "normalize")){
        normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "rescale")){
        rescale_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "ops")){
        operations(argv[2]);
    } else if (0 == strcmp(argv[1], "speed")){
        speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
    } else if (0 == strcmp(argv[1], "oneoff")){
        oneoff(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "oneoff2")){
        oneoff2(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "print")){
        print_weights(argv[2], argv[3], atoi(argv[4]));
    } else if (0 == strcmp(argv[1], "partial")){
        partial(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "visualize")){
        visualize(argv[2], (argc > 3) ? argv[3] : 0);
    } else if (0 == strcmp(argv[1], "mkimg")){
        mkimg(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argv[7]);
    } else if (0 == strcmp(argv[1], "imtest")){
        test_resize(argv[2]);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
```

### 활용 및 결과

yolo가 반드시 자율주행에 사용되는, 자율주행에 특화된 것은 아닙니다. 하지만 Object Detection의 관점에서 차량의 운행 과정에서 빠른 판단은 굉장히 중요한 요소 중에 하나입니다. 그렇기에 yolo를 통해서 실시간으로 검출할 수 있는 것이 많이 있습니다. 일례로 **차량의 후미등 검출**이 있을 수 있습니다. 후미등에 대한 데이터를 입력해두고 이를 검출하도록 설계해두면 빠르게 점멸하는 후미등에 대해서 몇 퍼센트의 확률로 후미등이 켜졌는지, 혹은 꺼졌는지 인지할 수 있도록 만들 수 있습니다. 이러한 모델은 자율주행에 있어서 앞 차의 상태를 판단하는데 있어 거리를 측정할 수 있는 센서와 더불어, 차량의 감속 여부 등을 확인하는데 보조적인 역할을 할 수 있을 것입니다.

# **3. 코드 실행 결과**

2.2에서 공부한 Darknet 프레임워크를 사용했습니다. 이미 학습이 된 모델을 활용하여 차량의 검출을 진행했습니다.

코드 링크 : [https://github.com/chulhee23/darknet-test/blob/master/DarkNetCarTest.ipynb](https://github.com/chulhee23/darknet-test/blob/master/DarkNetCarTest.ipynb)

### 실행환경

gpu 및 여러 패키지 환경을 통일하기 위해서 간단하게 코랩(CoLab) 환경을 사용하였습니다. 

### 코드 분석

우선 darknet 저장소를 clone 해옵니다.

```bash
!git clone https://github.com/AlexeyAB/darknet.git
```

CUDA가 정상적으로 되는지 확인하고 `darknet`을 `make` 해줍니다.

데이터를 가지고 처음부터 학습을 하기엔 무리가 있습니다. 따라서 이미 공개된 신경망을 가지고 확인하도록 하겠습니다. 아래 명령어로 [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) 주소에서 공개된 weights 파일을 다운받습니다.

```bash
!wget https://pjreddie.com/media/files/yolov3.weights
```

이제 테스트를 해보고 싶은 데이터를 기준으로 data/ 폴더에 원하는 이미지를 넣어둡니다. 저는 car.png 라는 사진을 첨부했습니다.

```bash
!./darknet detect cfg/yolov3.cfg yolov3.weights data/car.png
```

이후 예측값으로 나오는 predictions.jpg 파일은 다음과 같습니다.

![%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B2%E1%86%AF%E1%84%8C%E1%85%AE%E1%84%92%E1%85%A2%E1%86%BC%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8C%E1%85%B5%20Report%20ba9e7abc14cb43d1977b6992de82c659/Untitled%208.png](assets/Untitled%208.png)

Darknet 결과

차의 검출뿐만 아니라, 흐릿하게 가려진 차량도 57%의 확률로 트럭임을 검출해내는 성능을 확인할 수 있었습니다.