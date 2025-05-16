# 상공회의소 서울기술교육센터 인텔교육 6기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/kccistc/intel-06
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team projects

### 제출방법

1. 팀구성 및 프로젝트 세부 논의 후, 각 팀은 프로젝트 진행을 위한 Github repository 생성

2. [doc/project/README.md](./doc/project/README.md)을 각 팀이 생성한 repository의 main README.md로 복사 후 팀 프로젝트에 맞게 수정 활용

3. 과제 제출시 `인텔교육 6기 Github repository`에 `New Issue` 생성. 생성된 Issue에 하기 내용 포함되어야 함.

    * Team name : Project Name
    * Project 소개
    * 팀원 및 팀원 역활
    * Project Github repository
    * Project 발표자료 업로드

4. 강사가 생성한 `Milestone`에 생성된 Issue에 추가 

### 평가방법

* [assessment-criteria.pdf](./doc/project/assessment-criteria.pdf) 참고

### 제출현황

### Team: Overflower

차량 행태 인식 AI 프로젝트: 졸음운전, 과속차량 감지하여 사용자에게 알려주는 시스템

* Members
  | Name | Role |
  |----|----|
  | 송가람 | Project lead, 프로젝트를 총괄 및 차량 인식 알고리즘 구현 |
  | 황치영 | CAN 통신 및 하드웨어 시스템 구현 담당 |
  | 설유승 | data 수집 및 training |
  | 신경임 | CAN 통신 및 하드웨어 시스템 구현 담당 |

* Project Github : https://github.com/GaramSong-95/Project-DrivingAI
* 발표자료 : https://github.com/GaramSong-95/Project-DrivingAI/tree/main/presentation

### Team: AIMON

아이의 행동을 감지하여 놀아주거나 현 상태를 파악하는 시스템 구현

* Members
  | Name | Role |
  |----|----|
  | 임소연 | Project lead, 프로젝트를 총괄 및 YOLO 이미지 학습 담당 |
  | 이종범 | 원격 감지를 위한 APPLICATION 담당 |
  | 권태형 | STT, TTS를 이용한 AI CHAT BOT을 구현 담당 |
  | 강준형 | OPENPOSE를 이용한 자세, 움직임 감지 구현 담당 |
  | 이예지 | LCD GUI 및 SERVER 구현 담당 |


* Project Github : https://github.com/imso01/edge_ai_project
* 발표자료 : https://github.com/imso01/edge_ai_project/tree/main/presentation


