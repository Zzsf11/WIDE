# WIDE: Make Railway Surveillance Anomaly Detection Right
This a repo of an official implement of paper "WIDE: Make Railway Surveillance Anomaly Detection Right".
<div align="center">
  <img src="./fig/model.png" width="700" />
</div>

![image](https://github.com/user-attachments/assets/89af30e3-a787-4e0b-a17d-a036d4e27339)
## 📁Data prepation
We introduce WIDE (RailWay surveillance vIdeo anomaly DEtection dataset), the first large and diverse dataset specifically designed for anomaly detection in railway surveillance. WIDE provides: (i) high-resolution video frames from 1080p to 2160p, (ii) synthetic anomalies generated using a physics engine to ensure realistic motion, and (iii) diverse environments featuring various lighting conditions.
<div align="center">
  <img src="./fig/dataset.png" width="700" />
</div>

https://github.com/user-attachments/assets/6a61ad30-1218-409a-babd-905b8ab6367d

Download the WIDE dataset on [here](https://www.google.com.hk/search?q=%E6%8F%92%E5%85%A5pdf+github+readme&oq=%E6%8F%92%E5%85%A5pdf+github+readme&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCTEwMzAxajBqN6gCCLACAQ&sourceid=chrome&ie=UTF-8).

## 🛠️Installation
```
pip install -r requirements.txt
```

## 🚀Quick Start
```
├── data
│   ├── train
│   │   ├── grefcoco
|   |    
│   │   ├── images
|   |       ├── coco_2014
|   |       ├── saiapr_tc-12
│   │   ├── refclef
|   |       ├── instances.json
│   │   ├── refcoco
|   |       ├── instances.json
│   │       └── ...
```
