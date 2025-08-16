# yoloV5_ultralytics
about installation, setup and execution of yolov5

# yolov5
about installation, setup and execution of yolov5 
README - YOLOv5 Object Detection Project
=======================================

This project trains a YOLOv5 model to detect objects such as Heavy Gun, Person, and Pistol.

## Project Structure

  - mydata/
    - images/
      - train/
      - valid/
      - test/
    - labels/
      - train/
      - valid/
      - test/
    - data.yaml
## ** update Graphics Driver (Game Ready Driver) and keep Good airflow for device ** ##
##  Environment Setup:

    ## Install CUDA (12.1 Recommended)
    ## Install Miniconda if you don't have it, then:
    conda create -y -n yolov5 python=3.10
    conda activate yolov5

    # Install CUDA-enabled PyTorch (pick the CUDA build that matches your driver)
    # Go to pytorch.org > "Install" and copy the pip command for CUDA. Example (CUDA 12.x):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    ## Get YOLOv5
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip install -r requirements.txt

##  GPU check (Run this in your python file)
    import torch
    print("Torch CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
    print("OK" if torch.cuda.is_available() else "NOT OK")

## Steps to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Prepare dataset in the structure above and configure `data.yaml`:
   Example data.yaml:
   train:  .../mydata/images/train
   val:  .../mydata/images/valid
   test: .../dataset/mydata/images/test
   nc: 5
   names: ['Heavy Gun', '#####', 'Person', 'Pistol', 'pistol', 'rifle']

3. Train the model:
   python train.py --img 640 --batch 16 --epochs 100 --patience 20 --data ".../mydata/data.yaml" --weights yolov5s.pt --device 0 --workers 1 --cache

4. Run detection on source video:
   python detect.py --weights runs/train/exp/weights/best.pt --source input.mp4

5. Check training logs in `results.csv` and training plots in `results.png` under `runs/train/exp/`.

## Notes
- Safe GPU temperature: ≤ 80 °C
- Safe CPU temperature: ≤ 85 °C
- Reduce batch size if memory or temperature issues occur.

*** Credits & Attributions
This project includes and builds upon work from the following sources:

YOLOv5 by Ultralytics – Licensed under the GNU General Public License v3.0.

PyTorch – An open source deep learning framework, licensed under the BSD‑style license.

Any other external code, models, or datasets used are the property of their respective authors and redistributed here under their original licenses.

All modifications, configurations, and additions in this repository are documented in the project files.
