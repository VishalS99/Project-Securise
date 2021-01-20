# Project-Securise

An ANPR program for the campus

- Dataset available at: https://drive.google.com/drive/folders/1s-8IqlRPaawEEkk67yKc-l2SWr9wVPro?usp=sharing
- Move dataset dir to Dataset
- Trained Model: https://drive.google.com/drive/folders/1l-PkRb6YYTf4E36te2cScDGjCi-V5uQ4?usp=sharing
- If you have an NVIDIA gpu, install CUDA and CuDNN for gpu acceleration.

- The main.py File in Lib/Detection is the Detection Model with pretrained Yolo weights.

- Download the weights from the below link.
  yolov3.weights can be downloaded from here : https://pjreddie.com/media/files/yolov3.weights

- This Model has to be retrained in future so as to improve Indian Vehicle Detection Accuracy and to add indegenious vehicles.
- Run <code>python Lib/main.py</code> to start the program!

Currently the vehicle color, type, time of entry, number plate detection and character segmentation are analysed from the image fed to the program.
