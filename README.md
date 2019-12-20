# Facial-expression-recognition

## 1. Dataset
  Dataset is FER2013. You can find it on Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
  
  You can run ``` python download.py ```. This script will download the data and extract to data/fer2013/fer2013.csv or you can download and put file fer2013.csv to this folder.
  
  The dataset contains ~35k images(from pixels) from 7 classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.
  
  Problems: Class imbalance - Use class weight and data augmentation.
  
## 2. Preprocess
  On each image convert sequence of 2304 pixels to (48, 48) numpy array and grayscale to 3 channels. 
  
  You can preprocess and save train set, dev set and test set to file and next time just load it as numpy array. Example:
  
  ``` python setup.py --save True ```

## 3. Train model
  For this classification problem i use VGG19, ResNet(ResNet18, ResNet34, ResNet50, ResNet101, ResNet152).
  
  Example of training a model and chosing hyperparameters:
  
  ``` python train_model.py --model ResNet18 --epochs 100 --lr 0.01 --batch_size 128 ```
  
  This will save the checkpoint with best val_acc in model folder. Load it later by load_weights.
  
  The best accuracy~68% on test set(Kaggle winner~71%).
  
  Weights for pretrained model on imagenet dataset might reduce training time.

## 4. Real time facial expression recognition with OpenCV
  This part inspired from: https://github.com/omar178/Emotion-recognition.
  After saving checkpoint of the model I use OpenCV for deploying. Here I have pretrained model ResNet18 after training some epochs. 
  Of course you can do this for another model. Just run:
  
  ``` python real_time_video.py --model ResNet18 ```
  
  Here it is
  
  ![alt text](https://github.com/Cris-Nguyen/Facial-expression-recognition/blob/master/img/happy.png)

  ![alt text](https://github.com/Cris-Nguyen/Facial-expression-recognition/blob/master/img/neutral.png)
  
  ## Enjoy :)
