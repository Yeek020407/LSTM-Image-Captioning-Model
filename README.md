# 🧠 SC4001 Neural Network and Deep Learning Group Project
> An LSTM-based Image Captioning model that leverages Convolutional Neural Networks for image feature extraction, capable of generating descriptive captions for input images through deep learning techniques. 

For our image captioning model, We are using an encoder to capture the features of our images with the use of transfer learning from a CNN model, ResNet101. Following which, the output will be interpreted by our decoder with the help of “soft” attention to generate the image captions. We will then be using the BLEU Score to evaluate our image caption model, comparing the caption generated by our model and the caption that was provided from the dataset.

This model is implemented using the ideas proposed by the Show, Attend, and Tell paper. The authors' original implementation can be found here.

## 📚 Dataset
####  COCO Dataset 2017 (https://cocodataset.org/#home)

## 🛠️ Implementation 
![image](https://github.com/Yeek020407/LSTM-Image-Captioning-Model/assets/98008874/9ef234e9-368f-4e4e-b77a-649dae066dcb)

Attention is a mechanism where we dynamically allocate how much attention we should give to each position in a sequence. Our model computes an “attention” score, determining which part of the image would be focused on for generating each word for the input sentence. This is shown intuitively with parts of the image being highlighed when trying to generate a word. 

## 💯 Results
![image](https://github.com/Yeek020407/LSTM-Image-Captioning-Model/assets/98008874/8eb9c260-ae73-4a8c-8207-24291240dfec)

![image](https://github.com/Yeek020407/LSTM-Image-Captioning-Model/assets/98008874/a98587e2-50aa-44cf-88de-c14bbd1e7b08)

## 🎯 Application
We also created an application for our model, allowing users to upload any images and see what the model is able generate for image captions. 
![image](https://github.com/Yeek020407/LSTM-Image-Captioning-Model/assets/98008874/56739e01-3c72-4f4c-a673-9ddaa82616cf)

## 🖊️ Contributors
* Oi Yeek Sheng [@Yeek020407](https://github.com/Yeek020407)
* Royce Teng [@sleepreap](https://github.com/sleepreap)
* Sim Oi Liang [@SimonSim8455](https://github.com/SimonSim8455)
