# Poetic Text Generator
The **Poetic Text Generation** repository is a collection of the following two files:
* **Poetic_text_generation.py**
* **Poetic_Text_Generation_notebook.ipynb** which is a colab notebook consisting of outputs as well.

## Overview
This program generates poetic texts using Recurrent Neural Networks. To learn more about RNN, check out [here](https://www.ibm.com/cloud/learn/recurrent-neural-networks)!
* You can access the ```shakespeare.txt``` used to train the model from this [link](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt). This file consists of an enormous amount of poems and texts written by Shakespeare which will be used by the Neural Network to train itself and thereby generate Poetic texts like Shakespeare kinda! :D

* The ```sample``` function basically just picks one of the characters from the output. As parameters, it  takes the result of the prediction and a temperature. This temperature indicates how risky the pick shall be. If we have a high temperature, we will pick one of the less likely characters. A low temperature will cause a conservative choice.

* The ```generate_text``` function is the final function of our script which takes ```length``` (which corresponds to the length of the text you want the program to generate) and ```temperature``` as arguments and generates the final text. The ```model.predict()``` function predicts the likelihoods of the next characters.
# Installation
## Using Git
Type the following command in your Git Bash:

- For SSH:
```git clone git@github.com:Aditi-Asati/Poetic-Text-generation.git```
- For HTTPS: ```git clone https://github.com/Aditi-Asati/Poetic-Text-generation.git```

The whole repository would be cloned in the directory you opened the Git Bash in.

## Using GitHub ZIP download
You can alternatively download the repository as a zip file using the GitHub **Download ZIP** feature. 

*External modules used-*
- tensorflow 


Run the command ```pip install -r requirements.txt``` to install all these dependencies at once.

You are good to go!