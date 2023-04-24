# Colab Trainable Code for HLTR

### Description
Colab trainable code for [HLTR](https://github.com/venusasadude/Handwritten-Line-Text-Recognition-using-Deep-Learning-with-Tensorflow) model.


#### Dataset Used
* IAM dataset download from [here](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
* Only needed the lines images and lines.txt (ASCII).
* Place the downloaded files inside data directory  


To clone git repository
```markdown
$ !git clone https://github.com/ahmadraza-infento/HLTR_ENG_COLAB
```

To download dataset
```markdown
$ %cd HLTR_ENG_COLAB/
$ %mkdir data
$ %mkdir model
$ %cd data
$ !wget -O dataset.zip dataset_downloadable_link
$ !unzip dataset.zip
$ %cd ../

```
You can download self-dataset using link: [self dataset](https://drive.google.com/u/0/uc?id=1iXrh4cYIX7TilAVbuJ6jePh8oOroQmkP&export=download).

To install requirements, run:
```markdown
$ !pip install -r requirements.txt
```

Imports
```markdown
from helpers import train
from model import HLTRModel
from data_loader import DataLoader
```

Training setup
```markdown

imgDir = "data/self_lines"
labelFile = "data/lines.txt"
batchSize = 10
imgSize = (800, 64)
maxTextLen = 100

modelDir = "model/"
snapDir = "model/snapshot"

data_loader = DataLoader(imgDir, labelFile, batchSize, imgSize, maxTextLen, load_aug=True)
hltr_model = HLTRModel(data_loader.charList, modelDir=modelDir, snapDir=snapDir, 
                       batchSize=batchSize, imgSize=imgSize, maxTextLen=maxTextLen)
```

Training
```markdown
$ train(hltr_model, data_loader, modelDir)
```

# Code Source
Original repostitory can be found [here](https://github.com/venusasadude/Handwritten-Line-Text-Recognition-using-Deep-Learning-with-Tensorflow)
