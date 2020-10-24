# Agro-detect-model
Create model which when give an image of a crop infested with disease its able to detect which type of disease

### Installation and Setup

Download [Anaconda](https://www.anaconda.com/products/individual#) for your the specific platform 

Create ***conda environment*** for Python3

```
conda create -n "environment_name" python=3.7
```

Activate the environment.

```
conda activate environment_name
```

Clone the repo

```
https://github.com/r-wambui/Agro-detect-model.git
```

Navigate to the root folder

```
cd Agro-detect-model
```

create a data directory

```
mkdir data
```

- Download the dataset [here](https://data.mendeley.com/datasets/tywbtsjrjv/1). I'm using the dataset without augmentation and uzip the files. 

- Rename the folder to ```dataset```.

- We need to split this dataset into train and test. On the terminal run

```
python split_data.py
```

- This will create two folders train and val.

- Run ```jupyter notebook``` then open Agro-detect.ipynb

- You can now interact with the model and follow the tutorial [Build a Simple Crop Disease Detection Model with PyTorch](https://r-wambui.github.io/Agro-detect-model/).