# UoL-Final
Final Project - BSc Computer Science - University of London

 
This is repository for full code of an image Binary classification Machine learning model, for training on X-ray lung scans of either Pneumonia cases or healthy cases. 


It also has the code for a Windows tool which uses the trained model state to infer/predice new images as either Pneumonia or Healthy. The training code is inside the Jupyter Notebook, and was run on both a local GPU and Google Colab+.
The prepare_dataset.py is a module for various dataset preparation functions.

shuffled_dataset.zip contains all the images the model was trained on. It is resized and reorganized structure based on https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images


Requirements:
    Python >=3.10
    Latest PyTorch (with Cuda and supported GPU for faster training, only CPU for inference/prediction)
    Latest PyInstaller
    Matplotlib
    Seaborn
    PySide6
    Scikit-learn
    Jupyter-notebooks
    
    
There were multiple iterations and versions of the ML model based on Convolution, and also others on SVM and KNN. The last two were deemed too inaccurate to be included, especially considering the extremely high accuracy and recall of the custom CNN model.


The Windows tool, inside Windows_Tool, can be compiled into a single file by using PyInstaller with the following command:
    
    pyinstaller --noconsole --onefile --add-data="model/model_state.pth;model" Main.py


Model stats:

Evaluation on unseen testing set (800+ images):
-- Testing result -- 

Testing accuracy: 0.9578587699316629

Testing Recall: 0.9875195007800313

Testing Precision: 0.9561933534743202

Testing F1: 0.9716039907904835


    
