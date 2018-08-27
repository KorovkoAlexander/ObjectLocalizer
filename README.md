Object Localizer
===================


This project is a test case from Avito.
You re given with a few labeled images and you re to make a neural network to predict the bounding box
containing the main object on the photo.
The IoU baseline is 0.8, averaged by all photos in your test.
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659881-6c6ac200-aa0e-11e8-9491-4327128777a1.png)

----------


Architecture
-------------
Taking the inspiration from <a>https://github.com/DemonFuneral/Image-Object-Localization</a>, i tried to train vanila resnet18 with one additional fullyconnected layer, reducing the output size to 4.

Dataset
----------
Here <a>https://drive.google.com/drive/folders/1YUWbKclYUGtouRc5xgGuBhn65S4yzMVc</a> there is a tar archive images.tar with about 100k photos from avito.
Also with_labels.csv contains 1000 layouts, given from avito.
labeled_4491.csv contains 4491 layouts, which were labeled by me.

Training
----------
For training i used SmoothL1Loss, whitch is already implemented in pytorch.
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659880-6c6ac200-aa0e-11e8-9c54-f9c5d416951b.png)

Running Loss|Running IoU
------------|----------------
![ loss_formula](https://user-images.githubusercontent.com/35064209/44658998-70491500-aa0b-11e8-91e4-930853c33cf6.png ) | ![ loss_formula](https://user-images.githubusercontent.com/35064209/44658999-70491500-aa0b-11e8-829f-23b08952731d.png)

I ve got 0.889 target metric in my test.
Installation
------

    pip install -e .

Train
------
Perhaps you will need to install tensorboardX first.
To train you own model you should have the same structure of your dataset:

 dataset_path:
 - images
 - train.csv
 - val.csv

If necessary feel free to open train.py and change the names of train and val csv.
To run the code, execute: 

    python -m object_localizer.src.train <dir_to_save_the model> --batch_size 20 --show_interval 10 --dataset_path <path_to_the_dataset> --epochs 120

Test images|Test images
------------|----------------
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659000-70491500-aa0b-11e8-97e9-2c5b1b0632a0.png) | ![ loss_formula](https://user-images.githubusercontent.com/35064209/44659001-70491500-aa0b-11e8-8302-1d71a980110f.png)
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659002-70491500-aa0b-11e8-8978-f5c32ae920b4.png) | ![ loss_formula](https://user-images.githubusercontent.com/35064209/44659003-70e1ab80-aa0b-11e8-82b6-63f6e541492d.png)
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659004-70e1ab80-aa0b-11e8-992d-bec197ed1d33.png) | ![ loss_formula](https://user-images.githubusercontent.com/35064209/44659005-70e1ab80-aa0b-11e8-9f99-52c8e3393e27.png)
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659006-70e1ab80-aa0b-11e8-97db-2e14963b0338.png) | ![ loss_formula](https://user-images.githubusercontent.com/35064209/44659008-717a4200-aa0b-11e8-9b41-f2c4e9b31053.png)
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659009-717a4200-aa0b-11e8-936e-359919de293b.png) | ![ loss_formula](https://user-images.githubusercontent.com/35064209/44659010-717a4200-aa0b-11e8-96c6-88bc2016a4f9.png)
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659011-717a4200-aa0b-11e8-8af8-076ce4dd8430.png) | ![ loss_formula](https://user-images.githubusercontent.com/35064209/44659012-7212d880-aa0b-11e8-84ae-136f0f11d508.png)
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659015-7212d880-aa0b-11e8-9512-e32f99f397bb.png) | ![ loss_formula](https://user-images.githubusercontent.com/35064209/44659014-7212d880-aa0b-11e8-8ec5-c4123a9c7e44.png)
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659016-7212d880-aa0b-11e8-9691-bbd397ba92df.png) | ![ loss_formula](https://user-images.githubusercontent.com/35064209/44659017-72ab6f00-aa0b-11e8-9f2f-d41b1355251a.png)
![ loss_formula](https://user-images.githubusercontent.com/35064209/44659018-72ab6f00-aa0b-11e8-8568-a188fb384a40.png) 
