
Available (optional) models implemented as classifier(s):
==========================================================
vgg19
alexnet
densenet121



==========================================================
==========================================================


Command line commands/instructions 
==========================================================

Training command: 
python train.py data_dir --save_dir checkpoint_save_directory --arch "vgg19" --learning_rate 0.001 --hidden_units 512 --epochs 3 

Prediction command: 
python predict.py input checkpoint --top_k 3 --category_names cat_to_name.json 





