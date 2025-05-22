WHISPER

скачать датасет [здесь](https://kaggle.com/datasets/16e7fa2cdadd946fb5c4e8b9ef9888a2b385acc2e5709be8c3d487695f7f0801), разархивировать и вложить в папку data

скачать веса [здесь](https://drive.google.com/file/d/152FBgvGR7o-Au17gbd80RCtLuGDXWvfZ/view?usp=sharing), вложить в папку data/weights

чтобы запустить evaluation MOS на Whisper+bert ensemble на нормализированном SOMOS,

```
python evaluate.py 
```

чтобы запустить evaluation MOS на Whisper+bert ensemble на не нормализированном SOMOS,

```
python evaluate.py --dataset_format presplit --data_path data/archive/update_SOMOS_v2/update_SOMOS_v2/training_files_with_SBS/training_files/full/valid_mos_list.txt
```

чтобы запустить evaluation на Whisper+bert ensemble на русском не забыть маркер 

```
--text_model DeepPavlov/rubert-base-cased
```

чтобы запустить evaluation SBS на Whisper+bert ensemble,

```
python evaluate.py --dataset_format presplit --data_path data/archive/update_SOMOS_v2/update_SOMOS_v2/training_files_with_SBS/training_files/full/valid_mos_list.txt --metric SBS  
```

```
python -m unittest discover tests
```
чтобы провести тесты
