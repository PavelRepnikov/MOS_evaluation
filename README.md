1. `src.models` - классы для наших основных моделей. Нужно реализовать все методы из `BaseModel`.
2. `src.datasets` - классы для датасетов. На вход путь до датасета. Должно быть еще поле metric, которое показывает, какой тип датасета (mos или sbs). Для сомоса в зависимости от этого формируется датасет SBS или MOS.
3. `src.generators` - генераторы аугментированных датасетов, русская и английская версии.
4. `data` и `weights` - сюда складываем обученные данные и веса моделей. В readme обязательно написать инструкцию по тому, откуда скачивать, потому что эти папки не индексируются в гите и надо каждый раз качать заново
5. `experiments` - здесь должны быть все сохранившиеся ipynb с экспериментами по обучению.
6. `evaluate.py` - выдает метрики конкретной модели на конкретном датасете.
7. `tests` - на каждую модельку должен быть тест, который берет данные аудио (или текст + аудио) из `tests/data` и прогоняет модельку.
8. `streamlit_app.py` - фронтенд на streamlit.

Весь код должен прогоняться через flake8 (надо установить requirements.lint.txt и запустить `flake8` в терминале)

Алгоритм добавления нового кода:
1. Создаем новую ветку (для каждой задачи своя)
2. В нее коммитим
3. Пушим ветку в гитлаб
4. Создаем merge request
5. Я комментирую код, вы исправляете и так по кругу пока все не станет ок
6. Я делаю merge ветки в master и закрываю задачу



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
