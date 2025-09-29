# This is the constants of the project

models = ['fruits_cnnFINAL', 'fruits_cnn4', 'fruits_cnnDA']
print('choose model:')
for i, model_name in enumerate(models):
    print(f' {i+1}.) {model_name}')
choice = 1
while choice not in [1, 2, 3]:
    choice = int(input('choice: '))
MODEL_PATH = f'models/{models[choice - 1]}.h5'
if choice == 1:
    IMAGE_SIZE = 128
else:
    IMAGE_SIZE = 64

class_names2 = ['avocado', 'banana', 'blueberry', 'cherry', 'coconut', 'corn kernel', 'date', 'eggplant',
               'kiwi', 'mango', 'olive', 'pineapple', 'pumpkin', 'strawberry', 'vanilla']
class_names1 = ['avocado', 'blueberry', 'cherry', 'coconut', 'corn kernel', 'date',
               'eggplant', 'kiwi', 'mango', 'olive', 'pineapple', 'pumpkin', 'strawberry']
class_names3 = ['avocado', 'blueberry', 'cherry', 'coconut', 'corn kernel', 'date', 'dragonfruit', 'eggplant', 'etrog',
                'kiwi', 'mango', 'muskmelon', 'olive', 'pineapple', 'pomegranate', 'pomelo', 'prikly pear',
                'strawberry', 'vanilla', 'watermelon']
class_names = eval(f'class_names{4 - choice}')
print(f'model={MODEL_PATH}\nclass_names={class_names}\nIMAGE_SIZE={IMAGE_SIZE}')
SQUARE_START = 50
SQUARE_SIZE = 500