import os 
from random import sample
import shutil
path = ''
store_path = ''
SIZE_per_CLASS=8
for letter in os.listdir(path):
    print (f'letter:{letter}')
    for label in os.listdir(path+f'/{letter}'):
        subfolder_flag=False
        for subfolder in os.listdir(path+f'/{letter}/{label}'): 
            if os.path.isfile(path+f'/{letter}/{label}/{subfolder}'): break
            subfolder_flag=True
            print (f'label:{label}/{subfolder}')
            files_num = len(os.listdir (path+f'/{letter}/{label}/{subfolder}'))
            take_num = min(files_num, SIZE_per_CLASS)
            print (f'File_num:{files_num}, take_num:{take_num}')
            random_files = sample(os.listdir (path+f'/{letter}/{label}/{subfolder}'),take_num)
            for file in random_files:
                shutil.copyfile(path+f'/{letter}/{label}/{subfolder}/{file}', store_path + f'/-{letter}-{label}-{subfolder}-{file}')
        if not subfolder_flag:
            print (f'label:{label}')
            files_num = len(os.listdir (path+f'/{letter}/{label}'))
            take_num = min(files_num, SIZE_per_CLASS)
            print (f'File_num:{files_num}, take_num:{take_num}')
            random_files = sample(os.listdir (path+f'/{letter}/{label}'),take_num)
            for file in random_files:
                shutil.copyfile(path+f'/{letter}/{label}/{file}', store_path + f'/-{letter}-{label}-{file}')
        else:
            subfolder_flag=False
            
            
        
    
