
import cv2
import os


def image_comparation(im1_path,im2_path):
    
    original = cv2.imread(im1_path)
    duplicate = cv2.imread(im2_path)
    if original.shape == duplicate.shape:
        
    
        difference = cv2.subtract(original, duplicate)
        b, g, r = cv2.split(difference)

        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            return 1
    else:
        return 0
def search_uniques(file_list):
    for i in file_list:
        for ii in file_list:
            if i==ii:
                pass
            else:
                if image_comparation(i,ii)==1:
                    file_list.remove(ii)
                else:
                    pass
                pass
    return file_list
    
def copy_clean_files(file_list_unique):
    for i in file_list:
        command = 'cp "'+i+'" "'+ 'unique_'+i+'"'

        os.system(command)
def file_list_path(dir_name):
    file_list_path = [dir_name+'/'+i for i in os.listdir(dir_name) if not i.startswith('.')]
    return file_list_path

if __name__ == "__main__":
    dir_name='menemen'

    file_list=file_list_path(dir_name)
    try:
        os.mkdir('unique_'+dir_name)
    except:
        pass
    file_list_unique =search_uniques(file_list)
    copy_clean_files(file_list_unique)
