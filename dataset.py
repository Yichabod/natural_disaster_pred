import os
import numpy as np
from PIL import Image

def create_dataset(source_folder, eval_sessions):
    train_xs=[]
    train_ys=[]
    eval_xs = []
    eval_ys = []

    k=0
    for event in os.listdir(source_folder):
        name, month, day, year = event.split('_')
        date = int(year+month+day)
        before = []
        after = []
        print(event)
        for folder in os.listdir(source_folder+'/'+event):
            #print(folder)
            try:
                image_date = int(folder.split('T')[0])
            except(ValueError):
                print("Invalid name: ", folder)
                continue
            arrays = []
            for filename in os.listdir(source_folder+"/"+event+"/"+folder):
                #print(filename)
                if filename.split('.')[-1] == 'tif':
                    img = Image.open(source_folder+"/"+event+"/"+folder+"/"+filename)
                    arrays.append(np.expand_dims(np.array(img), axis=2))
                    img.close()
                    #print(np.shape(arrays[-1]))
            a = np.concatenate(arrays, axis=2)
            extra1 = (a.shape[0]-750)//2
            extra2 = (a.shape[1]-750)//2
            a = a[extra1:750+extra1,extra2:750+extra2]
            if image_date > date:
                after.append(a)
            elif image_date < date:
                before.append(a)
        print(len(before), len(after))
                
        for i in range(len(before)):
            if k in eval_sessions:
                for j in range(len(before)):
                    if j>=i:
                        eval_xs.append(np.concatenate([np.expand_dims(before[i],axis=0),
                                                       np.expand_dims(before[j],axis=0)],axis=0))
                        eval_ys.append([0])
                for j in range(len(after[:3])):
                    eval_xs.append(np.concatenate([np.expand_dims(before[i],axis=0),
                                                   np.expand_dims(after[j],axis=0)],axis=0))
                    eval_ys.append([1])
                    
            else:
                for j in range(len(before)):
                    if j>=i:
                        train_xs.append(np.concatenate([np.expand_dims(before[i],axis=0),
                                                       np.expand_dims(before[j],axis=0)],axis=0))
                        train_ys.append([0])
                for j in range(len(after[:3])):
                    train_xs.append(np.concatenate([np.expand_dims(before[i],axis=0),
                                                   np.expand_dims(after[j],axis=0)],axis=0))
                    train_ys.append([1])
        k+=1


    train_xs = np.concatenate([np.expand_dims(i,axis=0) for i in train_xs], axis=0)
    train_ys = np.array(train_ys)
    eval_xs = np.concatenate([np.expand_dims(i,axis=0) for i in eval_xs], axis=0)
    eval_ys = np.array(eval_ys)
    np.save('datasets/train_xs.npy', train_xs)
    np.save('datasets/train_ys.npy', train_ys)
    np.save('datasets/eval_xs.npy', eval_xs)
    np.save('datasets/eval_ys.npy', eval_ys)
    print("Data saved!")

def load():
    train_xs = np.load('datasets/train_xs.npy')
    train_ys = np.load('datasets/train_ys.npy')
    eval_xs = np.load('datasets/eval_xs.npy')
    eval_ys = np.load('datasets/eval_ys.npy')

    return train_xs, train_ys, eval_xs, eval_ys
