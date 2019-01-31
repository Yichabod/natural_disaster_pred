from matplotlib import pyplot as plt

source = 'Results/'

numbs = ['0-3','4-7','8-11','12-15','16-19']

train_accurs = {}
validation_accurs = {}

for numb in numbs:
    file = open('Results/validation_accurs_'+numb+'.txt', 'r')
    for line in file:
        try:
            epochs = int(float(line[:-1]))
        except(ValueError):
            if line.find('Train')!=-1:
                accur = float(line.split(':')[-1])
                try:
                    train_accurs[epochs].append(accur)
                except(KeyError):
                    train_accurs[epochs] = [accur]
                    
            elif line.find('Eval')!=-1:
                accur = float(line.split(':')[-1])
                try:
                    validation_accurs[epochs].append(accur)
                except(KeyError):
                    validation_accurs[epochs] = [accur]

xs = []
train_ys = []
val_ys = []
for key in train_accurs:
    print(key, sum(train_accurs[key])/len(train_accurs[key]))
    xs.append(key)
    train_ys.append(sum(train_accurs[key])/len(train_accurs[key]))

for key in validation_accurs:
    print(key, sum(validation_accurs[key])/len(validation_accurs[key]))
    val_ys.append(sum(validation_accurs[key])/len(validation_accurs[key]))

plt.plot(xs, train_ys, label='Traning set') #, title = 'Validation accuracy as a function of epochs'
plt.plot(xs, val_ys, label='Validation set')
plt.legend()
plt.ylabel('Balanced accuracy')
plt.xlabel('Epochs trained')
plt.show()
