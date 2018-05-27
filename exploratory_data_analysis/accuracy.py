import matplotlib.pyplot as plt
import numpy

acc_tr_super = []
acc_val_super = []
acc_tr_base = []
acc_val_base = []

with open("Superrun_full_20180526060437/" + 'training_performance_per_epoch.txt','r') as model_fn:
    for line in model_fn: 
        line = line.split("[")[1].split("]")[0].split(", ") #or some other preprocessing
        for number in line:
        	acc_tr_super.append(float(number)) #storing everything in memory!

with open("Superrun_full_20180526060437/" + 'validation_performance_per_epoch.txt','r') as model_fn:
    for line in model_fn: 
        line = line.split("[")[1].split("]")[0].split(", ") #or some other preprocessing
        for number in line:
        	acc_val_super.append(float(number)) #storing everything in memory!


with open("Base_run_full_20180521120423/" + 'training_performance_per_epoch.txt','r') as model_fn:
    for line in model_fn: 
        line = line.split("[")[1].split("]")[0].split(", ") #or some other preprocessing
        for number in line:
        	acc_tr_base.append(float(number)) #storing everything in memory!


with open("Base_run_full_20180521120423/" + 'validation_performance_per_epoch.txt','r') as model_fn:
    for line in model_fn: 
        line = line.split("[")[1].split("]")[0].split(", ") #or some other preprocessing
        for number in line:
        	acc_val_base.append(float(number)) #storing everything in memory!


# summarize history for accuracy
plt.plot(acc_tr_super[:-50])
plt.plot(acc_val_super[:-50])
plt.plot(acc_tr_base[:-50])
plt.plot(acc_val_base[:-50])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc_tr_super', 'acc_val_super', 'acc_tr_base', 'acc_val_base'], loc='upper left')
plt.show()

plt.savefig('output/acc_plot.png', bbox_inches='tight')
plt.clf()

f, axarr = plt.subplots(4, 15)
i = 0
j = 0
for elem in list(full_training_set):
    if elem == "date_time":
        continue
    full_training_set.boxplot(column=elem,grid=False, ax=axarr[j,i], rot=90, fontsize=8)
    i+= 1
    if i == 15:
        i = 0
        j += 1

plt.show()
plt.savefig('boxplot.png', bbox_inches='tight')
plt.clf()

# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()