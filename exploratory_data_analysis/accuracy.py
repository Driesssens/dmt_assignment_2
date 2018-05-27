import matplotlib.pyplot as plt
import numpy
import pandas

def plot_acc():
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



def plot_box():
    full_training_set = pandas.read_csv('../data/training_set_VU_DM_2014.csv')
    print("Dataset loaded")
    plt.figure()
    f, axarr = plt.subplots(1, 8,figsize=(30,8))
    i = 0
    j = 0
    for elem in list(full_training_set)[8:]:
        if elem == "date_time":
            continue
        full_training_set.boxplot(column=elem,grid=False, ax=axarr[i], rot=0, fontsize=12)
        axarr[i].xaxis.set_ticklabels([])
        axarr[i].set_title(elem, fontsize=18)
        i+= 1
        if i == 8:
            break
            i = 0
            j += 1



    plt.savefig('output/boxplot.png')

    plt.clf()

def missin_Val():
    full_training_set = pandas.read_csv('../data/training_set_VU_DM_2014.csv')
    print("Dataset loaded")
    plt.figure(figsize=(1000,20))
    percentage = full_training_set.isnull().sum()/len(full_training_set) * 100
    percentage2 = percentage.sort_values().to_frame()

    percentage2.columns = ["percentage missing"]
    percentage2["Feature name"] = percentage2.index
    percentage2.plot.bar(x='Feature name', color='k',width=0.9)
    plt.ylabel("percentage missing")
    plt.tight_layout()
    plt.tick_params(labelsize=7)
    plt.legend().set_visible(False)
    plt.yticks(numpy.arange(0, 120, step=20),['{:3.2f}%'.format(float(x)) for x in numpy.arange(0, 120, step=20)])
    plt.savefig('output/missing_values.png', bbox_inches='tight')

    plt.clf()


#plot_acc()
plot_box()
#missin_Val()