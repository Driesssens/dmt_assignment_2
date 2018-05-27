import matplotlib.pyplot as plt
import numpy

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
	f, axarr = plt.subplots(3, 18,figsize=(40,25))
	i = 0
	j = 0
	for elem in list(full_training_set):
	    if elem == "date_time":
	        continue
	    full_training_set.boxplot(column=elem,grid=False, ax=axarr[j,i], rot=0, fontsize=4)
	    i+= 1
	    if i == 18:
	        i = 0
	        j += 1

	f.delaxes(axarr[j,i])
	plt.show()
	plt.savefig('boxplot.png', bbox_inches='tight')

	plt.clf()

def missin_Val():