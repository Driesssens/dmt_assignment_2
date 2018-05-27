#from top_hustinx import HustinxExperimentTop
from deGroot2_experiment import DeGroot2Experiment

def run_final_experiments():
    DeGroot2Experiment().run_final_experiment(missing_values_old_style=True)


run_final_experiments()
