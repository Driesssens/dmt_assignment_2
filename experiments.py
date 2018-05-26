from super_experiment import SuperExperiment
from super_ctr_cvr_experiment import SuperCtrCvrExperiment


def run_some_experiments():
    SuperExperiment().run_full_experiment(reset_data=True)
    SuperCtrCvrExperiment().run_full_experiment(reset_data=True)


run_some_experiments()
