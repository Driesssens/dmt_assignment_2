from super_experiment import SuperExperiment
from super_ctr_cvr_experiment import SuperCtrCvrExperiment
from top_hustinx import HustinxExperimentTop


def run_some_experiments():
    HustinxExperimentTop().run_boosted_experiment(missing_values_old_style=True, reset_data=True)


run_some_experiments()
