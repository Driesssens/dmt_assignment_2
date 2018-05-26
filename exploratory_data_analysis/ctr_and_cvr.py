import pandas as pd
import json


def compute_ctr_and_cvr():
    data = pd.read_csv("../data/training_set_VU_DM_2014.csv")

    prop_ids = data['prop_id'].unique()

    ctr = {}
    cvr = {}

    # length = len(prop_ids)

    for i, prop_id in enumerate(prop_ids):
        # print "{}/{}".format(i, length)
        relevant_rows = data.loc[data['prop_id'] == prop_id]
        clicks = relevant_rows['click_bool']
        bookings = relevant_rows['booking_bool']

        ctr[str(prop_id)] = clicks.mean()
        cvr[str(prop_id)] = bookings.mean()

    with open('ctr.json', 'w+') as fp:
        json.dump(ctr, fp)

    with open('cvr.json', 'w+') as fp:
        json.dump(cvr, fp)

    print ctr
    print cvr


compute_ctr_and_cvr()
