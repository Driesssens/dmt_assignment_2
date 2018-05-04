import pandas
import numpy


def pandas_tuple_to_letor_row(tuple):
    qid = tuple.srch_id
    relevance = tuple.click_bool
    if tuple.booking_bool == 1: relevance = 5

    string = "{} qid:{} ".format(relevance, qid)

    feature_number = 1
    for name, value in tuple._asdict().iteritems():
        if name not in ['Index', 'srch_id', 'booking_bool', 'click_bool', 'date_time', 'gross_bookings_usd', 'position']:
            string += '{}:{} '.format(feature_number, value if not numpy.isnan(value) else '0.000000')
            feature_number += 1

    return string


def output_data_frame_as_letor_file(data_frame, file_name):
    with open(file_name, 'w') as output_file:
        for row in data_frame.itertuples():
            output_file.write(pandas_tuple_to_letor_row(row) + '\n')
