import pandas as pd
import pickle
import numpy


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = numpy.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + numpy.sum(r[1:] / numpy.log2(numpy.arange(2, r.size + 1)))
        elif method == 1:
            return numpy.sum(r / numpy.log2(numpy.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / float(dcg_max)


def ndcg_at_max(rel):
    return ndcg_at_k(rel, len(rel))


def check(csv_path, split_path, data=None):
    # full_data_set = pd.read_csv("../data/training_set_VU_DM_2014.csv")
    # with open(split_path, 'rb') as fp:
    #     qids = pickle.load(fp)
    #
    # answers = full_data_set[full_data_set.srch_id.isin(qids)]
    # answers.to_csv("ja.csv")
    answers = pd.read_csv("ja.csv")
    predictions = pd.read_csv(csv_path)
    # print predictions
    # print predictions.index
    ## first check ##
    answers_srch_ids = answers['srch_id'].tolist()
    predictions_srch_ids = predictions['SearchId '].tolist()

    if answers_srch_ids != predictions_srch_ids:
        print answers_srch_ids
        print predictions_srch_ids
        print "answers_srch_ids != predictions_srch_ids"
        raise ValueError

    ndcgs = []

    predictions_index = 0
    predictions_qid = predictions.loc[predictions_index]['SearchId ']

    answers_index = answers.index[0]
    print answers.index
    print answers_index
    answers_qid = answers.loc[answers_index]['srch_id']

    while True:
        if predictions_qid != answers_qid:
            print answers_qid
            print predictions_qid
            print "predictions_qid != answers_qid"
            raise ValueError

        predictions_prop_ids = []
        while predictions_index in predictions.index and predictions.loc[predictions_index]['SearchId '] == predictions_qid:
            predictions_prop_ids.append(predictions.loc[predictions_index]['PropertyId'])
            predictions_index += 1

        if predictions_index not in predictions.index:
            "Reached the end!"
            break

        answers_prop_ids = []
        while answers_index in answers.index and answers.loc[answers_index]['srch_id'] == answers_qid:
            answers_prop_ids.append(answers.loc[answers_index]['prop_id'])
            answers_index += 1

        if set(predictions_prop_ids) != set(answers_prop_ids):
            print answers_prop_ids
            print predictions_prop_ids
            print "set(predictions_prop_ids) != set(answers_prop_ids)"
            raise ValueError

        # print predictions_prop_ids
        rows = [answers.loc[(answers['srch_id'] == answers_qid) & (answers['prop_id'] == prop_id)] for prop_id in predictions_prop_ids]
        # rows = answers.loc[(answers['srch_id'] == answers_qid) & (answers['prop_id'].isin(predictions_prop_ids))]
        # print len(rows)
        relevances = []
        for row in rows:
            if int(row['booking_bool']) is 1:
                relevances.append(5)
            elif int(row['click_bool']) is 1:
                relevances.append(1)
            else:
                relevances.append(0)
        # relevances = [(5 if row['booking_bool'] is 1 else row['click_bool']) for row in rows]
        # print relevances
        ndcgs.append(ndcg_at_max(relevances))

        predictions_qid = predictions.loc[predictions_index]['SearchId ']
        answers_qid = answers.loc[answers_index]['srch_id']

    print ndcgs
    print sum(ndcgs) / float(len(ndcgs))


check("valid_set_values_without_rel.csv", "../splits/spl_20180518114037/mini/test_qids.pkl")
