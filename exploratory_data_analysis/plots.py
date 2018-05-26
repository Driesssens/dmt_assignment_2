import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlations(features, correlations, vertical=True):
    correlations, features = (list(t) for t in zip(*sorted(zip(correlations, features), key=lambda tup: abs(tup[0]), reverse=vertical)))
    colors = ['red' if correlation < 0 else "blue" for correlation in correlations]
    absolute_correlations = [abs(correlation) for correlation in correlations]

    fig, ax = plt.subplots()

    if vertical:
        plt.ylim(min(absolute_correlations), max(absolute_correlations))
        ax.bar(range(len(correlations)), absolute_correlations, color=colors)
        plt.xticks(range(len(absolute_correlations)), features)
        plt.xticks(rotation=90)
    else:
        plt.xlim(min(absolute_correlations), max(absolute_correlations))
        ax.barh(range(len(correlations)), absolute_correlations, color=colors)
        plt.yticks(range(len(absolute_correlations)), features)

    plt.tight_layout()

    plt.show()


def plot_single_feature_ndcg(features, ndcgs):
    ndcgs, features = (list(t) for t in zip(*sorted(zip(ndcgs, features), reverse=True)))

    fig, ax = plt.subplots()
    plt.ylim(min(ndcgs), max(ndcgs))
    ax.bar(range(len(ndcgs)), ndcgs)
    plt.xticks(range(len(ndcgs)), features)
    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.show()


# plot_correlations(['random_bool', 'prop_location_score2', 'promotion_flag', 'prop_review_score', 'srch_length_of_stay', 'prop_starrating', 'srch_booking_window', 'comp8_rate', 'comp5_rate', 'visitor_hist_starrating', 'comp2_rate', 'prop_brand_bool', 'visitor_hist_adr_usd', 'comp3_rate', 'srch_room_count', 'srch_saturday_night_bool', 'srch_adults_count', 'comp7_rate', 'comp4_rate', 'srch_query_affinity_score', 'srch_children_count', 'comp6_rate', 'comp3_inv', 'prop_location_score1',
#                    'comp7_rate_percent_diff_new', 'orig_destination_distance', 'comp7_rate_percent_diff', 'visitor_location_country_id', 'comp6_rate_percent_diff_new', 'comp6_rate_percent_diff', 'comp5_inv', 'comp1_inv', 'site_id', 'prop_country_id', 'comp2_rate_percent_diff_new', 'comp2_rate_percent_diff', 'prop_log_historical_price', 'srch_destination_id', 'prop_id', 'comp5_rate_percent_diff', 'comp1_rate', 'comp8_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_inv',
#                    'comp1_rate_percent_diff_new', 'comp7_inv', 'comp8_rate_percent_diff_new', 'comp1_rate_percent_diff', 'price_usd', 'comp6_inv', 'comp2_inv', 'comp4_rate_percent_diff_new', 'comp3_rate_percent_diff_new', 'comp8_inv', 'comp5_rate_percent_diff_new', 'comp4_rate_percent_diff'],
#                   [-0.08889105639259487, 0.07543675408534882, 0.036046924370283576, 0.025935806449296793, -0.02441209259496601, 0.021205985479062755, -0.019582215818154874, 0.014343623866078292, 0.014256960783880326, 0.011123257947035212, 0.010881269408780073, 0.009990631157615042, 0.009861018266822869, 0.00952737379602103, 0.007948277954115701, 0.005477514662799231, -0.005376241666741471, 0.004594473547159571, 0.004484334064470641, -0.004416863673775751, 0.0038716695261854007,
#                    0.0038678754787810177, -0.0033330378867110558, -0.003273212832235774, 0.0029211347955481032, -0.00268256655247564, 0.002637045129170967, 0.0025236772394496656, 0.0023750711692517193, 0.0021685243369910716, -0.0018283768996005534, -0.0014811671222171728, -0.001353501364401725, 0.0012274208349323061, 0.0011156347993593226, 0.0011108612780806905, -0.0008065346372432155, 0.0008003560159799413, -0.0005081730244986088, -0.0004319050366518186, 0.0003849477263090758,
#                    -0.0003126023795346904, -0.000246046110599506, -0.00024031150314908606, -0.00020290125006133038, -0.0001913096093171219, 0.00014526030157029792, 6.92432042671204e-05, 6.676156824433244e-05, -5.601460163192244e-05, 4.545839935207119e-05, 4.338099911402048e-05, 4.1279670764112543e-05, -3.4819714843338745e-05, 3.3452501729467344e-05, -3.3461456273618546e-06], vertical=False)

# plot_single_feature_ndcg(
#     ['date_time', 'site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score',
#      'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff'],
#     [0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.35454079479680234, 0.37050516319600685, 0.381531420776188, 0.37229803662347205, 0.3606032021245783, 0.3662625903120052, 0.43531325925752007, 0.36849156441905434, 0.38386140187507806, 0.38398719324631014, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.36150169356320344, 0.35351173337718367,
#      0.3544297394242687, 0.35437692578602703, 0.3543443407663558, 0.35370807942342336, 0.3583600696467722, 0.3551903222022826, 0.36006119435969625, 0.3584839855324564, 0.35471591288599896, 0.35982074072832837, 0.3549076935260042, 0.3545919843374264, 0.3541074858837556, 0.36123525238583737, 0.3540998120710616, 0.3614400899629054, 0.35457542842438405, 0.35435469285837734, 0.3544762148963635, 0.3558148133980496, 0.354179117754522, 0.35459501116437636, 0.3598658317263708, 0.3552612070118234,
#      0.35757913514143336])
