import matplotlib.pyplot as plt
import seaborn as sns


def plot_everything(old_features, old_correlations, old_ndcgs, new_features, new_correlations, new_ndcgs):
    old_absolute_correlations = [abs(correlation) for correlation in old_correlations]
    old_colors = ['red' if correlation < 0 else "green" for correlation in old_correlations]

    old_min_correlation = min(old_absolute_correlations)
    old_max_correlation = max(old_absolute_correlations) - old_min_correlation
    old_min_ndcg = min(old_ndcgs)
    old_max_ndcg = max(old_ndcgs) - old_min_ndcg

    old_features, old_absolute_correlations, old_ndcgs, old_colors = (
        list(t) for t in zip(*sorted(zip(old_features, old_absolute_correlations, old_ndcgs, old_colors),
                                     key=lambda tup: (tup[1] - old_min_correlation) / float(old_max_correlation) + (tup[2] - old_min_ndcg) / float(old_max_ndcg), reverse=True))
    )

    new_absolute_correlations = [abs(correlation) for correlation in new_correlations]
    new_colors = ['red' if correlation < 0 else "green" for correlation in new_correlations]

    new_min_correlation = min(new_absolute_correlations)
    new_max_correlation = max(new_absolute_correlations) - new_min_correlation
    new_min_ndcg = min(new_ndcgs)
    new_max_ndcg = max(new_ndcgs) - new_min_ndcg

    new_features, new_absolute_correlations, new_ndcgs, new_colors = (
        list(t) for t in zip(*sorted(zip(new_features, new_absolute_correlations, new_ndcgs, new_colors),
                                     key=lambda tup: (tup[1] - new_min_correlation) / float(new_max_correlation) + (tup[2] - new_min_ndcg) / float(new_max_ndcg), reverse=True))
    )

    figure, axes = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')

    axes[0][0].bar(range(len(old_absolute_correlations)), old_absolute_correlations, color=old_colors)
    axes[1][0].bar(range(len(old_ndcgs)), old_ndcgs)
    axes[1][0].set_ylim([min(old_ndcgs), max(old_ndcgs)])
    axes[0][0].tick_params(axis='y', labelsize=8)
    axes[1][0].tick_params(axis='y', labelsize=8)

    axes[0][1].bar(range(len(new_absolute_correlations)), new_absolute_correlations, color=new_colors)
    axes[1][1].bar(range(len(new_ndcgs)), new_ndcgs)
    axes[1][1].set_ylim([min(sorted(new_ndcgs)[4:]), max(new_ndcgs)])
    axes[0][1].tick_params(axis='y', labelsize=8)
    axes[1][1].tick_params(axis='y', labelsize=8)

    # plt.xticks(range(len(features)), features)
    # plt.xticks(rotation=90, fontsize=6)
    plt.margins(0)
    plt.tight_layout()

    plt.show()


def plot_both(features, correlations, ndcgs, remove_outliers=False):
    absolute_correlations = [abs(correlation) for correlation in correlations]
    colors = ['red' if correlation < 0 else "green" for correlation in correlations]

    min_correlation = min(absolute_correlations)
    max_correlation = max(absolute_correlations) - min_correlation
    min_ndcg = min(ndcgs)
    max_ndcg = max(ndcgs) - min_ndcg

    features, absolute_correlations, ndcgs, colors = (
        list(t) for t in zip(*sorted(zip(features, absolute_correlations, ndcgs, colors),
                                     key=lambda tup: (tup[1] - min_correlation) / float(max_correlation) + (tup[2] - min_ndcg) / float(max_ndcg), reverse=True))
    )

    widthscale = len(features) / 4
    figsize = (0.65 * widthscale, 6)  # fig size in inches (width,height)

    figure, axes = plt.subplots(nrows=2, sharex=True, figsize=figsize)

    axes[0].bar(range(len(absolute_correlations)), absolute_correlations, color=colors)
    axes[1].bar(range(len(ndcgs)), ndcgs)
    # if remove_outliers:
    #     axes[1].set_ylim([min(sorted(ndcgs)[4:]), max(ndcgs)])
    # else:
    #     axes[1].set_ylim([min(sorted(ndcgs)), max(ndcgs)])

    axes[0].set_ylim([0, 0.09])
    axes[0].set_yticks([0, 0.05, 0.09])

    axes[1].set_ylim([0.35, 0.44])
    axes[1].set_yticks([0.35, 0.4, 0.44])

    axes[0].tick_params(axis='y', labelsize=8)
    axes[1].tick_params(axis='y', labelsize=8)
    plt.xticks(range(len(features)), features)
    plt.xticks(rotation=90, fontsize=6)
    # plt.yticks(fontsize=8)
    plt.margins(0)
    plt.tight_layout()

    plt.show()


def plot_correlations(features, correlations, vertical=True):
    correlations, features = (list(t) for t in zip(*sorted(zip(correlations, features), key=lambda tup: abs(tup[0]), reverse=vertical)))
    colors = ['red' if correlation < 0 else "blue" for correlation in correlations]
    absolute_correlations = [abs(correlation) for correlation in correlations]

    fig, ax = plt.subplots()

    if vertical:
        plt.ylim(min(absolute_correlations), max(absolute_correlations))
        ax.bar(range(len(correlations)), absolute_correlations, color=colors)
        plt.xticks(range(len(absolute_correlations)), features)
        plt.xticks(rotation=90, ha="right", fontsize=6)
        plt.yticks(fontsize=8)
        plt.margins(0)
    else:
        plt.xlim(min(absolute_correlations), max(absolute_correlations))
        ax.barh(range(len(correlations)), absolute_correlations, color=colors)
        plt.yticks(range(len(absolute_correlations)), features)

    plt.tight_layout()

    plt.show()


def plot_single_feature_ndcg(features, ndcgs):
    ndcgs, features = (list(t)[:-4] for t in zip(*sorted(zip(ndcgs, features), reverse=True)))

    fig, ax = plt.subplots()
    plt.ylim(min(ndcgs), max(ndcgs))
    ax.bar(range(len(ndcgs)), ndcgs)
    plt.xticks(range(len(ndcgs)), features)
    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.show()


def plot_both_for_raw_data():
    plot_both(
        ['date_time', 'site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score',
         'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff'],
        [-0.00016115860009261983, -0.001353501364401725, 0.0025236772394496656, 0.011123257947035212, 0.009861018266822869, 0.0012274208349323061, -0.0005081730244986088, 0.021205985479062755, 0.025935806449296793, 0.009990631157615042, -0.003273212832235774, 0.07543675408534882, -0.0008065346372432155, 6.676156824433244e-05, 0.036046924370283576, 0.0008003560159799413, -0.02441209259496601, -0.019582215818154874, -0.005376241666741471, 0.0038716695261854007, 0.007948277954115701,
         0.005477514662799231, -0.004416863673775751, -0.00268256655247564, -0.08889105639259487, 0.0003849477263090758, -0.0014811671222171728, 6.92432042671204e-05, 0.010881269408780073, 4.545839935207119e-05, 0.0011108612780806905, 0.00952737379602103, -0.0033330378867110558, -0.000246046110599506, 0.004484334064470641, -0.00024031150314908606, -3.3461456273618546e-06, 0.014256960783880326, -0.0018283768996005534, -0.0004319050366518186, 0.0038678754787810177,
         -5.601460163192244e-05, 0.0021685243369910716, 0.004594473547159571, -0.0001913096093171219, 0.002637045129170967, 0.014343623866078292, -3.4819714843338745e-05, -0.0003126023795346904],
        [0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.35454079479680234, 0.37050516319600685, 0.381531420776188, 0.37229803662347205, 0.3606032021245783, 0.3662625903120052, 0.43531325925752007, 0.36849156441905434, 0.38386140187507806, 0.38398719324631014, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.36150169356320344, 0.35351173337718367,
         0.3544297394242687, 0.35437692578602703, 0.3543443407663558, 0.35370807942342336, 0.3583600696467722, 0.3551903222022826, 0.36006119435969625, 0.3584839855324564, 0.35471591288599896, 0.35982074072832837, 0.3549076935260042, 0.3545919843374264, 0.3541074858837556, 0.36123525238583737, 0.3540998120710616, 0.3614400899629054, 0.35457542842438405, 0.35435469285837734, 0.3544762148963635, 0.3558148133980496, 0.354179117754522, 0.35459501116437636, 0.3598658317263708, 0.3552612070118234,
         0.35757913514143336])


def get_sorted_new_features():
    correlation_features = ['comp1_rate_percent_diff|signed', 'comp2_rate_percent_diff|signed', 'comp3_rate_percent_diff|signed', 'comp4_rate_percent_diff|signed', 'comp5_rate_percent_diff|signed', 'comp6_rate_percent_diff|signed', 'comp7_rate_percent_diff|signed', 'comp8_rate_percent_diff|signed', 'comp_rate|sum', 'comp_rate|sum|positive', 'comp_rate|sum|negative', 'prop_starrating|personal_difference', 'price_usd|personal_difference', 'day_part', 'season',
                            'prop_starrating|srch_id_standardized', 'prop_review_score|srch_id_standardized', 'prop_location_score1|srch_id_standardized', 'prop_location_score2|srch_id_standardized', 'prop_log_historical_price|srch_id_standardized', 'price_usd|srch_id_standardized', 'srch_query_affinity_score|srch_id_standardized', 'orig_destination_distance|srch_id_standardized', 'comp_rate|sum|srch_id_standardized', 'comp_rate|sum|positive|srch_id_standardized',
                            'comp_rate|sum|negative|srch_id_standardized', 'prop_starrating|personal_difference|srch_id_standardized', 'price_usd|personal_difference|srch_id_standardized']

    correlation_values = [-0.00020290125006133038, 0.0011156347993593226, 4.1279670764112543e-05, 4.338099911402048e-05, 3.3452501729467344e-05, 0.0023750711692517193, 0.0029211347955481032, 0.00014526030157029792, 0.01873301951823676, 0.019233724236803598, 0.007495785181535264, 0.0006141973192462382, 2.6051366225824547e-05, -0.0016218023673024906, 0.0023883440910522987, 0.03786184235514443, 0.024783810699319706, 0.01878606566054998, 0.07263556419772177, 0.01196969531669563,
                          -0.04182124852909668, 0.03891157685641361, -0.0007875472206431265, 0.018362442957342406, 0.01687191152937256, 0.010514819719603354, -0.012602363708693158, -0.01619101761847791]

    ndcg_features = ['site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score',
                     'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff',
                     'comp1_rate_percent_diff|signed', 'comp2_rate_percent_diff|signed', 'comp3_rate_percent_diff|signed', 'comp4_rate_percent_diff|signed', 'comp5_rate_percent_diff|signed', 'comp6_rate_percent_diff|signed', 'comp7_rate_percent_diff|signed', 'comp8_rate_percent_diff|signed', 'comp_rate|sum', 'comp_rate|sum|positive', 'comp_rate|sum|negative', 'prop_starrating|personal_difference', 'price_usd|personal_difference', 'day_part', 'season', 'prop_starrating|srch_id_standardized',
                     'prop_review_score|srch_id_standardized', 'prop_location_score1|srch_id_standardized', 'prop_location_score2|srch_id_standardized', 'prop_log_historical_price|srch_id_standardized', 'price_usd|srch_id_standardized', 'srch_query_affinity_score|srch_id_standardized', 'orig_destination_distance|srch_id_standardized', 'comp_rate|sum|srch_id_standardized', 'comp_rate|sum|positive|srch_id_standardized', 'comp_rate|sum|negative|srch_id_standardized',
                     'prop_starrating|personal_difference|srch_id_standardized', 'price_usd|personal_difference|srch_id_standardized']

    ndcg_values = [0.354710344104, 0.354710344104, 0.354710344104, 0.354710344104, 0.354597007311, 0.368456655558, 0.381924505608, 0.372016359836, 0.360343399647, 0.365911034575, 0.436588428023, 0.37004754891, 0.385961593442, 0.385966954181, 0.354710344104, 0.354710344104, 0.354710344104, 0.354710344104, 0.354710344104, 0.354710344104, 0.354710344104, 0.359934404613, 0.354512909979, 0.354710344104, 0.355109913713, 0.354828383347, 0.354131934038, 0.35889451081, 0.354604550057, 0.361086773276,
                   0.359189476094, 0.355418961873, 0.359102158257, 0.35574160825, 0.35479654079, 0.355033948234, 0.361329581551, 0.355625924171, 0.358991780195, 0.355656137611, 0.354583041579, 0.355082858538, 0.355854656746, 0.353265198853, 0.355451205519, 0.360229999196, 0.35535461861, 0.358188189537, 0.35494508977, 0.359812403025, 0.359562829859, 0.355734441987, 0.36088215439, 0.354727269353, 0.354657268622, 0.360759432688, 0.367104864022, 0.366128979247, 0.357070335518, 0.354283561395,
                   0.35562102126, 0.354710344104, 0.354710344104, 0.378735192699, 0.366118341397, 0.0, 0.0, 0.0, 0.384872637331, 0.363540843196, 0.362671491731, 0.366378571943, 0.365172356173, 0.356537619961, 0.0, 0.355964518099]

    ndcg_features, ndcg_values = (list(t) for t in zip(*[tup for tup in zip(ndcg_features, ndcg_values) if tup[0] in correlation_features]))

    return ndcg_features, correlation_values, ndcg_values


def plot_both_for_new_features():
    ndcg_features, correlation_values, ndcg_values = get_sorted_new_features()
    plot_both(ndcg_features, correlation_values, ndcg_values)


def do_the_plot_everything():
    old_features = ['date_time', 'site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
                    'srch_query_affinity_score', 'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
                    'comp8_rate_percent_diff']

    old_correlations = [-0.00016115860009261983, -0.001353501364401725, 0.0025236772394496656, 0.011123257947035212, 0.009861018266822869, 0.0012274208349323061, -0.0005081730244986088, 0.021205985479062755, 0.025935806449296793, 0.009990631157615042, -0.003273212832235774, 0.07543675408534882, -0.0008065346372432155, 6.676156824433244e-05, 0.036046924370283576, 0.0008003560159799413, -0.02441209259496601, -0.019582215818154874, -0.005376241666741471, 0.0038716695261854007, 0.007948277954115701,
                        0.005477514662799231, -0.004416863673775751, -0.00268256655247564, -0.08889105639259487, 0.0003849477263090758, -0.0014811671222171728, 6.92432042671204e-05, 0.010881269408780073, 4.545839935207119e-05, 0.0011108612780806905, 0.00952737379602103, -0.0033330378867110558, -0.000246046110599506, 0.004484334064470641, -0.00024031150314908606, -3.3461456273618546e-06, 0.014256960783880326, -0.0018283768996005534, -0.0004319050366518186, 0.0038678754787810177,
                        -5.601460163192244e-05, 0.0021685243369910716, 0.004594473547159571, -0.0001913096093171219, 0.002637045129170967, 0.014343623866078292, -3.4819714843338745e-05, -0.0003126023795346904]

    old_ndcgs = [0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.35454079479680234, 0.37050516319600685, 0.381531420776188, 0.37229803662347205, 0.3606032021245783, 0.3662625903120052, 0.43531325925752007, 0.36849156441905434, 0.38386140187507806, 0.38398719324631014, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.3544297394242687, 0.36150169356320344,
                 0.35351173337718367, 0.3544297394242687, 0.35437692578602703, 0.3543443407663558, 0.35370807942342336, 0.3583600696467722, 0.3551903222022826, 0.36006119435969625, 0.3584839855324564, 0.35471591288599896, 0.35982074072832837, 0.3549076935260042, 0.3545919843374264, 0.3541074858837556, 0.36123525238583737, 0.3540998120710616, 0.3614400899629054, 0.35457542842438405, 0.35435469285837734, 0.3544762148963635, 0.3558148133980496, 0.354179117754522, 0.35459501116437636,
                 0.3598658317263708, 0.3552612070118234, 0.35757913514143336]
    new_ndcg_features, new_correlation_values, new_ndcg_values = get_sorted_new_features()

    plot_everything(old_features, old_correlations, old_ndcgs, new_ndcg_features, new_correlation_values, new_ndcg_values)


plot_both_for_new_features()
# plot_both_for_raw_data()

# plot_single_feature_ndcg(
#     ['site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score',
#      'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff',
#      'comp1_rate_percent_diff|signed', 'comp2_rate_percent_diff|signed', 'comp3_rate_percent_diff|signed', 'comp4_rate_percent_diff|signed', 'comp5_rate_percent_diff|signed', 'comp6_rate_percent_diff|signed', 'comp7_rate_percent_diff|signed', 'comp8_rate_percent_diff|signed', 'comp_rate|sum', 'comp_rate|sum|positive', 'comp_rate|sum|negative', 'prop_starrating|personal_difference', 'price_usd|personal_difference', 'day_part', 'season', 'prop_starrating|srch_id_standardized',
#      'prop_review_score|srch_id_standardized', 'prop_location_score1|srch_id_standardized', 'prop_location_score2|srch_id_standardized', 'prop_log_historical_price|srch_id_standardized', 'price_usd|srch_id_standardized', 'srch_query_affinity_score|srch_id_standardized', 'orig_destination_distance|srch_id_standardized', 'comp_rate|sum|srch_id_standardized', 'comp_rate|sum|positive|srch_id_standardized', 'comp_rate|sum|negative|srch_id_standardized',
#      'prop_starrating|personal_difference|srch_id_standardized', 'price_usd|personal_difference|srch_id_standardized'],
#     [0.354710344104, 0.354710344104, 0.354710344104, 0.354710344104, 0.354597007311, 0.368456655558, 0.381924505608, 0.372016359836, 0.360343399647, 0.365911034575, 0.436588428023, 0.37004754891, 0.385961593442, 0.385966954181, 0.354710344104, 0.354710344104, 0.354710344104, 0.354710344104, 0.354710344104, 0.354710344104, 0.354710344104, 0.359934404613, 0.354512909979, 0.354710344104, 0.355109913713, 0.354828383347, 0.354131934038, 0.35889451081, 0.354604550057, 0.361086773276, 0.359189476094,
#      0.355418961873, 0.359102158257, 0.35574160825, 0.35479654079, 0.355033948234, 0.361329581551, 0.355625924171, 0.358991780195, 0.355656137611, 0.354583041579, 0.355082858538, 0.355854656746, 0.353265198853, 0.355451205519, 0.360229999196, 0.35535461861, 0.358188189537, 0.35494508977, 0.359812403025, 0.359562829859, 0.355734441987, 0.36088215439, 0.354727269353, 0.354657268622, 0.360759432688, 0.367104864022, 0.366128979247, 0.357070335518, 0.354283561395, 0.35562102126, 0.354710344104,
#      0.354710344104, 0.378735192699, 0.366118341397, 0.0, 0.0, 0.0, 0.384872637331, 0.363540843196, 0.362671491731, 0.366378571943, 0.365172356173, 0.356537619961, 0.0, 0.355964518099])
