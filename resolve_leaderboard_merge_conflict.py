import json
import os
from collections import OrderedDict


def resolve_leaderboard_merge_conflict(location="output/spl_20180518114037", file_name="leaderboard.json"):
    with open(location + '/' + file_name, "a+") as conflicted_file:
        conflicted_leaderboards = []
        leaderboard = {}

        conflicted_file.seek(0)

        while True:
            line = conflicted_file.readline().rstrip()
            if not line:
                break
            else:
                try:
                    one_of_the_leaderboards = json.loads(line)
                    conflicted_leaderboards.append(one_of_the_leaderboards)
                except ValueError:
                    pass

        for i, conflicted_leaderboard in enumerate(conflicted_leaderboards):
            for experiment_size in conflicted_leaderboard:
                if experiment_size not in leaderboard:
                    leaderboard[experiment_size] = []

                for result in conflicted_leaderboard[experiment_size]:
                    leaderboard[experiment_size].append(tuple(result))

                if i == len(conflicted_leaderboards) - 1:
                    leaderboard[experiment_size] = sorted(list(OrderedDict.fromkeys(leaderboard[experiment_size])), key=lambda tup: tup[0], reverse=True)

        conflicted_file.truncate(0)
        json.dump(leaderboard, conflicted_file)

        with open(location + '/' + os.path.splitext(file_name)[0] + '.txt', "w") as txt_file:
            for experiment_size in leaderboard:
                txt_file.write(experiment_size + '\n')
                largest_experiment_name = max([len(row[1]) for row in leaderboard[experiment_size]])
                for i, row in enumerate(leaderboard[experiment_size]):
                    txt_file.write("   {ind}. {ndcg:1.6f} {name:<{len}} {run}\n".format(
                        ind='[' + str(i + 1) + ']', ndcg=row[0], name=row[1], run=row[2], len=largest_experiment_name)
                    )
                txt_file.write('\n')


resolve_leaderboard_merge_conflict()
