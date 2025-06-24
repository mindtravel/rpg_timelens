import json
experiment_names = ["completely_aligned", "slightly_disaligned", "disaligned", "completely_disaligned"]
biases = [0, 1e2, 1e3, 1e4]
for index, (name, bias) in enumerate(zip(experiment_names, biases)):
    experiment_params = {
        "name": name,
        "bias":  bias
    }

    with open("params" + "_" + str(index) + ".json", "w") as f:
        json.dump(experiment_params, f)