import json

for f in [
    "data/harmbench_test_std.json",
    "data/harmbench_test_ctx.json",
    "data/harmbench_test.json",
    "data/circuit_breakers_train.json",
    "data/circuit_breakers_val.json",
]:
    try:
        data = json.load(open(f))
        print(f, len(data["data"]))
    except Exception as e:
        print(f, e)