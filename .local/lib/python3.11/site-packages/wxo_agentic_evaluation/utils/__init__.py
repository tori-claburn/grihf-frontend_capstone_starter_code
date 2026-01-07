import json


def json_dump(output_path, object):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(object, f, indent=4)
