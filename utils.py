import json


def save_model_architecture_to_file(model, f_path):
    model_config = model.to_json()
    with open(f_path, "w") as f_out:
        json.dump(model_config, f_out)