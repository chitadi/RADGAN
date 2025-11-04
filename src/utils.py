import hashlib
import yaml

def safe_open_yaml(yaml_file, verbose=False):

    with open(yaml_file, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            if verbose:
                print("YAML content as dictionary:")
                print(yaml_data)
        except yaml.YAMLError as exc:
            print(exc)

    
    return yaml_data


def stringify(values: dict, delimiter: str = "_") -> str:
    pieces = []
    for key, value in values.items():
        if isinstance(value, dict):
            payload = yaml.safe_dump(value, sort_keys=True)
            digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
            hint = value.get("kind", "cfg")
            display = f"{hint}-{digest}"
        elif isinstance(value, (list, tuple, set)):
            payload = yaml.safe_dump({"value": list(value)}, sort_keys=True)
            digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
            display = digest
        else:
            display = str(value)
        pieces.append(f"{key}={display}")
    return delimiter.join(pieces)
