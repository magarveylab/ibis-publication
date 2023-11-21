from Ibis import curdir

def get_ec_dict(level: int = 1) -> Dict[int, str]:
    return {y: x for x, y in json.load(open(f'{curdir}/dat/class_dicts/ec_level_{level}.json')).items()}