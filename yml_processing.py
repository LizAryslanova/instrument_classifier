import yaml

YML_ADDRESS = '/Users/cookie/dev/instrument_classifier/model.yml'


def get_classes(address = YML_ADDRESS):
    """ Returs tuple of classes that are in the .yml file with model parameters """
    import yaml
    with open(address, 'r') as file:
        yaml_input = yaml.safe_load(file)

    classes = ()
    for i in range(yaml_input['model_parameters']['num_classes']):
        classes = classes + (yaml_input['train_loop']['classes']['c_'+str(i+1)], )
    return classes



def dict_of_classes(yaml_address = YML_ADDRESS):
    """ Takes in the yaml file with dictionary of classes and turnes ot into a dictionary where keys are classes and all value are 0 """
    import yaml
    with open(yaml_address, 'r') as file:
        yaml_input = yaml.safe_load(file)
    dictionary_of_classes = {}
    for i in range(yaml_input['model_parameters']['num_classes']):

        dictionary_of_classes [ yaml_input['train_loop']['classes']['c_'+str(i+1)] ] = 0
    return dictionary_of_classes




def dict_of_labels(yaml_address = YML_ADDRESS):
    """ Takes in the yaml file with dictionary of classes and turnes ot into a dictionary where keys are classes and all value are from 0 to num_classses - 1 """
    import yaml
    with open(yaml_address, 'r') as file:
        yaml_input = yaml.safe_load(file)
    dictionary_of_labels = {}
    for i in range(yaml_input['model_parameters']['num_classes']):

        dictionary_of_labels [ yaml_input['train_loop']['classes']['c_'+str(i+1)] ] = i
    return dictionary_of_labels




def create_modelTrates_yml(from_yaml_address = YML_ADDRESS, to_yaml_address = '/Users/cookie/dev/m_loudener/modelTraits.yml'):
    """ Takes info from yml and saves into a nes yml only what is needed for syncing with c++ """
    import yaml
    with open(from_yaml_address, 'r') as file:
        yaml_input = yaml.safe_load(file)

    yaml_output = {}

    tuple_classes = get_classes()
    labels = list(tuple_classes)

    yaml_output['labels'] = labels
    yaml_output['sample_rate'] = yaml_input['preprocessing']['sample_rate']
    yaml_output['fmax'] = yaml_input['preprocessing']['fmax']
    yaml_output['duration_sec'] = yaml_input['preprocessing']['duration_sec']
    yaml_output['fft_size'] = yaml_input['preprocessing']['fft_size']
    yaml_output['fft_hop'] = yaml_input['preprocessing']['fft_hop']
    yaml_output['mel_size'] = yaml_input['preprocessing']['mel_size']
    yaml_output['mel_power'] = yaml_input['preprocessing']['mel_power']
    yaml_output['time_size'] = yaml_input['preprocessing']['time_size']
    yaml_output['filter_dc'] = yaml_input['preprocessing']['filter_dc']

    with open(to_yaml_address, 'w') as file:
        documents = yaml.dump(yaml_output, file)

    import json
    with open(to_yaml_address, 'r') as file:
        configuration = yaml.safe_load(file)
    to_json_address = to_yaml_address[:-4] + '.json'

    with open(to_json_address, 'w') as json_file:
        json.dump(configuration, json_file)
    output = json.dumps(json.load(open(to_json_address)), indent=2)
    #print(output)

