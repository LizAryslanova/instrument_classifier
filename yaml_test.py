import yaml

with open('model.yml', 'r') as file:
    yaml_test = yaml.safe_load(file)


model = yaml_test['model_parameters']


print(model)
print(model['padding_conv_1'])
# print(yaml_test['train_loop']['num_classes'])