
def trace_the_model(model_folder):
    """ Traces the model from model_folder and saves traced version in the same folder  """
    import torch
    import torchvision
    import os
    import pickle
    import numpy as np
    from cnn import CNN

    file = model_folder.split('/')[-1]
    model_address = model_folder + '/' + file + '.pt'
    model_yaml = model_folder + '/' + file + '.yml'
    device = torch.device('cpu')
    model = torch.load(model_address)
    model = model.cpu().float()
    model.eval()

    import yaml
    with open(model_yaml, 'r') as yaml_file:
        yaml_input = yaml.safe_load(yaml_file)

    mel_size = yaml_input['preprocessing']['mel_size']
    time_size = yaml_input['preprocessing']['time_size']
    example = torch.rand(1, 1, mel_size, time_size).to(device)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(model_folder + '/' + 'm_loudener_' + file + '.pt')



    # Save to onnx
    torch_model = model
    torch_input = example

    import onnx
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)

    onnx_program.save(model_folder + '/' + 'm_loudener_' + file + '.onnx')



    onnx_model = onnx.load(model_folder + '/' + 'm_loudener_' + file + '.onnx')
    onnx.checker.check_model(onnx_model)





trace_the_model('/Users/cookie/Desktop/Release_Models/lr_2e-06_epochs_280_20240327-181755')