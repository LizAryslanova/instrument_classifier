
def trace_the_model():
    """


    """

    import torch
    import torchvision
    import os
    import pickle
    import numpy as np
    from cnn import CNN

    current_dir = os.path.dirname(os.path.realpath(__file__))
    device = torch.device('cpu') # ('mps' if torch.backends.mps.is_available() else 'cpu')

    # model built for .5 seconds
    model = torch.load('/Users/cookie/dev/instrument_classifier/model_results/m_loudener/lr_0.0003_epochs_150_20240302-051316/lr_0.0003_epochs_150_20240302-051316.pt')
    model = model.cpu().float()

    model.eval()


    # Un-Pickle Test sets
    with open(current_dir + '/pickles/test_x_m_loudener_14_classes_20240302-042433', 'rb') as f:
        X_test = pickle.load(f)

    _, a, b, c = X_test.shape

    #X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
    #print(X_test.shape)
    #example = X_test
    #print(example.shape)

    example = torch.rand(1, a, b, c).to(device)

    print(a, b, c)


    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("m_loudener_20240302-042433.pt")

trace_the_model()