import torch
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)