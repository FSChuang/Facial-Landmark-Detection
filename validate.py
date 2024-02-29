import torch
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn as nn
import util.Visualize

@torch.no_grad
def validate(model, val_data, save = None):
    objective = nn.MSELoss()
    cum_loss = 0.0

    model.eval()

    for features, labels in tqdm(val_data, desc = 'Validating', ncols = 600, bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}'):
        features = features.cuda()
        labels = labels.cuda()

        outputs = model(features)

        loss = objective(outputs, labels)

        cum_loss += loss.item()

        break

    util.Visualize.visualize_batch(features[:16].cpu(), outputs[:16].cpu(), shape = (4,4), size = 16, title = 'Validation')


    return cum_loss/len(val_data)