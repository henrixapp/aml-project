import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff

def myCustomLoss(my_outputs, my_labels):
    #specifying the batch size
    my_batch_size = my_outputs.size()[0] 
    #calculating the log of softmax values           
    my_outputs = F.log_softmax(my_outputs, dim=1)  
    #selecting the values that correspond to labels
    my_outputs = my_outputs[range(my_batch_size), my_labels] 
    #returning the results
    return -torch.sum(my_outputs)/number_examples
def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float64 #if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch
            imgs = imgs.to(device=device, dtype=torch.float64)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                pred = F.relu(mask_pred)
                tot += F.mse_loss(pred, true_masks).item()
            else:
                pred = F.relu(mask_pred)
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val