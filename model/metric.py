import torch

def mse(output, target, mask=None, variable=False):
    """
    Calculate the MSE between output and target. A mask of true missing values
    can be passed, and the calculated results can be requested at variable 
    level, rather than aggregate.
    """
    with torch.no_grad():
        if mask is None:
            if variable is True:
                output_ = output * (1-mask)
                target_ = target * (1-mask)
                denom = torch.sum((1-mask), dim=0)
                numer = torch.sum((output_-target_)**2, dim=0)
                return torch.div(numer, denom)
            else:
                return torch.mean((output - target)**2)
        else:
            output_ = output[mask==0]
            target_ = target[mask==0]
            return torch.mean((output_-target_)**2)

def mae(output, target, mask=None, variable=False):
    """
    Calculate the MAE between output and target. A mask of true missing values
    can be passed, and the calculated results can be requested at variable 
    level, rather than aggregate.
    """
    with torch.no_grad():
        if mask is None:
            if variable is True:
                output_ = output * (1-mask)
                target_ = target * (1-mask)
                denom = torch.sum((1-mask), dim=0)
                numer = torch.sum(torch.abs(output_-target_), dim=0)
                return torch.div(numer, denom)
            else:
                return torch.mean(torch.abs(output - target))
        else:
            output_ = output[mask==0]
            target_ = target[mask==0]
            return torch.mean(torch.abs(output_-target_))

def rmse(output, target, mask=None, variable=False):
    """
    Calculate the Root MSE between output and target. A mask of true missing
    values can be passed, and the calculated results can be requested at variable 
    level, rather than aggregate.
    """
    with torch.no_grad():
        if mask is None:
            if variable is True:
                output_ = output * (1-mask)
                target_ = target * (1-mask)
                denom = torch.sum((1-mask), dim=0)
                numer = torch.sum(torch.sqrt((output_-target_)**2), dim=0)
                return torch.div(numer, denom)
            else:
                return torch.mean(torch.sqrt((output - target)**2))
        else:
            output_ = output[mask==0]
            target_ = target[mask==0]
            return torch.mean(torch.sqrt((output_-target_)**2))

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
