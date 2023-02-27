from PyNetwork import optimizers


def get_optimizer(optimizer_str):
    available_optimizers = ['Adam', 'Nadam', 'SGD', 'RMSprop', 'Adadelta', 'Adamax']
    for existing_optimizer in available_optimizers:
        if existing_optimizer.lower() == optimizer_str.lower():
            return optimizers.__dict__[existing_optimizer]()

    raise ValueError(f"{optimizer_str} is not defined.")
