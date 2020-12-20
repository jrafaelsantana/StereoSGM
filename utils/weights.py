def getWeights (net, layerName):
    state = net.state_dict()
    items = state.items()
    weights = []
    bias = []

    for item in items:
        if item[0].startswith(layerName):
            if item[0].endswith('weight'):
                weights.append(item[1].cpu().numpy())
            elif item[0].endswith('bias'):
                bias.append(item[1].cpu().numpy())

    return weights, bias