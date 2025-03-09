import logging


def cfg_with_default(cfg, key_list, default):
    root = cfg
    for k in key_list:
        if k in root.keys():
            root = root[k]
        else:
            return default
    return root


def count_param(network_dict):
    for k in network_dict:
        logging.info(
            "{:.3f}M params in {}".format(
                sum(param.numel() for param in network_dict[k].parameters()) / 1e6, k
            )
        )
