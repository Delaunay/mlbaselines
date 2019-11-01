def delayed_geffnet(name):

    def make_geffnet(input_size, output_size):

        try:
            import geffnet
        except ImportError:
            print('geffnet is not installed!')
            print('> pip install geffnet')
            raise

        # pytorch is CHW
        return geffnet.create_model(name, num_classes=output_size, in_chans=input_size[0])

    return make_geffnet


names = [
    'mobilenetv3_050', 'mobilenetv3_075', 'mobilenetv3_100',
    'GenEfficientNet', 'mnasnet_050', 'mnasnet_075', 'mnasnet_100', 'mnasnet_b1', 'mnasnet_140',
    'semnasnet_050', 'semnasnet_075', 'semnasnet_100', 'mnasnet_a1', 'semnasnet_140', 'mnasnet_small',
    'fbnetc_100', 'spnasnet_100', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',  'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_es', 'efficientnet_em', 'efficientnet_el',
    'efficientnet_cc_b0_4e', 'efficientnet_cc_b0_8e', 'efficientnet_cc_b1_8e',
    'tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2', 'tf_efficientnet_b3',
    'tf_efficientnet_b4', 'tf_efficientnet_b5', 'tf_efficientnet_b6', 'tf_efficientnet_b7',
    'tf_efficientnet_es', 'tf_efficientnet_em', 'tf_efficientnet_el',
    'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e',
    'mixnet_s', 'mixnet_m', 'mixnet_l', 'mixnet_xl', 'tf_mixnet_s', 'tf_mixnet_m', 'tf_mixnet_l'
]

builders = {
    name: delayed_geffnet(name) for name in names
}

