import os
class DefaultConfigs(object):
    # 1.string parameters
    path_images = "./data/images/"
    path_weights = "./checkpoints_region/"
    path_best_models = os.path.join(path_weights, "best_model/")
    path_logs = "./logs/"
    augmen_level = "medium"  # "light","hard","hard2"
    skeletons_pth = './skeleton/'
    model_name = 'resnet50'

    # 2.numeric parameters
    epochs = 100
    batch_size = 16
    img_height = 256
    img_weight = 256
    num_classes = 639
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4
    save_frequency = 100
