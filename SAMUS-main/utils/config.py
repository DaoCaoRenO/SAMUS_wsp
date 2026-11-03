# This file is used to configure the training parameters for each task
class Config_US30K:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../../dataset/SAMUS/"  
    save_path = "./checkpoints/SAMUS/"
    result_path = "./result/SAMUS/"
    tensorboard_path = "./tensorboard/SAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# -------------------------------------------------------------------------------------------------
class Config_TN3K:
    data_path = "./dataset/SAMUS/"
    data_subpath = "./dataset/SAMUS/TN3K/"
    save_path = "./checkpoints/TN3K/"
    result_path = "./result/TN3K/"
    tensorboard_path = "./tensorboard/TN3K/"
    load_path = save_path + "/SAMUS_11020447_270_0.8349628402308167.pth" # 测试评估用 best
    load_classifier_path = save_path + "/Resnet18_01032018_59_0.7805483249970666.pth"
    save_path_code = "_"

    workers = 0                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8              # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAMUS"

    # 分类参数
    segmentation_data_path = ""

    classifier_epochs= 200
    classifier_batch_size = 128
    classifier_learning_rate = 1e-4  # initial learning rate (default: 0.001)
    classifier_momentum = 0.9  # momentum
    classifier_classes = 2  # the number of classes (normal + abnormal)
    classifier_name = "Resnet18"  # (Vit,Resnet18)
    classifier_size = 256 # (Vit:224,Resnet18 256)

class Config_KTD:
    data_path = "./dataset/SAMUS/KTD/"
    data_subpath = "./dataset/SAMUS/KTD/"
    save_path = "./checkpoints/KTD/"
    result_path = "./result/KTD/"
    tensorboard_path = "./tensorboard/KTD/"
    load_path = save_path + "/SAMUS_01111456_352_0.8360216348510668.pth" # SAMUS_12291902_47_0.8595338325001363.pth   SAMUS_01081717_8_0.42477967570244357.pth
    load_classifier_path = save_path + "/Resnet18_01032018_59_0.7805483249970666.pth"
    save_path_code = "_"

    workers = 0                  # number of data loading workers (default: 8)
    epochs = 100                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAMUS"

    # 分类参数
    segmentation_data_path = ""

    classifier_epochs= 200
    classifier_batch_size = 128
    classifier_learning_rate = 1e-4  # initial learning rate (default: 0.001)
    classifier_momentum = 0.9  # momentum
    classifier_classes = 2  # the number of classes (normal + abnormal)
    classifier_name = "Resnet18"  # (Vit,Resnet18)
    classifier_size = 256 # (Vit:224,Resnet18 256)
class Config_BUSI:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"
    data_subpath = "../../dataset/SAMUS/Breast-BUSI/"   
    save_path = "./checkpoints/BUSI/"
    result_path = "./result/BUSI/"
    tensorboard_path = "./tensorboard/BUSI/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-BUSI"   # the file name of training set
    val_split = "val-Breast-BUSI"       # the file name of testing set
    test_split = "test-Breast-BUSI"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CAMUS:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"  # 
    data_subpath = "../../dataset/SAMUS/Echocardiography-CAMUS/" 
    save_path = "./checkpoints/CAMUS/"
    result_path = "./result/CAMUS/"
    tensorboard_path = "./tensorboard/CAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 4                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-EchocardiographyLA-CAMUS"   # the file name of training set
    val_split = "val-EchocardiographyLA-CAMUS"       # the file name of testing set
    test_split = "test-Echocardiography-CAMUS"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camusmulti"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_OurDataset:
    # This dataset contain our the collected Kidney ultrasound dataset
    data_path = "../../dataset/SAMUS/"
    save_path = "./checkpoints/SAMUS/"
    result_path = "./result/SAMUS/"
    tensorboard_path = "./tensorboard/SAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1  # number of data loading workers (default: 8)
    epochs = 200  # number of total epochs to run (default: 400)
    batch_size = 8  # batch size (default: 4)
    learning_rate = 5e-4  # iniial learning rate (default: 0.001)
    momentum = 0.9  # momntum
    classes = 2  # thenumber of classes (background + foreground)
    img_size = 256  # theinput size of model
    train_split = "train"  # the file name of training set
    val_split = "val"  # the file name of testing set
    test_split = "test"  # the file name of testing set
    crop = None  # the cropped image size
    eval_freq = 1  # the frequency of evaluate the model
    save_freq = 2000  # the frequency of saving the model
    device = "cuda"  # training device, cpu or cuda
    cuda = "on"  # switch on/off cuda option (default: off)
    gray = "yes"  # the type of input image
    img_channel = 1  # the channel of input image
    eval_mode = "mask_slice"  # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
# ==================================================================================================
class Config_DDTIz(Config_TN3K):
    # 反向零样本验证：TN3K 训练的 SAMUS 测 blsamus 挑战赛 DDTI test(47张)
    test_split = "test_DDTIz"
    data_subpath = "./dataset/SAMUS/DDTIz/"
    load_path = "./checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
    result_path = "./result/DDTIz/"
    tensorboard_path = "./tensorboard/DDTIz/"

class Config_PTz(Config_TN3K):
    # 反向零样本验证：TN3K 训练的 SAMUS 测 blsamus private_Thyroid test(44张)
    test_split = "test_PTz"
    data_subpath = "./dataset/SAMUS/PTz/"
    load_path = "./checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
    result_path = "./result/PTz/"
    tensorboard_path = "./tensorboard/PTz/"

class Config_DDTIz_fda(Config_DDTIz):
    # 测试时频谱对齐(FDA)后的 DDTI 子集（低频幅度换到 TN3K 源域统计，由 fda_adapt.py 生成）
    test_split = "test_DDTIz_fda"
    data_subpath = "./dataset/SAMUS/DDTIz_fda/"
    result_path = "./result/DDTIz_fda/"

class Config_PTz_fda(Config_PTz):
    # 测试时频谱对齐(FDA)后的 private_Thyroid 子集
    test_split = "test_PTz_fda"
    data_subpath = "./dataset/SAMUS/PTz_fda/"
    result_path = "./result/PTz_fda/"

class Config_TN3K_hf(Config_TN3K):
    # 方案E：高频幅度增益后的 TN3K test（域内对照，由 hf_boost.py 生成）
    test_split = "test_TN3K_hf"
    data_subpath = "./dataset/SAMUS/TN3K_hf/"
    result_path = "./result/TN3K_hf/"

class Config_DDTIz_hf(Config_DDTIz):
    # 方案E：高频幅度增益后的 DDTI test
    test_split = "test_DDTIz_hf"
    data_subpath = "./dataset/SAMUS/DDTIz_hf/"
    result_path = "./result/DDTIz_hf/"

class Config_PTz_hf(Config_PTz):
    # 方案E：高频幅度增益后的 private_Thyroid test
    test_split = "test_PTz_hf"
    data_subpath = "./dataset/SAMUS/PTz_hf/"
    result_path = "./result/PTz_hf/"

def get_config(task="US30K"):
    if task == "US30K":
        return Config_US30K()
    elif task == "TN3K":
        return Config_TN3K()
    elif task == "DDTIz":
        return Config_DDTIz()
    elif task == "PTz":
        return Config_PTz()
    elif task == "DDTIz_fda":
        return Config_DDTIz_fda()
    elif task == "PTz_fda":
        return Config_PTz_fda()
    elif task == "TN3K_hf":
        return Config_TN3K_hf()
    elif task == "DDTIz_hf":
        return Config_DDTIz_hf()
    elif task == "PTz_hf":
        return Config_PTz_hf()
    elif task == 'KTD':
        return  Config_KTD()
    elif task == "BUSI":
        return Config_BUSI()
    elif task == "CAMUS":
        return Config_CAMUS()
    elif task == "OurDataset":
        return Config_OurDataset()
    else:
        assert("We do not have the related dataset, please choose another task.")