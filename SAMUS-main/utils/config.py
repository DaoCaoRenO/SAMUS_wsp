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


def _hf_variant_config(task):
    """动态注册 `Config_<Base>_hf<suffix>`：支持 hf_boost.py --suffix 产出的独立口径
    频域增强集（如 DDTIz_hf_g06），避免每换一个 gain 都要改本文件并覆写历史 _hf 数据。
    仅当 task 形如 <已注册task>_hf<非空后缀> 时生效，其余字段全部继承基类。"""
    i = task.find("_hf")
    if i <= 0 or i + 3 >= len(task):
        return None
    base, suffix = task[:i], task[i:]                 # 'DDTIz', '_hf_g06'
    base_cls = {"TN3K": Config_TN3K, "DDTIz": Config_DDTIz, "PTz": Config_PTz}.get(base)
    if base_cls is None:
        return None
    return type("Config_" + task, (base_cls,),
                {"test_split": "test_" + task,
                 "data_subpath": base_cls.data_path + task + "/",
                 "result_path": "./result/" + task + "/"})

# -------------------------------------------------------------------------------------------------
class Config_MultiOrgan:
    """Experiment A: USIC multi-organ (no thyroid) + TN3K → zero-shot thyroid test."""
    data_path = "./dataset/SAMUS/"
    data_subpath = "./dataset/SAMUS/TN3K/"  # val/test still from TN3K
    save_path = "./checkpoints/MultiOrgan/"
    result_path = "./result/MultiOrgan/"
    tensorboard_path = "./tensorboard/MultiOrgan/"
    load_path = save_path + "/SAMUS_best.pth"
    load_classifier_path = ""
    save_path_code = "_"

    workers = 0
    epochs = 100
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train_multiorgan"   # combined USIC+TN3K split
    val_split = "val"                  # TN3K val for best ckpt selection
    test_split = "test"                # not used directly; eval via rag_segment
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAMUS"
    segmentation_data_path = ""
class Config_MixedAll:
    """混合训练对照：TN3K train + DDTI(剔47测试) + private_Thyroid Train(剔44测试) 一起全参微调。
    非零样本(目标域标注进训练)；评测仍在 DDTIz/PTz/TN3K test 上跑，与零样本/E 结果同尺可比。"""
    data_path = "./dataset/SAMUS/"
    data_subpath = "./dataset/SAMUS/TN3K/"   # val/test 基础仍走 TN3K；跨域评测由 rag_segment --task 指定
    save_path = "./checkpoints/MixedAll/"
    result_path = "./result/MixedAll/"
    tensorboard_path = "./tensorboard/MixedAll/"
    load_path = save_path + "/SAMUS_best.pth"
    load_classifier_path = ""
    save_path_code = "_"

    workers = 0
    epochs = 50
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train_allsrc"    # 合并训练集(2803)
    val_split = "val"               # TN3K val 选 best ckpt
    test_split = "test"             # 不直接用；评测走 rag_segment
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAMUS"
    segmentation_data_path = ""


def get_config(task="US30K"):
    if task == "US30K":
        return Config_US30K()
    elif task == "TN3K":
        return Config_TN3K()
    elif task == "MultiOrgan":
        return Config_MultiOrgan()
    elif task == "MixedAll":
        return Config_MixedAll()
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
    elif "_hf" in task:
        dyn = _hf_variant_config(task)               # DDTIz_hf_g06 等独立后缀口径
        if dyn is not None:
            return dyn()
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