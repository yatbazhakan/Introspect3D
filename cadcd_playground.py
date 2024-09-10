from datasets.cadcd import cadcd
import argparse
from pathlib import Path
from cadcd_modules.generate_splits import generate_splits
import yaml
from easydict import EasyDict
from cadcd_modules.config import cfg_from_yaml_file, cfg_from_list, merge_new_config, log_config_to_file, cfg
from cadcd_modules.cadc_dataset import create_cadc_infos
from OpenPCDet.pcdet.datasets import build_dataloader
from OpenPCDet.tools.visual_utils.visualize_utils import draw_scenes


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="/media/samadi_a/ssd-roger/event/Introspect3D-test_distest/cadcd_modules/cadc_dataset.yaml", help='specify the config for training')
    args = parser.parse_args()
    return cfg_from_yaml_file(args.cfg_file, cfg)



if __name__ == '__main__':
    parse_config()

    # -----------------------create Infos---------------------------
    # dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
    # dataset_cfg = cfg
    # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    # create_cadc_infos(
    #     dataset_cfg=dataset_cfg,
    #     class_names=['Car', 'Pedestrian', 'Pickup_Truck'],
    #     data_path=ROOT_DIR / 'data' / 'cadc',
    #     save_path=Path(dataset_cfg.DATA_PATH)
    # )

    # -----------------------create dataset---------------------------
    dataset_cfg = cfg
    class_names = ['Car', 'Pedestrian', 'Pickup_Truck']
    dataset = cadcd(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        training=True,
        root_path=None,
        logger=None
        )
    print("Length of cadcd database: {}".format(len(dataset))) 
    
    # -----------------------create dataloader---------------------------
    dist_train = False
    merge_all_iters_to_one_epoch = False
    dataset_cfg = cfg
    class_names = ['Car', 'Pedestrian', 'Pickup_Truck']
    
    train_set, train_loader, train_sampler = build_dataloader(
        dataset = dataset,
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        batch_size=1,
        dist=dist_train, workers=50,
        logger=None,
        training=True,
        merge_all_iters_to_one_epoch=merge_all_iters_to_one_epoch,
        total_epochs=20
    )
    
    data_iter = iter(train_loader)
    x = next(data_iter)
    draw_scenes (
        points = x['points'],
        gt_boxes=x['gt_boxes'],
        ref_boxes=None,
        ref_scores=None,
        ref_labels=None
    ) 
    alaki = 1
    