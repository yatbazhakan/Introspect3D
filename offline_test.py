from datasets.activation_dataset import ActivationDataset
config = {
      'root_dir': '/mnt/ssd2/custom_dataset/vehicle_centerpoint_activations_aggregated_raw/',
      'label_file': 'vehicle_centerpoint_labels_aggregated_raw_filtered.csv',
      'classes': ['No Error', 'Error'],
      'label_field': 'is_missed',
      'layer': 0,
      'is_multi_feature': False,
      'name': 'custom'}
if __name__ == "__main__":
    
    dataset = ActivationDataset(config)
    