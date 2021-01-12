from pathlib import Path 
import utilities as util

# load data
data_dir = Path('/home/rey/ML/data/test')
list_ds, image_count = util.load_data(data_dir)

util.get_class_names(data_dir)
util.plot_ex_images(data_dir)

# allocate data for training and for validation
train_ds, val_ds = util.allocate_data(list_ds, image_count)

# set up parallel processing and performance
train_ds, val_ds = util.configuration(train_ds, val_ds)

model = util.base_model()

