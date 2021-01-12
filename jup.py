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

model = util.build_base_model()
model = util.build_base_model_dropout()

num_epochs = 1

history = util.fit_model(train_ds, val_ds, model, num_epochs)
model.save("cloud_model_" + str(num_epochs) + ".h5")

plot_results(history, num_epochs)


