import time
from pathlib import Path 
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import utilities as util
import pickle

# load data
data_dir = Path('./data/test')
list_ds, image_count = util.load_data(data_dir)

util.get_class_names(data_dir)
util.plot_ex_images(data_dir)

# allocate data for training and for validation
train_ds, val_ds = util.allocate_data(list_ds, image_count)

# set up parallel processing and performance
train_ds, val_ds = util.configuration(train_ds, val_ds)

num_epochs = 75
LOG_DIR = f"{int(time.time())}"

tuner = RandomSearch(
	util.build_hypermodel,
	objective='val_accuracy',
	max_trials=25,  # how many model variations to test
	executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
	directory=LOG_DIR)

tuner.search(train_ds,
             epochs=num_epochs,
             batch_size=64,
             validation_data=val_ds)

tuner.results_summary()
with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)


#model.save("cloud_model_" + str(num_epochs) + ".h5")
#util.plot_results(history, num_epochs)



