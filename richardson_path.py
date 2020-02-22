import os

#setup paths
DATA_PATH = os.path.join("[target_dir", "validation]")
META_PATH = os.path.join("[target_dir", "validation]", "Validation Meta data")
TTC_PATH = os.path.join("[target_dir", "ttc_images")

META_FILE = "validation-annotations-bbox.csv"
CLASS_FILE = "class-descriptions-boxable.csv"

TRAIN_PATH = 'richardson_images_train_set'
TEST_PATH = 'richardson_images_test_set'
VALIDATION_PATH = 'richardson_images_validation_set'
INFO_PATH = 'richardson_info_files'

ATT_PATH = 'path_1_models'
ATT_ANNOTATION_PATH = 'annotations'
ATT_VALIDATION_FILE = 'captions_val2020.json'
ATT_TRAIN_FILE = 'captions_train2020.json'

OBJ_TEST_RESULTS = os.path.join('trained_model_results', 'obj_test_results')

DEBUG_MODE = False

# get the human labels
human_labels = {
#		"/m/02p0tk3" : "Human body"
   #     "/m/01g317" : "Person"
#		"/m/04yx4" : "Man",
#		"/m/03bt1vf": "Woman"
		"/m/0dzct" : "Human Face",
		"/m/04hgtk": "Human Head"
	}

#dimensions of input image
WINDOW_X = 128
WINDOW_Y = 128