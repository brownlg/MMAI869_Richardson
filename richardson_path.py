#setup paths
DATA_PATH = "[target_dir/validation]/" 
META_PATH = "[target_dir/validation]/Validation Meta data"
META_FILE = "validation-annotations-bbox.csv"

TRAIN_PATH = 'richardson_images_train_set'
TEST_PATH = 'richardson_images_test_set'
VALIDATION_PATH = 'richardson_images_validation_set'
INFO_PATH = 'richardson_info_files'


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
WINDOW_X = 80
WINDOW_Y = 80