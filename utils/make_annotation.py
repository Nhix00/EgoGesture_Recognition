import os


from dataset.egogesture_json import convert_egogesture_csv_to_activitynet_json


annotation_dir = "./annotation_egogesture"
classind_dir ="./classind"

def create_json(csv_dir_path, class_types):
    if class_types == 'all':
        class_ind_file = 'classIndAll.txt'
    elif class_types == 'all_but_None':
        class_ind_file = 'classIndAllbutNone.txt'
    elif class_types == 'binary':
        class_ind_file = 'classIndBinary.txt'


    label_csv_path = os.path.join("classind", class_ind_file)
    train_csv_path = os.path.join(csv_dir_path, 'trainlist'+ class_types + '.txt')
    val_csv_path = os.path.join(csv_dir_path, 'vallist'+ class_types + '.txt')
    test_csv_path = os.path.join(csv_dir_path, 'testlist'+ class_types + '.txt')
    dst_json_path = os.path.join(csv_dir_path, 'egogesture' + class_types + '.json')

    convert_egogesture_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                                val_csv_path, dst_json_path, test_csv_path)
    print('Successfully wrote to json : ', dst_json_path)


from dataset.ego_prepare import create_trainlist

def make_annotation():


    if(not os.path.exists(annotation_dir)):
        print("No directory for annotation founded, dtart generation...")
        os.mkdir(annotation_dir)
        create_trainlist("training", "trainlistall.txt", "all")
        create_trainlist("training", "trainlistall_but_None.txt", "all_but_None")
        create_trainlist("training", "trainlistbinary.txt", "binary")
        create_trainlist("validation", "vallistall.txt", "all")
        create_trainlist("validation", "vallistall_but_None.txt", "all_but_None")
        create_trainlist("validation", "vallistbinary.txt", "binary")
        create_trainlist("testing", "testlistall.txt", "all")
        create_trainlist("testing", "testlistall_but_None.txt", "all_but_None")
        create_trainlist("testing", "testlistbinary.txt", "binary")

    nfiles = os.listdir(annotation_dir)

    assert len(nfiles) >=9, "Files missing in "+annotation_dir
    
    assert os.path.exists(classind_dir), "Missing class index directory: "+classind_dir
    
    create_json(annotation_dir, "all", )
    create_json(annotation_dir, "all_but_None")
    create_json(annotation_dir, "binary")


    