from dataset.egogesture import EgoGesture
#from datasets.egogesture_online import EgoGestureOnline


def get_training_set(video_path, 
                     annotation_path,  
                     spatial_transform, 
                     temporal_transform,
                     target_transform,
                     modality = "RGB",
                     sample_duration = 16,):
    
    
    
    training_data = EgoGesture(
        video_path,
        annotation_path,
        "training",
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=sample_duration,
        modality=modality)
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['egogesture']
    
    if opt.dataset == 'egogesture':
        validation_data = EgoGesture(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration)
    return validation_data


def get_test_set(video_path, 
                 annotation_path,  
                 spatial_transform, 
                 temporal_transform, 
                 target_transform,
                 n_val_samples = 1,
                 test_subset = "test",
                 modality = "RGB",
                 sample_duration = 16,
                 ):
    assert test_subset in ['val', 'test']

    if test_subset == 'val':
        subset = 'validation'
    elif test_subset == 'test':
        subset = 'testing'
    test_data = EgoGesture(
        root_path=video_path,
        annotation_path=annotation_path,
        subset=subset,
        n_samples_for_each_video=n_val_samples,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        modality=modality,
        sample_duration=sample_duration)
    return test_data

'''def get_online_data(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in [ 'egogesture']
    online_data = EgoGestureOnline(
        opt.annotation_path,  
        opt.video_path,
        opt.whole_path,  
        opt.n_val_samples,
        spatial_transform,
        temporal_transform,
        target_transform,
        modality="RGB-D",
        stride_len = opt.stride_len,
        sample_duration=opt.sample_duration)
    
    return online_data'''
