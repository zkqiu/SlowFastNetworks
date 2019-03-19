params = dict()

params['num_classes'] = 101

params['dataset'] = '/data1/data/UCF101'

params['epoch_num'] = 20
params['batch_size'] = 64
params['step'] = 10
params['num_workers'] = 4
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 10
params['pretrained'] = None
params['gpu'] = [0,1,2,3]
params['log'] = 'log'
params['save_path'] = 'UCF101'
params['clip_len'] = 64
params['frame_sample_rate'] = 1
params['dataset_ratio'] = 0.1

