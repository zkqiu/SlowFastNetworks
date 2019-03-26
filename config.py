params = dict()

params['num_classes'] = 10

#params['dataset'] = '/data1/data/Kinetics/Frames'
params['dataset'] = '/home/qzk/code'
params['epoch_num'] = 100
params['batch_size'] = 32
params['step'] = 10     #学习率衰减期
params['num_workers'] = 8
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 1
params['pretrained'] = None
params['gpu'] = [0,1,2,3]
params['log'] = 'log'
params['save_path'] = 'Kinetics'
params['clip_len'] = 64
params['frame_sample_rate'] = 2
params['dataset_ratio'] = 0.1

