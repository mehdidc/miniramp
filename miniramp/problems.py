iris = {
    'name': 'iris',
    'workflow': 'miniramp.workflows.classification',
    'workflow_options':{
        'final_model_strategy': 'retrain'
    },
    'data': {
        'name':  'miniramp.data.read_csv',
        'params':{
            'filename': 'https://raw.githubusercontent.com/ramp-data/iris/master/data/iris.csv',
            'y_col': 'species',
            'test_size': 0.3,
            'random_state': 42
        }
    },
    'validation' : {
        'name': 'miniramp.validation.kfold',
        'params':{
            'n_splits': 5
        }
    },
    'scores' : [
        'miniramp.scores.accuracy', 
        'miniramp.scores.log_loss'
    ],
}

cifar10 = {
    'name': 'cifar10',
    'workflow': 'miniramp.workflows.classification',
    'workflow_options':{
        'final_model_strategy': 'bagging'
    },
    'data': 'miniramp.data.cifar10',
    'validation' : {
        'name': 'miniramp.validation.shuffle_split',
        'params':{
            'n_splits': 1,
            'test_size': 0.1,
            'random_state': 42,
        }
    },
    'scores' : [
        'miniramp.scores.accuracy', 
        'miniramp.scores.log_loss',
    ],
}
