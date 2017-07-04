iris = {
    'name': 'iris',
    'workflow': 'miniramp.workflows.classification',
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
    ]
}
