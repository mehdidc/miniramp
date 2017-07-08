import warnings
import json
from datetime import datetime
import traceback

from clize import run

from lightjob.cli import load_db
from lightjob.db import SUCCESS, ERROR

from miniramp.utils import import_object


def train(*, sampler='miniramp.samplers.classifier', problem='miniramp.problems.iris', save=False, db_path=None):
    problem_d = import_object(problem)
    
    workflow = problem_d['workflow']
    options = problem_d['workflow_options']
    data = problem_d['data']
    validation = problem_d['validation']
    scores =  problem_d['scores']
    
    sampler_f = import_object(sampler)
    sample = sampler_f()
    codes = sample['codes']
    info = sample['info']

    wf = import_object(workflow)
    assert set(codes.keys()) == set(wf.requirements)

    start = datetime.now()
    try:
        out = wf(
            codes=codes, 
            data=data,
            validation=validation,
            scores=scores,
            options=options
        )
    except Exception as ex:
        traceback = _get_traceback() 
        warnings.warn('Raised an exception : {}. Putting state to error'.format(ex))
        state = ERROR
        out = {}
    else:
        state = SUCCESS
        traceback = '' 
    end = datetime.now()
    content = {
        'codes': codes,
        'info': info,
        'problem': problem,
        'sampler': sampler,
    }
    print(json.dumps(content, indent=2))
    print(json.dumps(out, indent=2))
    if save:
        db = load_db(db_path)
        db.safe_add_job(
           content,
           problem=problem,
           sampler=sampler,
           stats=out,
           start=str(start),
           end=str(end),
           duration=(end-start).total_seconds(),
           state=state,
           traceback=traceback
        )


def _get_traceback():
    lines  = traceback.format_exc().splitlines()
    lines = '\n'.join(lines)
    return lines


def main():
    run([train])
