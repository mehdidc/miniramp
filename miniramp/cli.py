import json
import time

from clize import run
from lightjob.cli import load_db

from miniramp.utils import import_object


def train(*, sampler='miniramp.samplers.classifier', problem='miniramp.problems.iris', save=False, db_path=None):
    problem_d = import_object(problem)
    
    workflow = problem_d['workflow']
    data = problem_d['data']
    validation = problem_d['validation']
    scores =  problem_d['scores']
    
    sampler_f = import_object(sampler)
    sample = sampler_f()
    codes = sample['codes']
    info = sample['info']

    wf = import_object(workflow)
    assert set(codes.keys()) == set(wf.requirements)

    start = time.time()
    out = wf(
        codes=codes, 
        data=data,
        validation=validation,
        scores=scores
    )
    end = time.time()
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
           start=start,
           end=end
        )

def main():
    run([train])
