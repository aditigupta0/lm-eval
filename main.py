import os
import argparse
import json
import re
import subprocess
from truefoundry.ml import get_client, ArtifactPath


def evaluate_model(model_name, task_name, batch_size):
    # run the process
    run_name = re.sub(r'[^a-zA-Z0-9]', '-', model_name+"-"+task_name)
    bashCommand = "lm_eval --model hf --model_args pretrained={model_name},dtype=float --tasks {task_name} --device cuda:0 --batch_size {batch_size} --output_path ./results --log_samples --limit 10  --wandb_args project=lm-eval-harness-integration,name={run_name}".format(
        model_name=model_name, task_name=task_name, batch_size=batch_size, run_name=run_name)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    client = get_client()
    # create an ML repo run
    run = client.create_run(ml_repo="lm-evaluation", run_name=run_name)
    run.log_params({
        "model_name": model_name,
        "task_name": task_name
    })

    # Log artifacts
    folder_path = os.getcwd() + "/results/" + re.sub(r'[^a-zA-Z0-9]', '-', model_name)
    all_files = os.listdir(folder_path)
    # pos = len(folder_path)        
    run.log_artifact(name=run_name, artifact_paths=[ArtifactPath(src=folder_path, dest=folder_path)])

    # Log metrics
    metric_file_name = max([f for f in all_files if os.path.isfile(os.path.join(folder_path, f)) and re.match("results_*", f)])
    with open(folder_path + "/" + metric_file_name) as f:
        d = json.load(f)
    metric_dict = dict()
    for task, dict_ in d['results'].items():
        for metric, value in dict_.items():
            if "none" in metric:
                metric_dict[metric.split(",")[0]] = value
    run.log_metrics(metric_dict)
    run.end()


if __name__ == "__main__":
    
    # Setup the argument parser by instantiating `ArgumentParser` class
    parser = argparse.ArgumentParser()
    # Add the parameters as arguments
    parser.add_argument(
        '--model_name', 
        type=str,
        required=True, 
        help='The huggingface name of the model that needs to be evaluated'
    )
    parser.add_argument(
        '--task_name',
        type=str, 
        help='The name of the task that needs to be evaluated'
    )
    parser.add_argument(
        '--batch_size', 
        type=str,
        required=True,  
        help='Batch Size'
    )
    # parser.add_argument(
    #     '--limit'
    #     type=str
    #     help="Specify the number of samples to test on like '--limit 10' if you need to test out the code."
    # )
    
    args = parser.parse_args()
    # Train the model
    evaluate_model(**vars(args))
    