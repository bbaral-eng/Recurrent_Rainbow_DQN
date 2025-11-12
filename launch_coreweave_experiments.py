from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException
import yaml
import itertools
import uuid
import os
import argparse

def create_experiment_job(params, job_name):
    """Create a Kubernetes job specification for CoreWeave."""
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": "tenant-your-namespace"  # Replace with your CoreWeave namespace
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "rainbow-dqn",
                        "image": "your-docker-image:latest",  # Replace with your image
                        "command": [
                            "python",
                            "train_coreweave.py",
                            f"--lr={params['lr']}",
                            f"--gamma={params['gamma']}",
                            f"--n_step={params['n_step']}",
                            f"--alpha={params['alpha']}",
                            f"--beta={params['beta']}",
                            f"--prior_eps={params['prior_eps']}",
                            f"--seed={params['seed']}",
                            f"--total_timesteps={params['total_timesteps']}"
                        ],
                        "resources": {
                            "requests": {
                                "cpu": "4",
                                "memory": "16Gi",
                                "nvidia.com/gpu": "1"
                            },
                            "limits": {
                                "cpu": "8",
                                "memory": "32Gi",
                                "nvidia.com/gpu": "1"
                            }
                        }
                    }],
                    "restartPolicy": "Never"
                }
            }
        }
    }

# Define parameter search space (reduced + compatible with non-distributional, non-noisy code)
# We intentionally keep the grid small to avoid generating thousands of jobs.
parameter_space = {
    # learning rate — typical stable choices for DQN
    'lr': [1e-4, 2.5e-4],
    # discount factor
    'gamma': [0.99, 0.995],
    # n-step (1 is vanilla DQN, 3 is common practical choice)
    'n_step': [1, 3],
    # prioritized replay alpha: include 0 to disable PER entirely
    'alpha': [0.0, 0.3, 0.4, 0.5],
    # importance-sampling beta (only relevant if alpha > 0)
    'beta': [0.6],
    # small prior epsilon to avoid zero priority
    'prior_eps': [1e-6],
    'seed': [1],  # Fixed seed for reproducibility
    # reasonable short run for tuning; increase when you run final experiments
    'total_timesteps': [50000]
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-only', action='store_true', help='Only generate Kubernetes YAML files instead of submitting to cluster')
    parser.add_argument('--outdir', type=str, default='k8s_jobs', help='Directory to write generated job YAMLs')
    args = parser.parse_args()

    generate_only = args.generate_only
    outdir = args.outdir

    batch_v1 = None
    if not generate_only:
        # Try loading kube config; if it fails, fall back to YAML generation
        try:
            config.load_kube_config()
            batch_v1 = client.BatchV1Api()
        except ConfigException as e:
            print('Warning: could not load kube-config:', e)
            print('Falling back to YAML generation mode. Use --generate-only to skip attempting to contact the cluster.')
            generate_only = True

    # Generate all combinations of parameters
    keys, values = zip(*parameter_space.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create and submit jobs (or write YAMLs)
    os.makedirs(outdir, exist_ok=True)
    for i, exp in enumerate(experiments):
        job_name = f"rainbow-dqn-{uuid.uuid4().hex[:8]}"
        job = create_experiment_job(exp, job_name)

        if generate_only:
            # Write YAML file to outdir
            filename = os.path.join(outdir, f"{job_name}.yaml")
            with open(filename, 'w') as f:
                yaml.safe_dump(job, f)
            print(f'Wrote job YAML: {filename}')
        else:
            try:
                batch_v1.create_namespaced_job(
                    body=job,
                    namespace=job['metadata'].get('namespace', 'default')
                )
                print(f"Created job {job_name}")
            except Exception as e:
                print(f"Failed to create job {job_name}: {e}")

    if generate_only:
        print('\nAll job manifests were written to', outdir)
        print('Apply them with: kubectl apply -f', outdir)

if __name__ == "__main__":
    main()