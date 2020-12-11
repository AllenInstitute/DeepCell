import pandas as pd


def get_experiment_genotype_map():
    experiment_metadata = pd.read_csv('ophys_metadata_lookup.txt')
    experiment_metadata.columns = [c.strip() for c in experiment_metadata.columns]
    return experiment_metadata[['experiment_id', 'genotype']].set_index('experiment_id') \
        .to_dict()['genotype']