import os

def F_get_filename(audio_file, data_dir):
    """
    description:
        returns the full-ath to audio feature file for a given audio file
    inputs:
        - audio_file (full-path)
    outputs:
        - audio feature file (full-path)
    """
    # --- create folder if does not exist
    if not os.path.exists(data_dir):  os.makedirs(data_dir) 
    
    root = audio_file.split('/')[-1].split('.')[0]
    out_file = f'{data_dir}/{root}.npz'
    return out_file
