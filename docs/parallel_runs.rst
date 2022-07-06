Running in parallel
===================

Since the typical application of DynaSig-ML involves computing thousands of Dynamical Signatures, it is very simple
to parallelize the computation. Simply divide the list of PDB files in a number of sub-lists equal to the number
of parallel jobs you would like to run. In the **dynasigml_vim2_example** repository, the make_jobs.py script is
already set up to generate 20 Slurm jobs that can be run in parallel, in a created folder called **parallel_jobs**.
It can easily be adapted to your parallel scheduler of choice if you cannot use Slurm. Each job should take somewhere
between 10-15 minutes to run.

The basic principle is to generate one DynaSigDF per sub-list of files, with all other parameters being the same
(beta_values, exp_measures, models). For example, the run_one_dynasigdf.py script does exactly that for the VIM-2
dataset, provided with an index from 0 to 19 (20 total parallel jobs) as its only command-line argument::

    from dynasigml.dynasig_df import DynaSigDF
    import sys
    import glob
    import json

    if __name__ == "__main__":
        if len(sys.argv) != 2:
            raise ValueError("I need one argument: the job index")
        job_index = int(sys.argv[1])
        n_total_jobs = 20
        filenames_list = sorted(glob.glob("mutants_modeller/*.pdb"))
        step = int(len(filenames_list)/float(n_total_jobs) + 1)
        start = job_index * step
        stop = (job_index+1) * step
        sub_filenames_list = filenames_list[start:stop]
        with open("vim2_data_dict.json") as f:
            data_dict = json.load(f)
        with open("vim2_exp_labels.json") as f:
            exp_labels = json.load(f)
        data_list = []
        for filename in sub_filenames_list:
            variant_id = filename.split('/')[-1].split('.')[0].split('_')[-1]
            data_list.append(data_dict[variant_id])
        dsdf_name = "separate_dsdfs/partial_dsdf_vim2_{}".format(job_index)
        dsdf = DynaSigDF(sub_filenames_list, data_list, exp_labels, dsdf_name)

