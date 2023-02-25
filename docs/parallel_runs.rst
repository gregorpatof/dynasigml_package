Running in parallel
===================

Since the typical application of DynaSig-ML involves computing thousands of Dynamical Signatures, we have made it simple
to parallelize the computation. Simply divide the list of PDB files in a number of sub-lists equal to the number
of parallel jobs you would like to run. In the **dynasigml_mir125a_example** repository, the make_jobs.py script is
already set up to generate 99 Slurm jobs that can be run in parallel, in a created folder called **parallel_jobs**.
It can easily be adapted to your parallel scheduler of choice if you cannot use Slurm. Each job should take somewhere
between 10-15 minutes to run.

The basic principle is to generate one DynaSigDF per sub-list of files, with all other parameters being the same
(beta_values, exp_measures, models). For example, the run_one_dynasigdf.py script does exactly that for the miR-125a
dataset, provided with an index from 0 to 98 (99 total parallel jobs) as its only command-line argument::

    from dynasigml.dynasig_df import DynaSigDF
    import sys
    import glob
    import numpy as np

    def load_data(filename):
        with open(filename) as f:
            lines = f.readlines()
        data_dict = dict()
        for line in lines[1:]:
            ll = line.split()
            data_dict[ll[2]] = [float(ll[0]), float(ll[1])]
        return data_dict


    if __name__ == "__main__":
        if len(sys.argv) != 2:
            raise ValueError("I need one argument: the job index")
        job_index = int(sys.argv[1])
        n_total_jobs = 99
        filenames_list = sorted(glob.glob("mir125a_variants/*.pdb"))
        step = int(len(filenames_list)/float(n_total_jobs) + 1)
        start = job_index * step
        stop = (job_index+1) * step
        sub_filenames_list = filenames_list[start:stop]
        data_dict = load_data('data_mir125.df')
        beta_values = [np.e ** (x / 2) for x in range(-6, 7)]
        exp_data = []
        for filename in sub_filenames_list:
            mutid = filename.split('.')[0].split('mir125a_')[-1]
            exp_data.append(data_dict[mutid])
        dsdf_name = "split_dsdfs/dsdf_{}".format(job_index)
        # eff is for maturation efficiency
        dsdf = DynaSigDF(sub_filenames_list, exp_data, ["eff", "mcfold_energy"], dsdf_name, beta_values=beta_values)

