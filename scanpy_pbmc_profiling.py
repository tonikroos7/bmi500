# %%

from typing_extensions import ParamSpecArgs
import numpy as np
import pandas as pd
import scanpy as sc

import time
import sys
import argparse
import cProfile

# %%
start_time = time.time()

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
# sc.settings.n_jobs = int(sys.argv[4])
sc.settings.n_jobs = 1

print(f"using {sc.settings.n_jobs} threads")

end_time = time.time()
print(f"Setting params elapsed time: {end_time - start_time:.4f} seconds")

# %%
start_time = time.time()

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--data-dir', type=str, help='Directory containing the dataset subdirectories', default='data')
parser.add_argument('--data-set', type=str, help='Dataset name, which is the subdirectory name', default='pbmc3k')
parser.add_argument('--out-dir', type=str, help='Output directory', required=False, default='data')
parser.add_argument('--num-threads', type=int, help='Number of threads', default=1, required=False)

args = parser.parse_args()

datadir = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
dataset = args.data_set 
outdir = args.out_dir if args.out_dir.endswith('/') else args.out_dir + '/'
nthreads = args.num_threads

end_time = time.time()
print(f"Argument processing elapsed time: {end_time - start_time:.4f} seconds")

#%%

# I/O
start_time = time.time()

results_file = "/".join([outdir, dataset + '.scanpy.h5ad'])  # the file that will store the analysis results

adata = sc.read_10x_mtx(
    #'/nethome/tpan7/scgc/data/' + dataset + '/filtered_gene_bc_matrices/hg19',  # the directory with the `.mtx` file
    "/".join([datadir, dataset, 'filtered_gene_bc_matrices']),  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)                              # write a cache file for faster subsequent reading

adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`

end_time = time.time()
print(f"I/O elapsed time: {end_time - start_time:.4f} seconds")

# %%
# preprocessing
start_time = time.time()
# basic filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
end_time = time.time()
print(f"Preprocessing elapsed time: {end_time - start_time:.4f} seconds")

#%%
start_time = time.time()
# metric
#adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
#sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# filtering by slicing the AnnData object
#adata = adata[adata.obs.n_genes_by_counts < 2500, :]
#adata = adata[adata.obs.pct_counts_mt < 5, :]


# and normalize to 10K reads per cell
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

end_time = time.time()
print(f"Normalization elapsed time: {end_time - start_time:.4f} seconds")

# %%
# highly variable genes
start_time = time.time()
#sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

# freeze data.
adata.raw = adata

# filtering by highly variable genes.
adata = adata[:, adata.var.highly_variable]

end_time = time.time()
print(f"Filtering elapsed time: {end_time - start_time:.4f} seconds")

#%%
start_time = time.time()
# regres out effects of total counts per cell an d% mitochondrial genes
#sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata)
end_time = time.time()
print(f"Regres out effects elapsed time: {end_time - start_time:.4f} seconds")

# %%
# report adata - so we can check ot see if we are comparable to Seurat
# adata.write(results_file)
# adata

# %%
# pca.  parallel via OMP_NUM_THREADS
start_time = time.time()
sc.tl.pca(adata, svd_solver='arpack', n_comps=30)
end_time = time.time()
print(f"PCA elapsed time: {end_time - start_time:.4f} seconds")
# adata.write(results_file)
# adata

# %%
# neighborhood graph
start_time = time.time()
sc.pp.neighbors(adata, n_pcs=30)
end_time = time.time()
print(f"Neighborhood graph elapsed time: {end_time - start_time:.4f} seconds")
# %% 
# for fixing disconnected clusters or connectivity issues:
#sc.tl.paga(adata)
#sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
#cs.tl.umap(adata, init_pos='paga')


# adata.write(results_file)
# adata


# %%
# clustering  (currently uses leiden,  previously using louvain (like Seurat).)
#sc.tl.leiden(adata)
start_time = time.time()
sc.tl.louvain(adata, resolution = 0.5)
end_time = time.time()
print(f"Clustering elapsed time: {end_time - start_time:.4f} seconds")

#%%
# umap
start_time = time.time()
sc.tl.umap(adata, n_components=30)
end_time = time.time()
print(f"Umap elapsed time: {end_time - start_time:.4f} seconds")

#%%
start_time = time.time()
adata.write(results_file)
adata
end_time = time.time()
print(f"Writing elapsed time: {end_time - start_time:.4f} seconds")

# %%
# support t-test, wilcoxon, logistic regression
# find marker genes
start_time = time.time()

cProfile.run("sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', use_raw=True)", sort="cumulative")


end_time = time.time()
print(f"Gene finding elapsed time: {end_time - start_time:.4f} seconds")
