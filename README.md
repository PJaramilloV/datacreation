# Dataset for automated point cloud object repair

## Members

- Joaquín Cruz
- Pablo Jaramillo
- Iván Sipirán (professor)

## Preprocess Routine

In order to make a dataset with this scheme you should run the preprocessing scripts as follows:

Before running a script we heavily suggest you read the available argument options on the ``argparse.ArgumentParser`` object

For every dataset source to use in your combined dataset you should run ``preproc_dataset``. This script will make ``.npy`` files holding point cloud data representative of object mesh files (.obj, .off, .stl).

```{bash}
~$ python path/to/preproc_dataset.py --datadir [path] --dataset [directory at [path]] 
```

Then you can collect the preprocessed data with ``collect_complete``. Modify the dataset_prefix dictionary with your own desired pairs of ``dataset names`` and ``prefixes`` for the objects once joined in a common ``collection`` folder. 

This script will create a ```collection`` directory in the dataset directory and some .csv files prefixed with ``__collection_`` necessary to represent the whole dataset and the train-test split, the split is generated automaticly where not provided in the shape of "train" and "test" subdirectories and inherited where those are provided. You run it without arguments as:

```{bash}
~$ python path/to/collect_complete.py
```

Finally you will run ``degrade_cloud_bottom`` to generate synthetic breaks on the geometries and split the data in newly created ``broken``, ``complete`` and ``repair`` directories at the same depth as ```collection``. In these directories we store the possibly augmented data, such that the objects to be repaired by an AI model are stored in ``broken``, the groud truth repairs are stored in ``repair``, and in ``complete`` whole copies of the object are stored for architectures which require these.

The cut provided by the script is a straight cut at a customizable and randomizable height and angle as allowed by the arguments: breakage, percentage for height of cut relative to the height of the object after a rotation; variance, multiplicative maximum both-ways deviation in height of cut; maxx, maxy & maxz, for maximum angle of rotation in x, y & z axis respectively.

Augment data with the n_breaks argument to make n different cuts to every object. Additionally this script generates voxel representation of broken and repair shapes.

This script will write at the end all the files generated in .csv files ``[dataset_name].csv``, ``train.csv``, and ``test.csv`` as given by the __colection .csv files. Run the script as follows:

```{bash}
~$ python path/to/degrade_cloud_bottom.py --dataset [path]
```

