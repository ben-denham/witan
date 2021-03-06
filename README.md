# Witan

Source-code for the Witan algorithm and accompanying experimentation
framework, as used in the paper "Witan: Unsupervised Labelling
Function Generation for Assisted Data Programming"

## Dependencies

In order to run this project, you must have the following dependencies
installed on your host:

* [Docker Community Edition](https://docs.docker.com/get-docker/) (>= 17.09)
* [Docker Compose](https://docs.docker.com/compose/install/) (>= 1.17)
  (Included with Docker Desktop on Mac/Windows)
* [Make](https://www.gnu.org/software/make/) (technically optional if
  you don't mind running the commands in the Makefile directly)

**Note:** If you use [Git bash](https://git-scm.com/downloads) on
Windows and also
[install `make`](https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058)
into Git bash, then you should be able to run this project on Windows.

## Reproducing Results

1. Ensure the dependencies listed above are installed.
2. Run `make run` in this directory.
   * This will perform all Docker image build steps and dependency
     installations every time you run it, so that you can never forget
     to rebuild. The first time you run this, it make take some time
     for the base Docker image and other dependencies to be
     downloaded.
3. Browse to http://localhost:9999 and enter the token displayed in
   the terminal (or just follow the link in the terminal).
4. Run the following Jupyter notebooks in the order listed here to
   reproduce the results:

| Notebook | Description |
| -------- | ----------- |
| [1_Datasets](http://localhost:9999/notebooks/1_Datasets.ipynb) | Summary of datasets. NOTE: Some datasets must be manually downloaded by the user. When such a dataset is referenced during execution, an exception will be raised informing the user where to download the dataset from, and where to place the results. |
| [2_RuntimeExperiments.ipynb](http://localhost:9999/notebooks/2_RuntimeExperiments.ipynb) | Experiments comparing method runtimes |
| [3_BinaryClassExperiments.ipynb](http://localhost:9999/notebooks/3_BinaryClassExperiments.ipynb) | Experiments with binary-class datasets |
| [4_MultiClassExperiments.ipynb](http://localhost:9999/notebooks/4_MultiClassExperiments.ipynb) | Experiments with multi-class datasets |
| [5_WitanLFs](http://localhost:9999/notebooks/5_WitanLFs.ipynb) | Examples of labelling functions generated with Witan |
| [6_AblationExperiments](http://localhost:9999/notebooks/6_AblationExperiments.ipynb) | Experiments with variants of the Witan algorithm |

### Linting

You can run [flake8](http://flake8.pycqa.org/en/latest/) linting with:
`make lint`.

### Testing

You can run [pytest](https://docs.pytest.org/en/latest/) unit tests
with: `make test`.

An HTML code-coverage reported will be generated for each module at:
`<module-dir>/test/coverage/index.html`.

### Type-Checking

You can run type checks with: `make types`.

## Managing Data

You may not want to commit the outputs of notebook cells to your Git
repository. If you have Python 3 installed, you can use
[nbstripout](https://github.com/kynan/nbstripout) to configure your
Git repository to exclude the outputs of notebook cells when running
`git add`:

1. `python3 -m pip install nbstripout nbconvert`
2. Run `nbstripout --install` in this directory (installs hooks into
   `.git`).

## Notes About Docker

### Opening a Shell

If you would like to open a bash shell inside the Docker container
running the Jupyter notebook server, use: `make bash` or `make
sudo-bash`. If `make run` is not currently running, you can instead
use `make run-bash` or `make run-sudo-bash`.

### System Dependencies and Other OS Configurations

To install system packages or otherwise alter the Docker image's
operating system, you can make changes in the Dockerfile. An example
section that will allow you to install `apt` packages is included.

### .dockerignore

Whenever the Docker image is rebuilt (after certain files are
changed), Docker will transmit the contents of this directory to the
Docker daemon.

To speed up build times, you should add an entry to the
`.dockerignore` file for any directories containing large files you do
not need to be included in the Docker image at build time.

### Managing Docker Image Storage

When Docker builds new versions of its images, it does not delete the
old versions. Over time, this can lead to a significant amount of disk
space being used for old versions of images that are no longer needed.

Because of this, you may wish to periodically run `docker image prune`
to delete any "dangling images" - images that aren't currently
associated with any image tags/names.

## Using Jupyter from Emacs

Did you know you can work with Jupyter notebooks from Emacs? All you
need to do is install
[EIN](http://millejoh.github.io/emacs-ipython-notebook/): `M-x
package-refresh-contents <Enter> M-x package-install <Enter> ein`

### Connecting EIN to your Jupyter server

1. Ensure `make run` is running.
2. `M-x ein:login` (URL: http://127.0.0.1:8888, Password: token from `make run`)
3. `M-x ein:notebooklist-open`

### Common EIN Commands

```
M-<enter> - Execute cell and move to next.
C-c C-c - Execute cell.
C-c C-z - Interrupt command
C-c C-x C-r - Restart session
C-<up/down> - Navigate cells.
M-<up/down> - Move cells.
C-c C-b - Insert cell below (C-a for above).
C-c C-l - Clear cell output.
C-c C-k - Delete cell.
C-c C-f - Open file.
C-c C-h - Help at cursor.
C-c C-S-l - Clear all output.
C-c C-t - Toggle cell type.
```
