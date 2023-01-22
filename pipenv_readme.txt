# to setup the pip environments used in this project execute the commands listed
# below (or run them in Colab directly in the jupyter notebook with the `!` 
# operator)

# cloning the repo
git clone https://github.com/benearnthof/open_science_mnist.git

# changing the working directory to the root of the git repo
cd /content/open_science_mnist
# print out the working directory to confirm
pwd

# installing pipenv
pip install --user pipenv

# on your local machine python will be required for this step
# on colab python is already installed alongside the OS 
# add the location of the pipenv scripts to $PATH (run in python3)
import os
os.environ['PATH'] += ':/root/.local/bin'

# this should now have /root/.local/bin added at the end
echo $PATH

# setting up a virtualenv for this project (should take about 3-5 minutes)
# this will check Pipfile.lock for dependencies
pipenv install
# if you wish to up- or downgrade individual packages make sure to use 
# pipenv install package==version
# to directly add the version to Pipfile and Pipfile.lock respectively

# to check out if the setup was successful and we obtained the required versions
pipenv graph
