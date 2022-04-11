from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
      name="DeepCell",
      use_scm_version=True,
      description=('CNN classifier for classifying ROIs in microscopy '
                   'recordings of mice brains'),
      author="Adam Amster",
      author_email="adam.amster@alleninstitute.org",
      url="https://github.com/AllenInstitute/DeepCell",
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      setup_requires=["setuptools_scm"],
      install_requires=required,
      include_package_data=True
)
