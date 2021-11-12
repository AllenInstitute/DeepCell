from setuptools import setup

setup(
      name="DeepCell",
      use_scm_version=True,
      description=('CNN classifier for classifying ROIs in microscopy '
                   'recordings of mice brains'),
      author="Adam Amster",
      author_email="adam.amster@alleninstitute.org",
      url="https://github.com/AllenInstitute/DeepCell",
      package_dir={"": "src"},
      setup_requires=["setuptools_scm"]
)
