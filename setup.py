from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


if __name__ == "__main__":
    setup(long_description=readme(), zip_safe=False)
