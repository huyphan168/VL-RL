from setuptools import setup

setup(
    name="gym-cards and virl",
    version="0.0.2",
    packages=['gym_cards', 'gym_virl', 'virl'],
    install_requires=["gymnasium", "numpy", "Pillow"]
)