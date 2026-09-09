from setuptools import setup, find_packages

setup(
    name="mgflow",
    version="2.0.0",
    description="Metallic Glass property prediction & inverse design (clean modular layout)",
    author="SciAgent",
    packages=find_packages(),
    package_data={
        "mgflow": [
            # ② 数据条目
            "data/elemental/*.npy",
            "data/datasets/*.npz",
            # ③ 正向预测产物
            "prediction/artifacts/pytorch_models/*.pt",
            "prediction/artifacts/pytorch_models/*.npz",
            "prediction/artifacts/pytorch_models/*.pkl",
            "prediction/artifacts/ensembles/*.pt",
            "prediction/artifacts/ensembles/*.npz",
            "prediction/artifacts/ensembles/*.json",
            # ④ 反向设计产物
            "inverse_design/vae_models/*.pt",
            "inverse_design/vae_models/*.json",
            "inverse_design/output/*.json",
            # 扩展：电阻率
            "resistivity/data/*.json",
            "resistivity/models/*.pt",
            "resistivity/models/*.npz",
            "resistivity/models/*.json",
        ],
    },
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
    ],
    extras_require={
        "prediction": ["torch", "openpyxl"],
        "classifier": ["scikit-learn"],
    },
    python_requires=">=3.9",
)
