# Curated Datasets for AI Engineers

Organized by domain and difficulty — with direct links and usage notes.

---

## How to Find Datasets

| Platform | Best For | URL |
|---|---|---|
| **Kaggle Datasets** | Competitions, tabular, CV, NLP | kaggle.com/datasets |
| **Hugging Face Datasets** | NLP, audio, CV, multimodal | huggingface.co/datasets |
| **UCI ML Repository** | Classical ML, tabular | archive.ics.uci.edu |
| **Papers with Code** | Benchmark datasets for SOTA | paperswithcode.com/datasets |
| **Google Dataset Search** | Any domain | datasetsearch.research.google.com |
| **AWS Open Data** | Large-scale, cloud-friendly | registry.opendata.aws |
| **Data.gov** | US government data | data.gov |

---

## Beginner Datasets (Modules 02–05)

| Dataset | Task | Rows | Features | Source |
|---|---|---|---|---|
| **Iris** | Multi-class classification | 150 | 4 | `sklearn.datasets.load_iris()` |
| **Titanic** | Binary classification | 891 | 11 | kaggle.com/c/titanic |
| **House Prices (Ames)** | Regression | 1,460 | 79 | kaggle.com/c/house-prices-advanced-regression-techniques |
| **Wine Quality** | Regression / classification | 6,497 | 11 | archive.ics.uci.edu/ml/datasets/Wine+Quality |
| **Diabetes (Pima)** | Binary classification | 768 | 8 | `sklearn.datasets.load_diabetes()` |
| **Boston Housing** | Regression | 506 | 13 | `sklearn.datasets.fetch_california_housing()` |
| **Breast Cancer** | Binary classification | 569 | 30 | `sklearn.datasets.load_breast_cancer()` |
| **SMS Spam Collection** | Text classification | 5,572 | text | archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection |

---

## Intermediate Datasets (Modules 06–10)

| Dataset | Task | Size | Source |
|---|---|---|---|
| **Telco Customer Churn** | Binary classification | 7,043 | kaggle.com/datasets/blastchar/telco-customer-churn |
| **Credit Card Fraud** | Anomaly detection | 284,807 | kaggle.com/datasets/mlg-ulb/creditcardfraud |
| **MovieLens 1M** | Recommendation | 1M ratings | grouplens.org/datasets/movielens/1m |
| **Mall Customer Segmentation** | Clustering | 200 | kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python |
| **NYC Taxi Trips** | Regression / EDA | 100M+ | nyc.gov / Kaggle |
| **Bike Sharing** | Regression / time series | 17,379 | archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset |
| **Heart Disease UCI** | Binary classification | 303 | archive.ics.uci.edu/ml/datasets/Heart+Disease |
| **Adult Income (Census)** | Binary classification | 48,842 | archive.ics.uci.edu/ml/datasets/Adult |
| **Amazon Reviews** | Sentiment / text | 3M+ | huggingface.co/datasets/amazon_us_reviews |
| **IMDB Movie Reviews** | Sentiment (binary) | 50,000 | huggingface.co/datasets/imdb |

---

## Computer Vision Datasets (Module 11)

| Dataset | Task | Images | Classes | Source |
|---|---|---|---|---|
| **MNIST** | Digit classification | 70,000 | 10 | `torchvision.datasets.MNIST` |
| **Fashion-MNIST** | Clothing classification | 70,000 | 10 | `torchvision.datasets.FashionMNIST` |
| **CIFAR-10** | Object classification | 60,000 | 10 | `torchvision.datasets.CIFAR10` |
| **CIFAR-100** | Fine-grained classification | 60,000 | 100 | `torchvision.datasets.CIFAR100` |
| **ImageNet** | Large-scale classification | 1.2M | 1,000 | image-net.org (requires registration) |
| **COCO** | Object detection / segmentation | 330,000 | 80 | cocodataset.org |
| **Open Images** | Detection / segmentation | 9M | 600 | storage.googleapis.com/openimages |
| **CelebA** | Face attributes | 202,599 | 40 attrs | mmlab.ie.cuhk.edu.hk/projects/CelebA.html |
| **Pascal VOC** | Object detection | 11,530 | 20 | host.robots.ox.ac.uk/pascal/VOC |
| **Stanford Dogs** | Fine-grained classification | 20,580 | 120 | vision.stanford.edu/acomp/data/stanford-dogs |
| **PlantVillage** | Plant disease detection | 54,306 | 38 | kaggle.com/datasets/emmarex/plantdisease |

---

## NLP Datasets (Module 12)

| Dataset | Task | Size | Source |
|---|---|---|---|
| **SQuAD 2.0** | Reading comprehension / QA | 150K Q&A pairs | rajpurkar.github.io/SQuAD-explorer |
| **GLUE Benchmark** | Multiple NLP tasks | Varies | gluebenchmark.com |
| **SuperGLUE** | Harder NLP benchmark | Varies | super.gluebenchmark.com |
| **AG News** | Text classification (4-class) | 120K | huggingface.co/datasets/ag_news |
| **Yelp Reviews** | 5-class sentiment | 650K | huggingface.co/datasets/yelp_review_full |
| **MultiNLI** | Natural language inference | 433K | huggingface.co/datasets/multi_nli |
| **XSum** | Abstractive summarisation | 226,711 | huggingface.co/datasets/xsum |
| **CNN/DailyMail** | Summarisation | 300K | huggingface.co/datasets/cnn_dailymail |
| **WMT14 En-De** | Machine translation | 4.5M pairs | huggingface.co/datasets/wmt14 |
| **CoNLL-2003** | Named Entity Recognition | 22,137 | huggingface.co/datasets/conll2003 |

---

## Time Series Datasets (Module 15)

| Dataset | Task | Frequency | Source |
|---|---|---|---|
| **M5 Forecasting (Walmart)** | Hierarchical sales forecasting | Daily | kaggle.com/c/m5-forecasting-accuracy |
| **Rossmann Store Sales** | Store sales forecasting | Daily | kaggle.com/c/rossmann-store-sales |
| **Air Passengers** | Classic forecasting benchmark | Monthly | `statsmodels.datasets.co2` |
| **Electricity** | Long-horizon forecasting | Hourly | archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams |
| **Beijing PM2.5** | Air quality forecasting | Hourly | archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data |
| **Yahoo Finance** | Stock price analysis | Daily | `yfinance` Python library |
| **Prophet Quick Start** | Trend + seasonality demo | Daily | `prophet.plot.plot_plotly_forecast` |

---

## Audio & Speech Datasets (Module 24)

| Dataset | Task | Hours | Language | Source |
|---|---|---|---|---|
| **LibriSpeech** | ASR | 960 hrs | English | openslr.org/12 |
| **CommonVoice** | ASR / multilingual | 20,000+ hrs | 100+ | commonvoice.mozilla.org |
| **VoxCeleb** | Speaker identification | 2,000+ hrs | English | robots.ox.ac.uk/~vgg/data/voxceleb |
| **ESC-50** | Environmental sound classification | 2,000 clips | — | github.com/karolpiczak/ESC-50 |
| **UrbanSound8K** | Urban sound classification | 8,732 clips | — | urbansounddataset.weebly.com |
| **AudioSet** | Audio classification (YouTube) | 5,000+ hrs | Mixed | research.google.com/audioset |
| **GTZAN** | Music genre classification | 1,000 clips | — | marsyasmusic.com/downloads.html |
| **LJSpeech** | TTS (speech synthesis) | 24 hrs | English | keithito.com/LJ-Speech-Dataset |

---

## Generative AI & LLM Datasets (Module 25)

| Dataset | Use Case | Source |
|---|---|---|
| **Alpaca** | Instruction fine-tuning | huggingface.co/datasets/tatsu-lab/alpaca |
| **OpenAssistant Conversations** | RLHF / chat fine-tuning | huggingface.co/datasets/OpenAssistant/oasst1 |
| **ShareGPT** | Chat fine-tuning | huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered |
| **Dolly 15k** | Instruction fine-tuning | huggingface.co/datasets/databricks/databricks-dolly-15k |
| **UltraFeedback** | DPO / preference data | huggingface.co/datasets/openbmb/UltraFeedback |
| **The Pile** | LLM pre-training | pile.eleuther.ai |
| **RedPajama** | Open LLM pre-training | huggingface.co/datasets/togethercomputer/RedPajama-Data-1T |
| **BEIR** | RAG / retrieval benchmarking | github.com/beir-cellar/beir |
| **HotpotQA** | Multi-hop QA for RAG | hotpotqa.github.io |

---

## Graph Datasets (Module 23)

| Dataset | Task | Nodes | Edges | Source |
|---|---|---|---|---|
| **Cora** | Node classification (papers) | 2,708 | 5,429 | `torch_geometric.datasets.Planetoid` |
| **CiteSeer** | Node classification | 3,327 | 4,732 | `torch_geometric.datasets.Planetoid` |
| **MUTAG** | Graph classification (molecules) | 188 graphs | — | `torch_geometric.datasets.TUDataset` |
| **REDDIT-BINARY** | Graph classification (social) | 2,000 graphs | — | `torch_geometric.datasets.TUDataset` |
| **Open Graph Benchmark (OGB)** | Various graph tasks | Large-scale | — | ogb.stanford.edu |
| **SNAP Social Networks** | Link prediction, community | Millions | Billions | snap.stanford.edu/data |

---

## Reinforcement Learning Environments

| Environment | Task | Library | Notes |
|---|---|---|---|
| **CartPole-v1** | Classic control | `gymnasium` | Balance a pole — starter RL env |
| **LunarLander-v2** | Continuous control | `gymnasium` | Land a rocket; good for PPO/SAC |
| **Atari Games** | Pixel-based control | `gymnasium[atari]` | Classic DQN benchmark |
| **MuJoCo Environments** | Robotics simulation | `gymnasium[mujoco]` | Standard continuous control benchmark |
| **PettingZoo** | Multi-agent RL | `pettingzoo` | Cooperative and competitive |

---

## Loading Data Efficiently

```python
# Hugging Face Datasets — streaming large datasets
from datasets import load_dataset

# Load normally
dataset = load_dataset("imdb")

# Stream a large dataset without downloading all
streamed = load_dataset("the_pile", split="train", streaming=True)
for example in streamed.take(100):
    print(example["text"][:100])

# Kaggle API
# 1. Install: pip install kaggle
# 2. Put kaggle.json in ~/.kaggle/
import subprocess
subprocess.run(["kaggle", "competitions", "download", "-c", "titanic"])

# sklearn built-in datasets
from sklearn.datasets import (
    load_iris, load_breast_cancer, load_wine,
    fetch_california_housing, fetch_openml,
    make_classification, make_regression
)

# Generate synthetic data for testing
X, y = make_classification(n_samples=10000, n_features=20,
                            n_informative=10, n_redundant=5,
                            n_classes=2, class_sep=1.5, random_state=42)
```
