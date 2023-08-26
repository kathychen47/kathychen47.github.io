---
title: How to Download Kaggle dataset in Linux server
date: 2023-03-1
mathjax: true
categories:
  - [Software Setting, Kaggle]
tags:
  - Kaggle
  - Linux
---

## 1. Install Kaggle in Linux

```python
pip install kaggle
```

## 2. Get Kaggle API

Kaggle -> Account -> Create New API Token -> download kaggle.json -> move kaggle. json file to home/user/.kaggle in linux

## 3. Change Download Directory

```python
kaggle config set -n path -v /data/XRay_data
```

## 4. Download Dataset

```python
kaggle competitions download -c rsna-2022-cervical-spine-fracture-detection

```
