---
title: he
date: 2023-08-26
mathjax: true
categories:
  - [Software Setting, Kaggle]
tags:
  - test
  - test1
---

# Add toc in the article

## 1. install hexo-toc plug-in

```css
npm install hexo-toc --save
```

## 2. add such code in the file of the theme

```YAML
toc:
  max_depth: 3
```

## 3. add such code in the article

```MARKDOWN
---
title: My Article
---

<!-- toc -->

## Section 1

...
```

## 4. generate the website to check

```css
hexo generate
```

```css
hexo server
```

## 5. Final effect

![](https://raw.githubusercontent.com/kathychen47/Img4KathyBlog/main/toc.png)
