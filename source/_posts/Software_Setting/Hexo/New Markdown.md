---
title: How to Download Kaggle dataset in Linux server
date: 2023-03-1
mathjax: true
categories:
  - [Software Setting, Hexo]
tags:
  - Hexo
---



# Hexo在文章内部添加目录

## 1.安装hexo-toc插件

```css
npm install hexo-toc --save
```



## 2. 在主题中的_config.yml文件中加入

```yaml
toc:
  max_depth: 3
```



## 3. 在文章中添加

<!-- toc -->

```markdown
---
title: My Article
---

<!-- toc -->

## Section 1
...

```



## 4. 本地查看

```css
hexo generate
```

```css
hexo server
```



## 5. 效果

![](https://raw.githubusercontent.com/kathychen47/Img4KathyBlog/main/toc.png)



