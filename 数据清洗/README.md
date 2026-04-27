# EDGAR 文件分类清洗

目标：清洗 `data/10Ks/edgar` 里的 EDGAR 文件，并按文件类型分类放到 `output`。原始数据不修改，输出文件名保持原名。

## 运行

```powershell
python scripts/clean_edgar_purpose_text.py --input data/10Ks/edgar --output output/cleaned/edgar_by_type
```

## 输出结构

```text
output/cleaned/edgar_by_type/
├── metadata_json/       # *_metadata.json，保留有效元数据，删除 URL 和失效本地路径
├── submissions_json/    # submissions.json，只保留公司基础信息和 10-K 提交记录
├── html/                # *.htm，清洗为可读正文，文件名不变
└── txt/                 # *_full_submission.txt，抽取 10-K 正文并清洗，文件名不变
```

每个分类目录下继续保留公司目录，例如：

```text
output/cleaned/edgar_by_type/html/AAPL/2023_10K_2023-11-03.htm
```

清洗会去掉：

- URL
- HTML/XML 标签
- XBRL 标签和属性，如 `us-gaap:*`、`dei:*`、`xmlns`、`contextRef`、`unitRef`
- SEC/FASB/XBRL taxonomy 引用
- 隐藏 XBRL 事实、命名空间、内部 ID
- SEC 封面页模板句

