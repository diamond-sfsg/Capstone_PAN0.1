# EDGAR 分类清洗规则

目标：把 `data/10Ks/edgar` 中的文件按类型清洗后放入 `output/cleaned/edgar_by_type`，原始数据不修改，输出文件名不改。

## 分类

- `metadata_json/`：清洗 `*_metadata.json`
- `submissions_json/`：清洗 `submissions.json`
- `html/`：清洗 `.htm`
- `txt/`：清洗 `_full_submission.txt`

每类目录下继续保留公司目录，例如 `html/AAPL/2023_10K_2023-11-03.htm`。

## JSON 清洗

`*_metadata.json` 保留公司、CIK、申报类型、日期、accession number、主文档名等有效字段，删除 URL 和失效本地路径。

`submissions.json` 保留公司基础信息和 `10-K` 提交记录，删除大段无关提交历史。

## 文本清洗

`.htm` 和 `_full_submission.txt` 清洗为可读正文，删除：

- URL
- HTML/XML 标签
- XBRL 标签和属性，如 `us-gaap:*`、`dei:*`、`xmlns`、`contextRef`、`unitRef`
- SEC/FASB/XBRL taxonomy 引用
- 隐藏 XBRL 事实、命名空间、内部 ID
- SEC 封面页模板句和明显目录噪声

