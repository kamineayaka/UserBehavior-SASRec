# 发布到 GitHub（首次）

本地独立仓库已初始化并完成首次 commit。按下列步骤在 GitHub 创建远程并推送。

## 1. 在 GitHub 创建空仓库

1. 打开 https://github.com/new  
2. Repository name：`UserBehavior-SASRec`（或与下方 remote URL 一致）  
3. 选择 **Public**  
4. **不要**勾选 “Add a README”（避免与本地冲突）  
5. 点击 Create repository  

## 2. 推送本地代码

在仓库根目录执行（remote 已配置为 `kamineayaka/UserBehavior-SASRec`，若用户名或仓库名不同请先 `git remote set-url`）：

```bash
git push -u origin main
```

## 3. 团队克隆

```bash
git clone https://github.com/kamineayaka/UserBehavior-SASRec.git
cd UserBehavior-SASRec
pip install -r requirements.txt
```

训练数据 `data/*.parquet` 不在 Git 中，请从 [GitHub Releases](https://github.com/kamineayaka/UserBehavior-SASRec/releases) 下载，或运行 `python scripts/download_release_assets.py`（见 [docs/使用指南.md](docs/使用指南.md)）。

## 与 monorepo 的关系

- 全量开发仓库：`UserBehavior_Analysis`（根目录 `.gitignore` 已忽略 `SASRec/`，避免重复跟踪）  
- 团队复现仓库：本目录独立 `git`，单独 push  

同步算法包：`python scripts/sync_sasrec_core.py --force`（源为上级 monorepo 的 `sasrec_core/`）
