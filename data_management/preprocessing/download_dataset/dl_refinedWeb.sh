# リポジトリ丸ごととると巨大なので，特定のディレクトリのみをcloneする方式
# sparse-checkout参考: https://tech.asoview.co.jp/entry/2023/03/14/095235
# --filter=blob:none: clone したときにファイルの中身(blob)を DL させないようにする -> .git ディレクトリ削減
# --no-checkout: clone 時には.git ディレクトリを作成

# スクリプトを以下のように作成
```sh
git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/tiiuae/falcon-refinedweb

cd falcon-refinedweb
# 主なデータはdata にあるので， dataのみを取得 (DatasetsページのFiles and versionsで確認可能)
git sparse-checkout set data
```


# -------------- 失敗 --------------
# Download RefinedWeb dataset 
# git clone https://huggingface.co/datasets/tiiuae/falcon-refinedweb
# -> ./.git 1.6T, ./.data 1.5T 以上ある．途中で止め，再度上記コマンドはDL途中の途中からうまく再開しない(以下のエラー)
# fatal: destination path '.' already exists and is not an empty directory.