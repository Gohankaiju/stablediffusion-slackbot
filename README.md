# Stablediffusion-Slackbot

Slack上でAI画像生成を行うボットです。任意のSDXL(SD1.5)モデルを使用して、指定したプロンプトに基づいて画像を生成します

slackチャンネル上での画像生成を可能とすることで、外出先や、資料作成の際に手軽に画像生成ができます

## 機能

- Slackコマンドを通じて画像生成
- カスタマイズ可能なパラメータ（サイズ、ネガティブプロンプト、CFGscale、Step、画像枚数）
- 生成された画像の自動アップロード
- 画像生成情報の保存とアップロード（オプション）

## 必要条件

- Python 3.8+
- Slack bot設定
     - socket mode を使用し、Event Subscriptionsを"Yes"に設定してください
     - slackbotに必要な権限: app_mentions:read, chat:write, files:write
- SDXL(SD1.5)での画像生成が可能なGPU

## インストール

1. リポジトリをクローン：

```shell
git clone https://github.com/Gohankaiju/stablediffusion-slackbot.git

cd stablediffusion-slackbot
```

2. ライブラリのインストール：

```shell
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

pip install --upgrade -r requirements.txt
```

3. 環境変数を設定：
`env.py`ファイルに slack api tokenを記載：

```python
SLACK_BOT_TOKEN='your_slack_bot_token'
SLACK_APP_TOKEN='your_slack_app_token'
```


4. モデルをダウンロード：
必要なモデルファイルを`/slack-bot/models/`ディレクトリに配置してください。
その後、`utils.py`の`MODEL_PATH`を編集します

## 使用方法

1. ボットを起動：

```shell
python run.py
```


2. Slackで以下のようにボットにメンションを送ります：

```shell
@test プロンプト | ネガティブプロンプト | 高さ | 幅 | CFG_scale | Step | 画像枚数

※ネガティブプロンプト、高さ、幅、CFG_scale 、Step、画像枚数 は省略可能です
  省略した場合はデフォルトの値が使用されます
```

```shell
@test 1girl | low quality | 1024 | 1024 | 7.5 | 30 | 4
```

3. オプションで`create_info`を追加すると、画像生成情報をテキストファイルとして出力します：
```shell
@test 1girl | low quality | 1024 | 1024 | 7.5 | 30 | 4 create_info
```

## 使用例
![dog](https://github.com/user-attachments/assets/049c01b4-a9e9-40b6-be9a-669dfc1eba60)

## カスタマイズ

- `utils.py`の`make_pipe`関数を編集して、異なるモデルや設定を使用できます
- `run.py`の`generate`関数のパラメータを調整して、デフォルトの生成設定を変更できます


## ライセンス

このプロジェクトは[MITライセンス](https://choosealicense.com/licenses/mit/)の下で公開されています。
