# 版の付け方

## 版番号

[セマンティック バージョニング](https://semver.org/lang/ja/) の `メジャー.マイナー.パッチ`。1.0.0 までは試作扱い。

| 上げる場所 | いつ | 例 |
|---|---|---|
| パッチ `0.0.x` | 不具合直し・調整（しきい値、辞書の追加など） | 0.0.1 → 0.0.2 |
| マイナー `0.x.0` | 機能やモデルの入れ替え（LLM v4、表情、SDカード対応など） | 0.0.2 → 0.1.0 |
| メジャー `x.0.0` | 1.0.0 = 人に渡して普通に使える版。以後は書き込み手順や互換が変わるとき | 0.9.0 → 1.0.0 |

## 版を出す手順

1. `CHANGELOG.md` の `[Unreleased]` を `[0.0.2] - 日付` にして中身を書く
2. `firmware/platformio.ini` の `-DZUNDA_VERSION=\"0.0.2\"` を直す（起動画面とシリアルの `ready` に出る）
3. コミットして注釈付きタグを付ける
   ```
   git commit -am "Release v0.0.2"
   git tag -a v0.0.2 -m "v0.0.2"
   git push origin main --tags
   ```
4. `tools/make_release.sh` で `dist/` を作り、GitHub Releases の v0.0.2 に添付する
   - `M5stack-zundamon-v0.0.2-core2.bin`（0x0 に書く。esptool / [ESP Web Tool](https://espressif.github.io/esptool-js/) で書ける）
   - `SHA256SUMS.txt`
   - `M5stack-zundamon-voice-bank.bin`（音声バンクを変えたときだけ）

## ブランチ

一人で作っている間は `main` だけでよい。試したいことは `feat/…` ブランチで作り、動いたら `main` にまとめる。

## 重みと音声データ

大きなバイナリ（音声バンク）は git に入れず（`.gitignore`）、Releases に添付する。
モデルを変えたら `CHANGELOG.md` にモデル名を書く（返事の中身が変わるため）。
