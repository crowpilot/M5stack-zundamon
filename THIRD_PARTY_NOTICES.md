# 第三者の著作物・規約

このリポジトリのコード（LICENSE、MIT）以外に、次のものを使っています。

| もの | 場所 | ライセンス・規約 |
|---|---|---|
| llama2.c（Andrej Karpathy） | `firmware/src/llm_core.c` の元 | MIT License, Copyright (c) 2023 Andrej |
| M5Unified / M5GFX（M5Stack） | ビルド時に取得（同梱しない）。日本語フォント efont を含む | MIT License（efont は BSD 系） |
| ずんだもんLLM（tiny-ja-llm、zunda-small-tagend-679） | `models/zunda-small-tagend-679_q8.bin`, `models/tok_tagend.bin` | MIT。学習データは Gemma 4（Apache 2.0）で生成 |
| 登録文のキャッシュ | `models/prompt_cache_679.bin` | 上の LLM で登録文を計算した途中結果（MIT） |
| 読み辞書 | `tools/yomi/yomi.tsv`, `firmware/src/yomi_dict.h` | pyopenjtalk（MIT）と Open JTalk 辞書（Modified BSD、NAIST-jdic 由来）で LLM の返事から取り出した読み |
| 音声認識の重み（kana_asr） | `models/kana_asr.bin` | 作者が学習したもの。学習データは下の「音声認識の学習データ」 |
| ずんだもんの音声（VCV単位バンク） | Releases の `M5stack-zundamon-voice-bank.bin` | VOICEVOX:ずんだもん。VOICEVOX Core で作った音声を短く切り出して並べたもの。[音源利用規約](https://zunko.jp/con_ongen_kiyaku.html) に従う |

## クレジット

VOICEVOX:ずんだもん

## 音声認識の学習データ

`models/kana_asr.bin` は次のデータで学習しています（データそのものは同梱していません）。

- **ReazonSpeech**（Reazon Holdings）: 日本語音声コーパス。https://research.reazon.jp/projects/ReazonSpeech/ ・
  https://huggingface.co/datasets/reazon-research/reazonspeech 。CDLA-Sharing-1.0、日本の著作権法30条の4に基づく情報解析目的での提供
- **VOICEVOX の合成音声**: 日本語の短文約200文を、VOICEVOX（0.25 系）の全話者のスタイルからランダムに選んで読ませたもの。
  クレジット: VOICEVOX:四国めたん、VOICEVOX:ずんだもん、VOICEVOX:春日部つむぎ、VOICEVOX:雨晴はう、VOICEVOX:波音リツ、
  VOICEVOX:玄野武宏、VOICEVOX:白上虎太郎、VOICEVOX:青山龍星、VOICEVOX:冥鳴ひまり、VOICEVOX:九州そら、VOICEVOX:もち子さん、
  VOICEVOX:剣崎雌雄、VOICEVOX:WhiteCUL、VOICEVOX:後鬼、VOICEVOX:No.7、VOICEVOX:ちび式じい、VOICEVOX:櫻歌ミコ、
  VOICEVOX:小夜/SAYO、VOICEVOX:ナースロボ＿タイプＴ、VOICEVOX:†聖騎士 紅桜†、VOICEVOX:雀松朱司、VOICEVOX:麒ヶ島宗麟、
  VOICEVOX:春歌ナナ、VOICEVOX:猫使アル、VOICEVOX:猫使ビィ、VOICEVOX:中国うさぎ、VOICEVOX:栗田まろん、VOICEVOX:あいえるたん、
  VOICEVOX:満別花丸、VOICEVOX:琴詠ニア、VOICEVOX:Voidoll、VOICEVOX:ぞん子、VOICEVOX:中部つるぎ、VOICEVOX:離途、
  VOICEVOX:黒沢冴白、VOICEVOX:ユーレイちゃん、VOICEVOX:東北ずん子、VOICEVOX:東北きりたん、VOICEVOX:東北イタコ、
  VOICEVOX:あんこもん、VOICEVOX:夜語トバリ、VOICEVOX:暁記ミタマ、VOICEVOX:里石ユカ
- **作者自身の声**の録音（Core2 のマイクで録ったもの）

ずんだもんは東北ずん子・ずんだもんプロジェクトのキャラクターです。このリポジトリは非公式のファン作品で、
[キャラクター利用の手引き](https://zunko.jp/guideline.html) に沿って扱っています。

## llama2.c のライセンス文

```
MIT License

Copyright (c) 2023 Andrej

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
