# M5stack-zundamon

M5Stack Core2 1台で、話しかけると **聞き取って → ずんだもんLLMが返事を考えて → 画面に出して → ずんだもんの声で読み上げる** 試作です。
Wi-Fi・サーバー・SDカードは使いません。すべて Core2（ESP32、PSRAM 4MB、フラッシュ 16MB）の中で動きます。

版: **v0.0.1**（最初の試作。[CHANGELOG.md](CHANGELOG.md)）

## しくみ

```
マイク → 発話区間検出 → かな音声認識（kana_asr）→ 登録フレーズ照合
       → ずんだもんLLM（約70万パラメータ、気持ちタグつき）→ 画面に返事
       → 読み辞書でカタカナ＋簡易アクセント → VCV単位接続TTS → スピーカー
```

| 処理 | 時間（実機） |
|---|---|
| 聞き取り | 約0.3秒（話し終わりの検出 0.7秒は別） |
| 返事を作る | 約2〜2.5秒（10トークン前後） |
| 音声の合成 | 約0.08秒（2〜3秒の音声） |

## 必要なもの

- M5Stack Core2（16MB フラッシュ）と USB ケーブル
- [PlatformIO](https://platformio.org/)（`pip install platformio` か VS Code 拡張）
- モデルと音声データ（下記）

## モデルと音声データ

| ファイル | 中身 | リポジトリ |
|---|---|---|
| `models/zunda-small-tagend-679_q8.bin`, `models/tok_tagend.bin` | ずんだもんLLM（MIT） | 同梱 |
| `models/kana_asr.bin` | 音声認識の重み（約0.7MB。学習データは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)） | 同梱 |
| `voice/bank_mu.bin` | ずんだもんの VCV 単位音声（8bit μ-law、約9MB） | [Releases](https://github.com/crowpilot/M5stack-zundamon/releases) の `M5stack-zundamon-voice-bank.bin` を `voice/bank_mu.bin` として置く |

すぐ試すだけなら、Releases の `M5stack-zundamon-v0.0.1-core2.bin` を 0x0 に、`M5stack-zundamon-voice-bank.bin` を 0x490000 に書けば動きます（ビルド不要）。
```
esptool.py --chip esp32 --baud 460800 write_flash 0x0 M5stack-zundamon-v0.0.1-core2.bin 0x490000 M5stack-zundamon-voice-bank.bin
```

音声バンクを自分で作り直すとき（VOICEVOX Core 0.16 以降の Python 版が必要。`VV_DIR` に Core の場所）:
```
cd tools/voice
python gen_all.py                       # 単位生成（~/vv/units にキャッシュ、約15分）
python export_bank.py out/              # out/bank.bin（int16、18MB）
python make_bank_mu.py out/bank.bin ../../voice/bank_mu.bin
```

## 書き込み

```
cd firmware
pio run -t upload            # ファーム（models/ から重みのヘッダを自動生成）
../tools/flash_voice.sh      # 音声バンクを 0x490000 へ（初回とバンクを変えたときだけ、約2分）
pio device monitor           # 115200
```
音声バンクがなくても、画面表示までは動きます。

## 使い方

- 話しかけると「あなた：…」「ずんだもん：…〈気持ち〉」が出て、読み上げます
- ボタン A: 雑音レベルを測り直す / B: 直前の返事をもう一度読む / C: 同じ話しかけに別の返事（温度0.8）
- シリアル（115200）
  - `@こんにちは` … マイクを通さずにその文で返事させる
  - `!<joy>コンニチワ'、ボクモダイスキナ'ノダ` … 読み上げ記法をそのまま喋らせる
  - 1往復ごとに JSON を1行出す（聞き取り・返事・気持ち・記法・各時間・空きメモリ）

聞き取りで当てはまる登録フレーズは `firmware/src/phrases.h` の12個です。

## フォルダ

| パス | 内容 |
|---|---|
| `firmware/` | PlatformIO プロジェクト。`partitions.csv`（アプリ4.5MB＋音声11.4MB）、`gen_headers.py`（重み→ヘッダ） |
| `firmware/src/asr_core.*` | 音声認識（C99） |
| `firmware/src/llm_core.*`, `zunda_brain.*` | LLM 推論（llama2.c 由来）と、返事・気持ちタグの生成 |
| `firmware/src/yomi.*`, `yomi_dict.h` | 返事（漢字かな交じり）→ 読み上げ記法 |
| `firmware/src/tts.*`, `kana.*`, `moras.h` | VCV 単位接続TTS と記法パーサ |
| `tools/llm/` | PC で LLM を動かす確認用（テストベクタ照合・返事コーパス作り） |
| `tools/yomi/` | 読み辞書の作り直し（pyopenjtalk） |
| `tools/voice/` | 音声バンクの作り直し（VOICEVOX Core） |
| `tools/make_release.sh` | リリース用の書き込みファイルを `dist/` に作る |

## 既知の制限

- アクセントは簡易規則（句読点で区切り、「…ノダ」は ノ の前、他は平板）なので、抑揚が平たいです
- 辞書にない漢字は読まずに飛ばします
- 聞き取りは登録フレーズ以外だと崩れやすく、返事も噛み合いにくいです

## 版の付け方

[docs/VERSIONING.md](docs/VERSIONING.md)

## ライセンス・クレジット

コードは MIT（[LICENSE](LICENSE)）。同梱物と規約は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

**VOICEVOX:ずんだもん**　（ず・ω・きょ）

音声認識の学習には ReazonSpeech と VOICEVOX の各話者の合成音声を使っています（参照元とクレジットは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）。

ずんだもんは東北ずん子・ずんだもんプロジェクトのキャラクターです。非公式の非商用ファン作品として、
[キャラクター利用の手引き](https://zunko.jp/guideline.html) と [音源利用規約](https://zunko.jp/con_ongen_kiyaku.html) に沿って扱っています。
