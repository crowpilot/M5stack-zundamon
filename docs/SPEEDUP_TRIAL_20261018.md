# Core2 応答時間短縮の試行（2026-10-18）

**実装・ホスト検証・実機書き込み・比較を完了した。5文のシリアル試験では、再生登録までが0.317〜0.491秒（19〜23%）短縮した。接続中のCore2には試行版0.0.3-s1を残している。**

## 作業場所と対象

- ブランチ: `codex/voice-latency-trial`
- 作業フォルダ: `/Users/matsuyamatoshiaki/workspace/M5stack-zundamon-speedup`
- 基準: v0.0.3、`f7d6a0c3acb81ccc94068ff96f109d3e6bbcf317`
- 試行版表示: `0.0.3-s1`。正式リリースのタグは付けていない。
- Core2 / USBシリアル `0225680D`。同一機体・同一モデル・同一音声バンクで比較。
- Claude側の `/Users/matsuyamatoshiaki/workspace/M5stack-zundamon` は別worktree。mainへのマージ・pushは行っていない。

先行調査は [SPEEDUP_REVIEW_20261018.md](SPEEDUP_REVIEW_20261018.md)。同資料の「未実装・未計測」は調査時点の記録で、今回実施した2点については本書を参照する。

## 実装

1. 起動時に `<BOS>Ａ：` を実際のトークナイザで処理し、先頭2位置のKVを用意。各リクエストの先頭トークンIDが一致する場合だけ既存領域から再利用する。追加のKVコピー領域は不要で、管理情報はint 3個。
2. プロンプトの途中では最終RMSNormと全語彙分類を省き、最後の位置だけlogitsを計算する。各層の計算とKV作成は従来どおり。

`llm_prefill()` は1・2トークンの短い入力では最後の位置を再計算し、古いlogitsを返さない。`llm_init()` はキャッシュを初期化し、`llm_forward()` で位置0・1を上書きした場合も再利用を無効にする。異なる先頭IDでは通常計算に戻る。モデル・KVをAPI外で直接書き換える場合は `prefix_len=0` にする必要がある。

返事の生成、コールバックの位置、声色の決定、音声データ、発話終了判定、ASR、クロック設定は変更していない。起動時に先頭の処理を前払いするため、その分だけ起動準備は増える。

シリアルの `ready.fast_prefill` と返答の `cached_prompt_tokens` で有効状態を確認できる。`m5stack-core2-reference` では最適化を無効にできる。

## 検証方法

- ホストではv0.0.3のソースをGitから取り出して別実行ファイルを作り、試行版・最適化無効版と比較。登録29文・追加3文・空入力・話者記号・長いASCII・長い日本語を往復順で実行した。
- 温度0/0.8、KV長256/64の組み合わせで、返事、気持ち、全コールバックのlogits、トークン数、乱数状態がビット単位で一致。短いKV長ではプロンプト切り詰めも通る。ASan/UBSanで確認した。
- 別の検証で、プロンプトの全KVと最後のlogitsの一致、短い入力、先頭ID変更、直接forward後の無効化、別インスタンス、不正入力を確認。
- Core2の通常版・比較版をビルド。
- 実機は `@文` のシリアル経路を利用。5文を同じ順で各3回。各版とも起動後最初の「おはよう」を含む。温度0。

`first_sound_ms` は処理開始から `M5.Speaker.playRaw()` 呼び出し直後まで。物理的な発声開始や、発話終了判定・音声認識は含まない。`first_piece_ms` も従来のコールバック到達時刻を維持した。

## 実機結果

各3回の中央値。単位はms。2つの最適化を合わせた結果で、個別の寄与は分離していない。

| 入力 | v0.0.3 再生登録まで | 試行版 再生登録まで | 短縮 | 短縮率 |
| --- | ---: | ---: | ---: | ---: |
| おはよう | 1,427 | 1,110 | 317 | 22.2% |
| こんにちは。 | 1,927 | 1,555 | 372 | 19.3% |
| 今日はいい天気ですね | 1,787 | 1,369 | 418 | 23.4% |
| 上司が厳しくて困ってる | 2,172 | 1,698 | 474 | 21.8% |
| テストで100点とったよ！ | 2,299 | 1,808 | 491 | 21.4% |

同じ文の3回の値の幅は、基準版が最大6ms、試行版が最大1ms。試行版の初回「おはよう」は1,111ms、以後は1,110msで、初回からキャッシュを使用した。

全比較で返答・気持ち・読み・声色・句数・音声長・生成トークン数が一致。読み飛ばしと不足音声は0、`cached_prompt_tokens` は2だった。通常利用状態のPSRAM空きは双方562,959B、内部RAM空きは基準136,048B・試行136,040B。起動直後のPSRAM表示だけ8,208B減るのは、従来初回エンコード時に作っていたトークナイザの索引を起動時に作るためで、返答後の空きは変わらない。

基準・試行とも、起動時のcore dump partition警告とマイク停止時の `I2S port 1 has not installed` ログが出る。既存のログであり、本試行では修正していない。試行版の起動、TTSバンク読み込み、各返答の完了を確認した。

記録:

- [実機基準値](measurements/20261018-prefill/baseline.jsonl) / [実機試行値](measurements/20261018-prefill/trial.jsonl)
- [中央値・範囲・各時間の集計](measurements/20261018-prefill/summary.json) / [起動情報](measurements/20261018-prefill/startup.json)
- [ホスト一致確認](measurements/20261018-prefill/host-check.txt) / [バイナリSHA256](measurements/20261018-prefill/SHA256SUMS.txt)

採用した実機基準値は `dist/latency-trial/baseline-complete.jsonl`。それ以前の取得はシリアルポートへの重複アクセスで中断したため除外し、バックアップ完了・ポート解放後に全測定をやり直した。試行は `dist/latency-trial/trial.jsonl`。上記Git管理ログには比較対象の応答行を保存し、起動情報を別ファイルに保存した。

## 再実行

```sh
cd /Users/matsuyamatoshiaki/workspace/M5stack-zundamon-speedup
python3 tools/llm/check_prefill.py
pio run -d firmware -e m5stack-core2
pio run -d firmware -e m5stack-core2-reference
```

シリアル計測にはpyserialが必要。このMacではPlatformIO付属Pythonから実行できる。計測中はモニターや書き込みツールを閉じること。計測スクリプトは既存の結果ファイルを上書きしない。

```sh
/opt/homebrew/Cellar/platformio/6.2.0_1/libexec/bin/python \
  tools/llm/serial_latency.py /dev/cu.usbserial-0225680D dist/new-run.jsonl
python3 tools/llm/compare_latency.py BASELINE.jsonl TRIAL.jsonl
```

## 元に戻す

書き込み前の16MBフラッシュ全体と、アプリ領域だけのコピーを `dist/latency-trial/` に保存した。バックアップのアプリ先頭はmainのv0.0.3ビルドと一致し、パーティションテーブルも確認済み。大きなバイナリはGit管理外。ハッシュは同フォルダの `SHA256SUMS.txt` を参照。

この試行で書き換えたのはアプリ領域だけで、esptoolのハッシュ検証も通った。元のアプリへ戻す場合は次を実行する。

```sh
cd /Users/matsuyamatoshiaki/workspace/M5stack-zundamon-speedup
/opt/homebrew/Cellar/platformio/6.2.0_1/libexec/bin/python \
  /Users/matsuyamatoshiaki/.platformio/packages/tool-esptoolpy/esptool.py \
  --chip esp32 --port /dev/cu.usbserial-0225680D --baud 921600 \
  write_flash 0x10000 dist/latency-trial/core2-before-app.bin
```

## 確認範囲

実際のマイク発話から聞こえるまでの時間、聴感上の音切れ、物理ボタン操作、長時間連続利用は、このシリアル比較だけでは確認できない。発話終了判定700msとASRは引き続き待ち時間に加わる。
