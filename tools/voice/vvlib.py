import json, dataclasses
from pathlib import Path
from voicevox_core import UserDictWord
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile, UserDict
# 辞書に無いキャラ名など（アクセント型は 0=平板）
USER_WORDS = [('ずんだもん', 'ズンダモン', 0), ('ずんだ', 'ズンダ', 0)]
import os, sys as _sys
VC = Path(os.environ.get("VV_DIR", Path.home()/"vv/vc"))   # downloader の出力先
_ORT = "libvoicevox_onnxruntime.1.17.3.dylib" if _sys.platform == "darwin" else "libvoicevox_onnxruntime.so.1.17.3"
_syn = None
def syn():
    global _syn
    if _syn is None:
        ort = Onnxruntime.load_once(filename=str(VC/"onnxruntime/lib"/_ORT))
        oj = OpenJtalk(str(VC/"dict/open_jtalk_dic_utf_8-1.11"))
        ud = UserDict()
        for sf, pr, ac in USER_WORDS:
            ud.add_word(UserDictWord(sf, pr, ac, word_type='PROPER_NOUN', priority=8))
        oj.use_user_dict(ud)
        _syn = Synthesizer(ort, oj, cpu_num_threads=int(__import__("os").environ.get("VVT","4")))
        with VoiceModelFile.open(str(VC/"models/vvms/0.vvm")) as m:
            _syn.load_voice_model(m)
    return _syn
