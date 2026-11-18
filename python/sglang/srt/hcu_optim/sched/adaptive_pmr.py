"""按请求长度动态决定「一次预填充放几条请求」。

实测依据（2026-09-12，frozen5 基线）：同一个 --prefill-max-requests，最优值随输入长度反向变化：
    1024 c16  pmr=1 → TTFT +15.7%（更差）    2048 c16  pmr=1 → +5.5%（更差）
    16384 c16 pmr=1 → −2.8%                  32768 c4/c16 → −21.3% / −11.2%
    65536 c8  → −6.5%                        131072 c2 → −1.9%
机制：长请求本身要切多块 → 串行做完一条再做下一条，前面的请求早拿到首 token，平均首响降；
     短请求远小于一块 → 几条凑进同一次预填充能摊掉固定开销，拆开反而每条都付一次。

改动点【核 schedule_policy.py:993】：
  `if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x: return OTHER`
  在调用前按本批类型改 self.prefill_max_requests（PrefillAdder 每轮新建，改动不跨轮）。

=====================================================================
v3：修复「续块泄漏」（2026-09-14 源码定位，对应 G5）
=====================================================================
v2 的判定条件是 `len(self.can_run_list) == 0`。但亲读源码确认：

  scheduler.py:3027    有续块时先调 adder.add_chunked_req(self.chunked_req)
  schedule_policy.py:828  add_chunked_req 内 `self.can_run_list.append(req)`

⇒ **续块在任何 add_one_req 之前就已入列**，故续块轮里 `len(can_run_list)==0` 恒为 False
  → 判定分支不执行 → PrefillAdder 是新实例、`_apmr_long` 不存在
  → `getattr(self,"_apmr_long",False)` 取 False → 走短请求分支、限额取 2 而非 1。

后果：一条 32768 请求切 5 块（1 首块 + 4 续块），**本补丁在 80% 的轮次里失效**。
而 pmr=1 在 32768 c4 上实测值 −21.3%，那正是全表最差的一行（O 68.8）。

修复：有续块即判为长模式——能被切块的请求必然 ≥ chunked_prefill_size(8192) > 阈值(4000)。
开关 SGLANG_ADAPTIVE_PMR_FIXCHUNK=1 启用（默认 0，保持 v2 行为以便 A/B）。

预期：O 上行（报告估计 +3~8.9），P 下行（−0.5~−4）。
我们 O=88.20 不达标、P=108.97 有 8.97 点余量，这笔交易方向正确。
"""
import os

_ON = os.environ.get("SGLANG_ADAPTIVE_PMR", "0") != "0"
_THRESH = int(os.environ.get("SGLANG_ADAPTIVE_PMR_THRESH", "8192"))
_LONG = int(os.environ.get("SGLANG_ADAPTIVE_PMR_LONG", "1"))
_SHORT_SET = "SGLANG_ADAPTIVE_PMR_SHORT" in os.environ
_SHORT = int(os.environ.get("SGLANG_ADAPTIVE_PMR_SHORT", "2"))
_FIXCHUNK = os.environ.get("SGLANG_ADAPTIVE_PMR_FIXCHUNK", "0") != "0"
_st = {"n_long": 0, "n_short": 0, "n_cont": 0, "logged": False, "cont_logged": False}


def install(mod):
    if not _ON or getattr(mod, "_adaptive_pmr_installed", False):
        return
    PA = getattr(mod, "PrefillAdder", None)
    if PA is None or not hasattr(PA, "add_one_req"):
        print("[adaptive_pmr] 未找到 PrefillAdder.add_one_req，不安装", flush=True)
        return
    orig = PA.add_one_req

    def add_one_req(self, req, *a, **kw):
        try:
            if getattr(self, "_apmr_base", None) is None:
                self._apmr_base = self.prefill_max_requests
            if not hasattr(self, "_apmr_long"):
                if self.can_run_list:
                    # 续块轮：can_run_list 里已有续块（add_chunked_req 先于本调用）
                    if _FIXCHUNK:
                        self._apmr_long = True
                        _st["n_cont"] += 1
                        if not _st["cont_logged"]:
                            _st["cont_logged"] = True
                            print("[adaptive_pmr] ✅ 续块修复首次生效：本轮有续块 → 判为长模式"
                                  f"（限额 {_LONG}）", flush=True)
                    # _FIXCHUNK=0 时不设属性，保持 v2 行为（下面 getattr 取 False → 短模式）
                else:
                    need = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
                    self._apmr_long = need >= _THRESH
                    if not _st["logged"]:
                        _st["logged"] = True
                        print(f"[adaptive_pmr] ✅ 首次生效：首条待填 {need} token → 本批"
                              f"{'长' if self._apmr_long else '短'}模式（阈值 {_THRESH}）", flush=True)
            if getattr(self, "_apmr_long", False):
                self.prefill_max_requests = _LONG
                _st["n_long"] += 1
            else:
                base = self._apmr_base if self._apmr_base is not None else _SHORT
                self.prefill_max_requests = _SHORT if _SHORT_SET else base
                _st["n_short"] += 1
        except Exception as e:
            if not _st.get("err_logged"):
                _st["err_logged"] = True
                print(f"[adaptive_pmr] ⚠ 判定失败，本请求退回原限额: {type(e).__name__}: {e}", flush=True)
        return orig(self, req, *a, **kw)

    PA.add_one_req = add_one_req
    mod._adaptive_pmr_installed = True
    print(f"[adaptive_pmr] 已安装 阈值={_THRESH} 长={_LONG} 短={_SHORT} "
          f"续块修复={'开' if _FIXCHUNK else '关'} (pid={os.getpid()})", flush=True)
