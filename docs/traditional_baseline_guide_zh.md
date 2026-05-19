# Traditional Baseline 浣跨敤璇存槑

杩欎唤鏂囨。鍐欑粰璐熻矗浼犵粺浜嬩欢琛ㄧず baseline 鐨勫悓瀛︺€傜洰鏍囨槸鎶婁紶缁熸柟娉曟帴鍒板拰
learning-based 鏂规硶鐩稿悓鐨?benchmark 娴佺▼閲岋紝鏂逛究鏈€鍚庡仛鍏钩瀵规瘮銆?

## 鐜板湪鏀寔鍝簺浼犵粺琛ㄧず

褰撳墠宸茬粡瀹炵幇浜?5 绫诲父鐢ㄤ紶缁熶簨浠惰〃绀猴細

| 鏂规硶鍚?| 鍚箟 | 杈撳嚭閫氶亾 |
|---|---|---:|
| `event_frame` / `event_count` | 姝ｈ礋鏋佹€т簨浠惰鏁板浘 | 2 |
| `binary_event_image` | 姝ｈ礋鏋佹€т簩鍊间簨浠跺浘锛屾湁浜嬩欢灏辨槸 1 | 2 |
| `timestamp_image` | 姣忎釜鍍忕礌鏈€杩戜竴娆′簨浠剁殑褰掍竴鍖栨椂闂?| 2 |
| `time_surface` | 鎸囨暟琛板噺鏃堕棿琛ㄩ潰锛岃〃绀轰簨浠舵湁澶氣€滄柊鈥?| 2 |
| `voxel_grid` | 鎸夋椂闂村垎 bin 鐨勪綋绱犵綉鏍硷紝榛樿 5 涓?bin | 10 |

鎵€鏈夋柟娉曢兘浣跨敤鍚屼竴涓緭鍏ヨ緭鍑烘帴鍙ｏ細

```text
杈撳叆: events, shape = Nx4, columns = [x, y, t, p]
杈撳嚭: representation, shape = CxHxW, dtype = float32
```

閫氶亾椤哄簭缁熶竴涓猴細姝ｆ瀬鎬у湪鍓嶏紝璐熸瀬鎬у湪鍚庛€備緥濡?`voxel_grid` 鐨勫墠 5 涓?
channel 鏄鏋佹€ф椂闂?bin锛屽悗 5 涓?channel 鏄礋鏋佹€ф椂闂?bin銆?

## 瑕嗙洊鍝簺浠诲姟

traditional baseline 璁″垝瑕嗙洊 4 鏉＄嚎锛?

| 鏁版嵁闆?| 浠诲姟 | 涓嬫父妯″瀷 |
|---|---|---|
| N-MNIST | 鍒嗙被 | ResNet18 |
| N-Caltech101 | 鍒嗙被 | ResNet18 |
| GEN1 | 鐩爣妫€娴?| YOLOv6 |
| MVSEC | 鍏夋祦浼拌 | EV-FlowNet-like decoder |

鐩墠浠ｇ爜宸茬粡瀹屾垚杩欎簺鎺ュ叆锛?

- N-MNIST / N-Caltech101: `train_traditional_classification.py`
- Prophesee mini detection: `scripts/detection/prophesee/train.py --method <traditional_method>`
- MVSEC optical flow: `optical-flow/scripts/run_original_protocol.py --adapter <traditional_method>`

## 璺戝疄楠屽墠鍏堟鏌ョ幆澧?

鍦?GPU 鏈哄櫒涓婂厛妫€鏌?PyTorch 鍜?CUDA锛?

```bash
python - <<'PY'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

濡傛灉 `cuda` 鏄?`False`锛屽厛涓嶈姝ｅ紡璺戙€傞渶瑕佹崲 PyTorch/CUDA 闀滃儚锛屾垨鑰呴噸鏂板畨瑁?
鍖归厤 CUDA 鐗堟湰鐨?`torch` 鍜?`torchvision`銆?

鐒跺悗瀹夎椤圭洰渚濊禆锛?

```bash
pip install -r requirements.txt
```

## 鍏堣窇娴嬭瘯锛屼笉瑕佺洿鎺ヨ窇婊?100 epoch

姣忔涓婃柊鏈哄櫒锛屽厛璺戣交閲忔祴璇曪細

```bash
python -m unittest discover -s tests -p "test_*.py" -v

cd optical-flow
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/run_smoke.py
cd ..
```

濡傛灉鏈湴鎴栨湇鍔″櫒娌℃湁瀹夎 torch锛岄儴鍒?torch 鐩稿叧娴嬭瘯浼氭樉绀?`skipped`銆傝繖涓嶄唬琛?
traditional representation 鍑洪敊锛屽彧鏄缁冪浉鍏虫祴璇曟病鏈夋墽琛屻€?

## N-MNIST 鍒嗙被 smoke test

N-MNIST 姣旇緝灏忥紝鏈€閫傚悎鍏堟鏌ヨ缁冮棴鐜€傜涓€娆″彧璺?1 涓?epoch 鍜屽皯閲忔牱鏈細

```bash
python train_traditional_classification.py \
  --dataset nmnist \
  --root /mnt/datasets \
  --method event_frame \
  --epochs 1 \
  --train-limit 512 \
  --val-limit 128 \
  --test-limit 128 \
  --batch-size 32 \
  --num-workers 0 \
  --device cuda \
  --output-dir /mnt/outputs/traditional/classification/nmnist/event_frame_smoke
```

杩欎釜鍛戒护鐨勭洰鐨勪笉鏄拷姹傞珮鍑嗙‘鐜囷紝鑰屾槸妫€鏌ワ細

- 鏁版嵁鑳戒笉鑳芥甯镐笅杞芥垨璇诲彇
- representation 鑳戒笉鑳芥甯告瀯寤?
- ResNet18 鑳戒笉鑳?forward/backward
- loss 鍜?accuracy 鏄笉鏄湁闄愭暟鍊?
- checkpoint 鍜屾棩蹇楄兘涓嶈兘姝ｅ父淇濆瓨

璺戝畬鍚庢鏌ヨ緭鍑虹洰褰曪細

```bash
ls /mnt/outputs/traditional/classification/nmnist/event_frame_smoke
```

搴旇鑳界湅鍒帮細

```text
config.json
history.jsonl
metrics.json
progress.json
representation_stats.json
train.log
checkpoints/
```

## N-MNIST 姝ｅ紡 baseline

smoke test 閫氳繃鍚庯紝鍐嶈窇 5 涓柟娉曪細

```bash
for method in event_frame binary_event_image timestamp_image time_surface voxel_grid; do
  python train_traditional_classification.py \
    --dataset nmnist \
    --root /mnt/datasets \
    --method $method \
    --epochs 100 \
    --early-stop-patience 10 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda \
    --resume \
    --output-dir /mnt/outputs/traditional/classification/nmnist/$method
done
```

璁粌閲囩敤 `max 100 epochs + early stopping`銆備篃灏辨槸璇达紝涓嶄竴瀹氭瘡涓柟娉曢兘浼氳窇婊?
100 杞紱濡傛灉楠岃瘉闆嗛暱鏈熶笉鎻愬崌锛屼細鎻愬墠鍋滄銆傛渶缁堟姤鍛婂簲浣跨敤 best checkpoint 鐨?
test metric锛岃€屼笉鏄渶鍚庝竴杞殑缁撴灉銆?

## N-Caltech101 鍒嗙被

N-Caltech101 姣?N-MNIST 鏇村ぇ锛屽缓璁厛璺?smoke test锛?

```bash
python train_traditional_classification.py \
  --dataset ncaltech101 \
  --root /mnt/datasets \
  --method event_frame \
  --epochs 1 \
  --train-limit 512 \
  --val-limit 128 \
  --test-limit 128 \
  --batch-size 16 \
  --num-workers 0 \
  --device cuda \
  --output-dir /mnt/outputs/traditional/classification/ncaltech101/event_frame_smoke
```

濡傛灉鏄惧瓨瓒冲锛屽啀姝ｅ紡璺戯細

```bash
for method in event_frame binary_event_image timestamp_image time_surface voxel_grid; do
  python train_traditional_classification.py \
    --dataset ncaltech101 \
    --root /mnt/datasets \
    --method $method \
    --epochs 100 \
    --early-stop-patience 10 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda \
    --resume \
    --output-dir /mnt/outputs/traditional/classification/ncaltech101/$method
done
```

濡傛灉閬囧埌 CUDA OOM锛屾妸 `--batch-size 32` 鏀规垚 `16` 鎴?`8`銆?

## Prophesee mini detection

The maintained object-detection pipeline now uses the Prophesee mini detection
dataset. Build window metadata first, then run a small smoke test or the full
six-method benchmark through the scripts under `scripts/detection/prophesee/`.

```bash
python scripts/detection/prophesee/build_window_index.py \
  --root /path/to/mini_dataset

python scripts/detection/prophesee/train.py \
  --root /path/to/mini_dataset \
  --method event_frame \
  --epochs 1 \
  --train-limit 32 \
  --val-limit 16 \
  --test-limit 16 \
  --output-dir outputs/debug/prophesee_event_frame_smoke
```

## MVSEC optical flow

MVSEC 鐨?traditional adapter 宸茬粡鎺ュ叆 `optical-flow`銆傚厛璺?mock smoke锛?

```bash
cd optical-flow
python scripts/run_smoke.py
python scripts/run_linear_benchmark.py --adapter event_frame --use-mock
```

鐪熷疄 MVSEC formal protocol 鍙互鏄惧紡鎸囧畾 adapter锛?

```bash
python scripts/run_original_protocol.py \
  --adapter event_frame \
  --data-root /path/to/mvsec \
  --epochs 100 \
  --early-stop-patience 10 \
  --device cuda
```

鍏蜂綋鐪熷疄鏁版嵁璺緞鍜屽弬鏁颁互 optical-flow 缁勫綋鍓嶆枃妗ｄ负鍑嗐€備紶缁熸柟娉曚笉瑕侀粯璁ゆ贩杩涘師鏈?
six-method learning-based suite锛涢渶瑕佽窇鏃舵樉寮忔寚瀹?adapter銆?

## 璁粌杩囩▼涓璁板綍浠€涔?

姣忎釜 run 鐨勮緭鍑虹洰褰曢噷锛岄噸鐐圭湅杩欎簺鏂囦欢锛?

| 鏂囦欢 | 鐢ㄩ€?|
|---|---|
| `config.json` | 鏈瀹為獙瀹屾暣閰嶇疆 |
| `history.jsonl` | 姣忎竴杞?train/val 鎸囨爣 |
| `metrics.json` | 鏈€缁堟眹鎬荤粨鏋?|
| `progress.json` | 褰撳墠鎴栨渶缁堣繘搴?|
| `representation_stats.json` | 琛ㄧず寮犻噺缁熻锛屼緥濡傞潪闆舵瘮渚嬨€佹瀯寤烘椂闂?|
| `train.log` | 浜鸿兘鐩存帴璇荤殑璁粌鏃ュ織 |
| `checkpoints/best.pt` | 楠岃瘉闆嗘渶浣虫ā鍨?|
| `checkpoints/last.pt` | 鏈€杩戜竴杞ā鍨嬶紝鏂逛究 resume |

鍐欐姤鍛婃椂寤鸿璁板綍锛?

- method
- dataset
- downstream model
- best epoch
- test accuracy / mAP / AEE
- early stopping 鏄惁瑙﹀彂
- representation shape
- nonzero ratio
- mean build time

杩欐牱缁撴灉涓嶆槸榛戠洅锛屽悗闈㈠彲浠ヨВ閲婁负浠€涔堟煇涓紶缁熻〃绀烘洿蹇€佹洿绋€鐤忥紝鎴栬€呭噯纭巼鏇撮珮銆?

## 鎺ㄨ崘鎵ц椤哄簭

鏈€鐪侀挶銆佹渶绋崇殑椤哄簭鏄細

1. N-MNIST `event_frame` smoke test
2. N-MNIST 浜斾釜鏂规硶姝ｅ紡璺?
3. N-Caltech101 `event_frame` smoke test
4. N-Caltech101 浜斾釜鏂规硶姝ｅ紡璺?
5. MVSEC mock smoke
6. MVSEC formal protocol
7. Prophesee mini detection smoke
8. Prophesee mini detection formal run

涓嶈鍏堣窇 GEN1銆傚畠渚濊禆閲嶃€佹樉瀛樺拰鏃堕棿鎴愭湰閮芥洿楂橈紝閫傚悎鏀惧湪鏈€鍚庛€?

