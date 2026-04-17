# IDX .JK brute-force finder

Ini workaround kalau roster resmi IDX susah diambil otomatis.

## Apa yang script ini lakukan
- generate semua kombinasi simbol 4 huruf `AAAA.JK` s/d `ZZZZ.JK`
- kirim ke endpoint quote Yahoo secara batch
- simpan yang dibalas valid ke CSV

## Kapan dipakai
- kalau lo mau net tambahan dari Yahoo sendiri
- **bukan** pengganti roster resmi IDX

## Kenapa 4 huruf
Mayoritas kode emiten biasa IDX pakai 4 huruf.

## Jalankan
```bash
pip install requests
python discover_jk_universe_bruteforce.py --output data/idx_universe_bruteforce.csv --resume
```

## Test cepat dulu
```bash
python discover_jk_universe_bruteforce.py --output data/idx_universe_bruteforce.csv --resume --max-batches 50
```

## Full run
```bash
python discover_jk_universe_bruteforce.py --output data/idx_universe_bruteforce.csv --resume --batch-size 100 --sleep 0.25
```

## Output
- `data/idx_universe_bruteforce.csv`
- `data/idx_universe_bruteforce.audit.json`

## Catatan penting
- ini bisa lama
- ini bisa miss ticker yang Yahoo nggak balas dengan clean
- ini bisa include simbol yang valid di Yahoo tapi bukan target universe final lo
- tetap paling bagus merge hasil ini dengan roster resmi / file master lokal
