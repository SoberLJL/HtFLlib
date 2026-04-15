conda activate fedtgp

$outFile = "test-Cifar100-HtFE-img-8-fd=512-FedMGP.out"

python -u main.py `
  -t 1 `
  -ab 0 `
  -gr 10 `
  -lr 0.01 `
  -jr 1 `
  -lbs 10 `
  -ls 1 `
  -nc 20 `
  -ncl 100 `
  -data Cifar100 `
  -m HtFE-img-8 `
  -fd 512 `
  -did 0 `
  -algo FedMGP `
  -lam 10 `
  -se 10 `
  -mart 100 `
  -nsp 2 `
  -tm 1.0 `
  -cw 1.0 `
  -wr 3 `
  -go test_mgp_v2 2>&1 | Tee-Object -FilePath $outFile

Write-Host "`nLog saved to: $outFile"
