conda activate fedtgp

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
  -algo FedTGP `
  -lam 10 `
  -se 10 `
  -mart 100 `
  -go test_10rounds
