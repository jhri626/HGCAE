gpu=0
seed=1234
save=1

export HGCAE_HOME=$(pwd)
export DATAPATH="$HGCAE_HOME/data"
export LOG_DIR="$HGCAE_HOME/logs"
export HGCAE_CACHE_DIR="$HGCAE_HOME/cache"
mkdir -p "$LOG_DIR"
mkdir -p "$HGCAE_CACHE_DIR"

n_head=1
lr=0.01
normalize_feats=0
att_logit=exp

model=HGCAE
att_type=sparse_adjmask_dist
dataset=cora
act=tanh
c=1
c_trainable=0
dropout=0.7
weight_decay=0.001
hidden_dim=256
dim=16
lambda_rec=10
lambda_iso=0.1
manifold=PoincareBall
iso_measure=least_squares_distortion
optimizer=Adam

python train_solver.py --model $model \
    --seed $seed --dataset $dataset --lr $lr --normalize-feats $normalize_feats \
    --min-epochs 100 --save $save --log-freq 10 --cuda $gpu \
    --hidden-dim $hidden_dim --dim $dim --num-layers 2 --act $act --bias 1 \
    --dropout $dropout --weight-decay $weight_decay \
    --alpha 0.2 --n-heads $n_head \
    --manifold $manifold --c $c --c-trainable $c_trainable \
    --lambda-rec $lambda_rec --lambda-iso $lambda_iso \
    --iso-measure $iso_measure \
    --optimizer $optimizer \
    --use-att 1 --att-type $att_type --att-logit $att_logit
