ln -s ../1-dataset/train_Pt.xyz train.xyz
ln -s ../1-dataset/test_Pt.xyz test.xyz
mace_run_train --name="Pt_tinymodel" --train_file="train.xyz" --test_file="test.xyz" --valid_fraction=0.1 --config_type_weights='{"Default":1.0}' --E0s='{78:-1}' --model="MACE" --batch_size=20 --hidden_irreps='8x0e + 8x1o' --r_max=5.0 --max_num_epochs=100 --device=cuda --swa --start_swa=80 --ema --ema_decay=0.99 --amsgrad
