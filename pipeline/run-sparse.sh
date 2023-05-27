export TF_CPP_MIN_LOG_LEVEL=2
export dataset='compressed_animals'

for i in $(seq 0 9);
do
    echo Iteration: $i

    ## collect reference unfiltered dataset for evaluation statistics
    python pipeline/sample.py --dataset ${dataset} --iteration ${i} \
            --max_samples 10000 --mask_param 0 \
            --evaluate True --identical_batch False \
            --savepath samples/${i}_reference

    JAX_PLATFORMS=cpu python pipeline/save_sizes.py \
        --dataset ${dataset} --iteration ${i} \
        --loadpath samples/${i}_reference

    ## collect finetuning dataset
    python pipeline/sample.py --dataset $dataset --iteration $i

    JAX_PLATFORMS=cpu python pipeline/save_sizes.py \
        --dataset ${dataset} --iteration ${i} \
        --loadpath samples/${i}

    ## finetune
    python pipeline/finetune.py --dataset ${dataset} --iteration ${i}

    ## rm logs folder with local model checkpoints
    sleep 1m
    rm -rf logs
done

## save samples from last model
i=$(($i+1))

python pipeline/sample.py --dataset ${dataset} --iteration ${i} \
        --max_samples 10000 --mask_param 0 \
        --evaluate True --identical_batch False \
        --savepath samples/${i}_reference

JAX_PLATFORMS=cpu python pipeline/save_sizes.py \
    --dataset ${dataset} --iteration ${i} \
    --loadpath samples/${i}_reference
