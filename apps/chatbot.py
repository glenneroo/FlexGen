"""Run a chatbot with FlexGen and OPT models."""
import argparse
import signal
import sys
import time

from transformers import AutoTokenizer
from flexgen.flex_opt import (Policy, OptLM, TorchDevice, TorchDisk, TorchMixedDevice,
                              CompressionConfig, Env, Task, get_opt_config, str2bool)

show_times = True  # show debug timing


def main(args):
    start_time = time.perf_counter()

    # Initialize environment
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = Env(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    # Offloading policy
    policy = Policy(1, 1,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    overlap=True, sep_layer=True, pin_weight=args.pin_weight,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=args.compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=args.compress_cache,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    # Model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    tokenizer.add_bos_token = False
    stop = tokenizer("\n").input_ids[0]

    opt_config = get_opt_config(args.model)
    model = OptLM(opt_config, env, args.path, policy)
    if show_times:
        print(f"OptLM model loaded in [{time.perf_counter() - start_time:0.2f}] seconds.")

    model.init_all_weights()
    if show_times:
        print(f"Weights loaded in [{time.perf_counter() - start_time:0.2f}] seconds.")
    print("---------------------------------- start of context ----------------------------------")

    context_org = (
        "A chat between a curious human and a knowledgeable artificial intelligence assistant.\n"
        "Human: Hello! What can you do?\n"
        "Assistant: As an AI assistant, I can answer questions and chat with you.\n"
        "Human: What is the name of the tallest mountain in the world?\n"
        "Assistant: Everest.\n"
    )

    print(context_org, end="")
    print("----------------------------------- end of context -----------------------------------")
    max_new_tokens = 96
    temp = 0.7
    context = context_org

    # start the chat session - press <ENTER> with empty line to exit
    while True:
        inp = input("Human: ")
        if not inp:
            break

        if inp.lower() == "reset":
            print("Resetting context.")
            context = context_org
            continue
        if inp.lower().startswith("tokens="):
            print("New max_new_tokens = " + inp[7:])
            max_new_tokens = int(inp[7:])
            continue
        if inp.lower().startswith("temp="):
            print("New temperature = " + inp[5:])
            temp = float(inp[5:])
            continue

        start_time = time.perf_counter()
        context += "Human: " + inp + "\n"
        inputs = tokenizer([context])

        output_ids = model.generate(
            inputs.input_ids,
            do_sample=True,
            temperature=temp,
            max_new_tokens=max_new_tokens,
            stop=stop)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        try:
            index = outputs.index("\n", len(context))
        except ValueError:
            outputs += "\n"
            index = outputs.index("\n", len(context))

        outputs = outputs[:index + 1]
        response = str(outputs[len(context):]).replace("\n", "")
        if show_times:
            response += f" ({time.perf_counter() - start_time:0.2f}s)"

        print(response + "\n", end="")
        context = outputs

    # TODO: optimize the performance by reducing redundant computation.

    print("Shutting down...")
    model.delete_all_weights()
    disk.close_copy_threads()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
                        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
                        help="The path to the model weights. If there are no cached weights, "
                             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
                        help="The directory to offload tensors. ")
    parser.add_argument("--percent", nargs="+", type=int,
                        default=[100, 0, 100, 0, 100, 0],
                        help="Six numbers. They are "
                             "the percentage of weight on GPU, "
                             "the percentage of weight on CPU, "
                             "the percentage of attention cache on GPU, "
                             "the percentage of attention cache on CPU, "
                             "the percentage of activations on GPU, "
                             "the percentage of activations on CPU")
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
                        const=True, default=True)
    parser.add_argument("--compress-weight", action="store_true",
                        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
                        help="Whether to compress cache.")
    args = parser.parse_args()

    assert len(args.percent) == 6

    main(args)
