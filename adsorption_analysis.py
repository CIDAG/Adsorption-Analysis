# future
from __future__ import annotations

# built-in
import os
import sys
import json
import signal
import pathlib

# numpy
import numpy as np
import numpy.typing as npt

# sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# matplotlib
import matplotlib.pyplot as plt

# tqdm
import tqdm

# custom
import libs.adsorption as adsp
from libs.cluster import Cluster


def validate_adsp_json(j: dict) -> bool:
    """
    Returns True if JSON is valid or False otherwise
    """

    # rng seed
    if type(j["seed"]) != int:
        print("[ERROR] seed must be an integer")
        return False

    # system_name
    if j.get("system_name", "ERRNF") == "ERRNF" or not j["system_name"]:
        print("[ERROR] system_name not specified")
        return False
    if type(j["system_name"]) != str:
        print("[ERROR] system_name must be a string")
        return False

    # input_folder_path
    if (
        j.get("input_folder_path", "ERRNF") == "ERRNF"
        or not j["input_folder_path"]
    ):
        print("[ERROR] input_folder_path folder not specified")
        return False
    if type(j["input_folder_path"]) != str:
        print("[ERROR] input_folder_path must be a string")
        return False

    # output_folder_path
    if (
        j.get("output_folder_path", "ERRNF") == "ERRNF"
        or not j["output_folder_path"]
    ):
        print("[ERROR] output_folder_path not specified")
        return False
    if type(j["output_folder_path"]) != str:
        print("[ERROR] output_folder_path must be a string")
        return False

    # method to select number k of clusters
    if (
        j.get("method_of_k_selection", "ERRNF") == "ERRNF"
        or not j["method_of_k_selection"]
    ):
        print("[ERROR] method_of_k_selection not specified")
        return False
    if type(j["method_of_k_selection"]) != str:
        print("[ERROR] method_of_k_selection must be a string")
        return False

    # number k of clusters (if method_of_k_selection == "user")
    if type(j["number_k_of_clusters"]) != int:
        print("[ERROR] number_k_of_clusters must be an integer")
        return False
    if j["number_k_of_clusters"] < 2:
        print("[ERROR] number_k_of_clusters must be greater than 1")
        return False

    # silhouette_range (if method_of_k_selection == "silhouette")
    if j.get("silhouette_range", "ERRNF") == "ERRNF":
        print("[ERROR] silhouette_range not specified")
        return False
    if type(j["silhouette_range"]) != list or len(j["silhouette_range"]) != 3:
        print(
            "[ERROR] silhouette_range must be a list with 3 integers"
        )
        return False
    for i in j["silhouette_range"]:
        if type(i) != int:
            print("[ERROR] silhouette_range elements must be integers")
            return False

    # number of random runs of K-Means for a given K
    if type(j["number_of_random_runs"]) != int:
        print("[ERROR] number_of_random_runs  must be an integer")
        return False
    if j["number_of_random_runs"] < 1:
        print("[ERROR] number_of_random_runs  must be greater than 0")
        return False

    # molecule_indices
    if j.get("molecule_indices", "ERRNF") == "ERRNF":
        print("[ERROR] molecule_indices not specified")
        return False
    if type(j["molecule_indices"]) != list or not j["molecule_indices"]:
        print(
            "[ERROR] molecule_indices must be a list with at least 1 integer"
        )
        return False
    for i in j["molecule_indices"]:
        if type(i) != int:
            print("[ERROR] molecule_indices elements must be integers")
            return False

    # site_metric
    if j.get("site_metric", "ERRNF") == "ERRNF" or not j["site_metric"]:
        print("[ERROR] site_metric not specified")
        return False
    if type(j["site_metric"]) != str:
        print("[ERROR] site_metric must be a string")
        return False

    # metric_val
    if j["site_metric"] == "site_size":
        if j.get("site_size", "ERR_NOT_FOUND") == "ERR_NOT_FOUND":
            print("[ERROR] site_size not specified")
            return False
        if type(j["site_size"]) != int:
            print("[ERROR] site_size must be an integer")
            return False
    elif j["site_metric"] == "site_radius":
        if j.get("site_radius", "ERR_NOT_FOUND") == "ERR_NOT_FOUND":
            print("[ERROR] site_radius not specified")
            return False
        if type(j["site_radius"]) != float and type(j["site_radius"]) != int:
            print("[ERROR] site_radius must be an integer or a float")
            return False
    else:
        print("[ERROR] invalid site_metric. Either site_size or site_radius.")
        return False

    # fixed_substrate_atomic_number
    if (
        j.get("fixed_substrate_atomic_number", "ERR_NOT_FOUND")
        == "ERR_NOT_FOUND"
    ):
        print("[ERROR] fixed_substrate_atomic_number not specified")
        return False
    if type(j["fixed_substrate_atomic_number"]) != int:
        print("[ERROR] fixed_substrate_atomic_number must be an integer")
        return False
    if (
        j["fixed_substrate_atomic_number"] < 0
        or j["fixed_substrate_atomic_number"] > 110
    ):
        print(
            "[ERROR] fixed_substrate_atomic_number must be between 1 and 110 "
            "for fixed Z or 0 for variable Z"
        )
        return False

    # z_exp
    if j.get("z_exp", "ERR_NOT_FOUND") == "ERR_NOT_FOUND":
        print("[ERROR] z_exp not specified")
        return False
    if type(j["z_exp"]) != float and type(j["z_exp"]) != int:
        print("[ERROR] z_exp must be an integer or a float")
        return False

    # d_exp
    if j.get("d_exp", "ERR_NOT_FOUND") == "ERR_NOT_FOUND":
        print("[ERROR] d_exp not specified")
        return False
    if type(j["d_exp"]) != float and type(j["d_exp"]) != int:
        print("[ERROR] d_exp must be an integer or a float")
        return False

    # use_energy
    if j.get("use_energy", "ERRNF") == "ERRNF":
        print("[ERROR] use_energy not specified")
        return False
    if type(j["use_energy"]) != bool:
        print("[ERROR] use_energy must be a bool")
        return False

    # scale_dataset
    if j.get("scale_dataset", "ERRNF") == "ERRNF":
        print("[ERROR] scale_dataset not specified")
        return False
    if type(j["scale_dataset"]) != bool:
        print("[ERROR] scale_dataset must be a bool")
        return False

    # projection_numbers
    if j.get("projection_numbers", "ERRNF") == "ERRNF":
        print("[ERROR] projection_numbers not specified")
        return False
    if type(j["projection_numbers"]) != bool:
        print("[ERROR] projection_numbers must be a bool")
        return False

    # proj representatives
    if j.get("projection_repr", "ERRNF") == "ERRNF":
        print("[ERROR] projection_repr not specified")
        return False
    if type(j["projection_repr"]) != bool:
        print("[ERROR] projection_repr must be a bool")
        return False

    # projection_tsne
    if j.get("projection_tsne", "ERRNF") == "ERRNF":
        print("[ERROR] projection_tsne not specified")
        return False
    if type(j["projection_tsne"]) != bool:
        print("[ERROR] projection_tsne must be a bool")
        return False

    return True


def signal_handler(sig: int, frame: signal.frame):
    """
    Exits the program gracefully on a SIGINT
    """
    print("\nOperation canceled.")
    exit()


def collect_input() -> (
    tuple[
        int,
        str,
        str,
        str,
        str,
        int,
        npt.NDArray[int],
        int,
        npt.NDArray[int],
        str,
        int | float,
        int,
        int | float,
        int | float,
        bool,
        bool,
        bool,
        bool,
    ]
):
    """
    Collect, parse and validate input data
    """
    # message to be displayed in case of an error
    arg_error = f"Usage: python3 {sys.argv[0]} input_file"

    # check number of args
    if len(sys.argv) != 2:
        print(arg_error)
        exit()

    # get input file
    try:
        with open(sys.argv[1]) as f:
            j = json.load(f)
    except FileNotFoundError:
        print("[ERROR] input json file not found")
        exit()
    except json.decoder.JSONDecodeError:
        print("[ERROR] input json format is invalid")
        exit()
    except Exception:
        print("[ERROR] an unexpected error happened trying to read the input")
        exit()

    # validate json input data
    if not validate_adsp_json(j):
        exit()

    return (
        j["seed"],
        j["system_name"],
        j["input_folder_path"],
        j["output_folder_path"],
        j["method_of_k_selection"],
        j["number_k_of_clusters"],
        j["silhouette_range"],
        j["number_of_random_runs"],
        np.array(j["molecule_indices"]),
        j["site_metric"],
        j["site_size"]
        if j["site_metric"] == "site_size"
        else j["site_radius"],
        j["fixed_substrate_atomic_number"],
        j["z_exp"],
        j["d_exp"],
        j["use_energy"],
        j["scale_dataset"],
        j["projection_numbers"],
        j["projection_repr"],
        j["projection_tsne"],
    )


def main() -> None:
    """
    Routine called if the CLI is ran
    """
    # matplotlib configuration
    plt.rcParams["figure.figsize"] = (10, 10)

    # register the signal to exit the program gracefully on a SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    # collect information from the input JSON file
    args = collect_input()

    # send everything to the pipeline
    adsp.pipeline(*args)


if __name__ == "__main__":
    main()
