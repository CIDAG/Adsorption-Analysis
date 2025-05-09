# fixed_substrate_atomic_number future
from __future__ import annotations

# built-in
import glob
import pathlib
import typing
import shutil

# numpy
import numpy as np
import numpy.typing as npt

# sklearn
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# matplotlib
import matplotlib.pyplot as plt

# tqdm
import tqdm

# custom
import libs.xyz_tools as tools
from libs.cluster import Cluster

# matplotlib configuration
plt_figsize = (10, 10)


def treat_fname(fname):
    """
    Treats filename so that it does not contain certain characters
    """
    return fname.replace("/", "_").replace("$", "_").replace("\\", "_")


def get_data_from_path(
    input_folder_path: str,
    use_energy: bool,
) -> tuple[list[str], list[tools.XYZData]]:
    """
    Load XYZ files. If the program cannot load them, exit with an error
    """
    xyz_file_names = sorted(glob.glob(input_folder_path + "/*.xyz"))
    xyz_data = [tools.read_xyz(x, use_energy) for x in xyz_file_names]

    if not xyz_data:
        print("[ERROR] Could not load XYZ files from the specified folder.")
        exit()

    return xyz_file_names, xyz_data


def extract_molecule(
    data: tools.XYZData, molecule_indices: npt.NDArray[int]
) -> tuple[tools.XYZData, tools.XYZData]:
    """
    Separate molecule and cluster atoms
    """
    molecule = data[molecule_indices]
    cluster = tools.XYZData(
        np.delete(data.atoms, molecule_indices),
        np.delete(data.coords, molecule_indices, axis=0),
        data.energy,
    )

    return molecule, cluster


def validate_molecules(
    files: list[str], data: tools.XYZData, molecule_indices: npt.NDArray[int]
) -> None:
    """
    Check if the molecules of all the files are the same
    """

    # set correct blank molecule's name and atoms as the first one
    corr_file = files[0]
    corr_mol, _ = extract_molecule(data[0], molecule_indices)
    corr_mol = np.sort(corr_mol.atoms)

    # check if all the molecules are equal
    for name, entry in zip(files, data):
        molecule, _ = extract_molecule(entry, molecule_indices)
        molecule = np.sort(molecule.atoms)
        if not np.array_equal(molecule, corr_mol):
            print(f"[ERROR] Molecule from '{name}' and '{corr_file}' differ!")
            exit()


def find_closest_atoms_to_mol(
    mol: tools.XYZData, clus: tools.XYZData, site_size: int
) -> tools.XYZData:
    """
    This function finds the N closest atoms in the cluster to the molecule and
    returns their XYZData. These atoms compose the hypothesized adsorption site
    """
    # create list of tuples (mol_atom_index, clus_atom_index, distance)
    distances = []
    for i, m in enumerate(mol):
        for j, c in enumerate(clus):
            distances.append((i, j, np.linalg.norm(m.coords - c.coords)))

    # sort by shortest distances
    distances = np.array(sorted(distances, key=lambda x: x[2]))

    # get N unique cluster atoms that have not already been processed so that
    # only the best (smallest) values of distance for each are kept
    best_atoms = np.array([[-1, -1, -1]])
    for d in distances:
        if d[1] not in best_atoms[:, 1]:
            best_atoms = np.vstack([best_atoms, d])
        if len(best_atoms) == site_size + 1:
            break  # stop early if N atoms have already been found

    return clus[best_atoms[1:, 1].astype(int)]


def equalize_z(data: tools.XYZData, z: int) -> None:
    """
    Transform all atoms inside given structure into a single given type. The
    input should be the desired atomic number (Z) or 0 to bypass this function.
    NOTE: the function is in-place
    """
    # nothing to be done
    if z == 0:
        return data

    # change atoms and return new data
    data.atoms = np.repeat([tools.get_symbol(z)], len(data.atoms))


def build_dataset_site_size(
    xyz_file_names: list[str],
    xyz_data: list[tools.XYZData],
    molecule_indices: npt.NDArray[int],
    site_size: int,
    fixed_substrate_atomic_number: int,
    z_exp: int | float,
    d_exp: int | float,
    scale: bool,
) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float], list,]:
    """
    Build Coulomb Matrices eigenvalues dataset and substrate info. This dataset
    takes into account the molecules to be adsorbed and the adsorption sites
    """
    # generate base dataset
    dataset = np.empty((1, len(molecule_indices) + site_size))

    # generate array to store the energies
    energies = np.empty(0)

    # generate array to store average of the atomic numbers in substrate
    subs = np.empty(0)

    # site atoms
    site_atoms = []

    # populate the dataset with the eigenvalues
    for entry in tqdm.tqdm(xyz_data, total=len(xyz_data)):
        mol, clus = extract_molecule(entry, molecule_indices)
        closest = find_closest_atoms_to_mol(mol, clus, site_size)

        # populate site atoms arrays with atoms
        site_atoms.append(closest.atoms)

        # populate substrates array with average of atomic nums in substrate
        subs = np.append(
            subs, np.array([tools.atomic_num[x] for x in closest.atoms]).mean()
        )

        # populate energy array with energy
        energies = np.append(energies, entry.energy)

        equalize_z(closest, fixed_substrate_atomic_number)
        mol_join_closest = mol + closest
        eigen = tools.eigen_coulomb(mol_join_closest, z_exp, d_exp)
        dataset = np.vstack([dataset, eigen])

    # remove first useless line
    dataset = dataset[1:]

    # scale the data, if so desired
    if scale:
        dataset = StandardScaler().fit_transform(dataset)

    return dataset, subs, energies, site_atoms


def find_atoms_in_site_radius(
    mol: tools.XYZData, clus: tools.XYZData, site_radius: float
) -> int:
    """
    Find the count of atoms inside the site radius
    """
    # find the closest cluster atom to the molecule
    closest_atom = find_closest_atoms_to_mol(mol, clus, 1)

    # calc dists between the closest clus atom to the mol and other clus atoms
    distances = [np.linalg.norm(closest_atom.coords - c.coords) for c in clus]

    # return the count of atoms within the specified radius
    return np.sum(np.array(distances) < site_radius)


def calculate_average_atoms(
    xyz_file_names: list[str],
    xyz_data: list[tools.XYZData],
    molecule_indices: npt.NDArray[int],
    site_radius: float,
) -> int:
    """
    Get the average number of atoms in site radius
    """
    avg_atoms_in_site_radius = 0

    for entry in xyz_data:
        mol, clus = extract_molecule(entry, molecule_indices)
        avg_atoms_in_site_radius += find_atoms_in_site_radius(
            mol, clus, site_radius
        )

    # calculate average and round to the next integer
    return np.ceil(avg_atoms_in_site_radius / len(xyz_file_names)).astype(int)


def build_dataset_site_radius(
    xyz_file_names: list[str],
    xyz_data: list[tools.XYZData],
    molecule_indices: npt.NDArray[int],
    site_radius: float,
    fixed_substrate_atomic_number: int,
    z_exp: int | float,
    d_exp: int | float,
    scale: bool,
) -> tuple[
    npt.NDArray[float],
    npt.NDArray[float],
    npt.NDArray[float],
    list,
    int,
]:
    """
    Build the dataset considering elements within a radius to belong to the
    adsorption site
    """

    avg_atoms_in_site_radius = calculate_average_atoms(
        xyz_file_names, xyz_data, molecule_indices, site_radius
    )

    # generate base dataset
    dataset = np.empty((1, len(molecule_indices) + avg_atoms_in_site_radius))

    # generate array to store the energies
    energies = np.empty(0)

    # generate array to store average of the atomic numbers in substrate
    subs = np.empty(0)

    # site atoms
    site_atoms = []

    # populate the dataset with the eigenvalues
    for entry in tqdm.tqdm(xyz_data, total=len(xyz_data)):
        mol, clus = extract_molecule(entry, molecule_indices)

        closest_atom = find_closest_atoms_to_mol(mol, clus, 1)
        closest = find_closest_atoms_to_mol(
            closest_atom, clus, avg_atoms_in_site_radius
        )

        # populate site atoms arrays with atoms
        site_atoms.append(closest.atoms)

        # populate substrates array with average of atomic nums in substrate
        subs = np.append(
            subs, np.array([tools.atomic_num[x] for x in closest.atoms]).mean()
        )

        # populate energy array with energy
        energies = np.append(energies, entry.energy)

        equalize_z(closest, fixed_substrate_atomic_number)
        mol_join_closest = mol + closest
        eigen = tools.eigen_coulomb(mol_join_closest, z_exp, d_exp)
        dataset = np.vstack([dataset, eigen])

    # remove first useless line
    dataset = dataset[1:]

    # scale the data, if so desired
    if scale:
        dataset = StandardScaler().fit_transform(dataset)

    return dataset, subs, energies, site_atoms, avg_atoms_in_site_radius


def calculate_silhouette(
    data: npt.NDArray[float], labels: npt.NDArray[int], k: int
) -> float:
    """
    Compute Silhouette for each sample, take the average for each cluster and
    return (#clusters that surpassed the general Silhouette) / (#clusters)
    """
    # Calculate silhouette for individual samples
    silh_samples = silhouette_samples(data, labels)

    # Calculate general silhouette
    silh_score = silhouette_score(data, labels)

    # Create cluster-silhouette array and order it by label number
    csilh = np.array(list(zip(labels, silh_samples)))
    csilh = csilh[csilh[:, 0].argsort()]

    # Group silhouette scores belonging in the same cluster together
    groups = np.split(
        csilh[:, 1], np.unique(csilh[:, 0], return_index=True)[1][1:]
    )

    # Return (#clusters that surpassed the general Silhouette) / (#clusters)
    return sum([group.mean() >= silh_score for group in groups]) / k


def perform_clustering_single_k(
    seed: int, data: npt.NDArray[float], k: int
) -> Cluster:
    """
    Perform K-Means clustering for a single value of K
    """

    c = Cluster(1)

    kmeans = KMeans(n_clusters=k, random_state=seed).fit(data)
    score = calculate_silhouette(data, kmeans.labels_, k)
    wcss = kmeans.inertia_

    c.best_k = k
    c.insert(k, kmeans.labels_, kmeans.cluster_centers_, score, wcss)

    return c


def perform_clustering(
    seed: int, data: npt.NDArray[float], srange: npt.NDArray[int]
) -> Cluster:
    """
    Perform K-Means clustering for all possible values of K
    """
    # this stores clustering data for all values of K
    c = Cluster(len(data))

    best_score = float("-inf")

    # perform K-Means for all possible values of K
    for k in tqdm.tqdm(range(*srange)):
        kmeans = KMeans(n_clusters=k, random_state=seed).fit(data)
        score = calculate_silhouette(data, kmeans.labels_, k)
        wcss = kmeans.inertia_

        if score > best_score:
            best_score = score
            c.best_k = k

        c.insert(k, kmeans.labels_, kmeans.cluster_centers_, score, wcss)

    return c


def perform_clustering_many_runs_single_k(
    seed_list: int, data: npt.NDArray[float], k: int
) -> list[Cluster]:
    """
    Perform K-Means clustering number_of_random_runs times for a single value K
    """

    # this stores clustering data for result values of all available seeds
    clusterings = []

    # perform K-Means for all available seeds
    for i, seed in enumerate(seed_list):
        kmeans = KMeans(n_clusters=k, random_state=seed).fit(data)
        score = calculate_silhouette(data, kmeans.labels_, k)
        wcss = kmeans.inertia_

        c = Cluster(1)
        c.best_k = k
        c.insert(k, kmeans.labels_, kmeans.cluster_centers_, score, wcss)
        clusterings.append(c)

    return [clusterings]


def perform_clustering_many_runs(
    seed_list: int, data: npt.NDArray[float], srange: npt.NDArray[int]
) -> list[list[Cluster]]:
    """
    Perform K-Means clustering number_of_random_runs times for all values of K
    """

    # this stores clustering data of all seeds for all ks
    all_clusterings = []

    # perform K-Means for all possible values of K
    for k in tqdm.tqdm(range(*srange)):
        # this stores clustering data for result values of all available seeds
        clustering = []

        # perform K-Means for all available seeds
        for i, seed in enumerate(seed_list):
            kmeans = KMeans(n_clusters=k, random_state=seed).fit(data)
            score = calculate_silhouette(data, kmeans.labels_, k)
            wcss = kmeans.inertia_

            c = Cluster(1)
            c.best_k = k
            c.insert(k, kmeans.labels_, kmeans.cluster_centers_, score, wcss)
            clustering.append(c)

        all_clusterings.append(clustering)

    return all_clusterings


def get_min_energy_list(
    labels: npt.NDArray[int], energies: npt.NDArray
) -> list[tuple[int, float]]:
    """
    Find minimum energy elements from each cluster and put them in a list
    """

    energy_list = [[] for i in range(len(set(labels)))]

    for i in range(len(energy_list)):
        for label, energy in zip(labels, zip(range(len(energies)), energies)):
            if label == i:
                energy_list[i].append(energy)

    min_energy_list = []

    for energy_tuple in energy_list:
        min_energy_list.append(min(energy_tuple, key=lambda x: x[1]))

    return min_energy_list


def get_representative_list(
    labels: npt.NDArray[int], centroids: npt.NDArray[int], data: npt.NDArray
) -> list[int]:
    """
    Find closest elements to the centroids and return them as representatives
    """
    # get tuples of (index, label, data)
    labeled_data = list(zip(range(len(data)), labels, data))

    # empty representative elements list
    representatives = []

    # find representative elements
    for i, c in enumerate(centroids):
        # valid elements are the ones in the same cluster
        valid = list(filter(lambda x: x[1] == i, labeled_data))
        # find the representative index
        r = np.argmin(list(map(lambda x: np.linalg.norm(x[2] - c), valid)))
        # append the representative indices to the array
        representatives.append(valid[r][0])

    return representatives


def output(
    k: int,
    display_k,
    output_folder_path: str,
    xyz_file_names: list[str],
    data: npt.NDArray[float],
    c: Cluster,
    energies: npt.NDArray[float],
    n_atoms: int,
    fixed_substrate_atomic_number: int,
    z_exp: int | float,
    d_exp: int | float,
    use_energy: bool,
    scale_dataset: bool,
    projection_numbers: bool,
    projection_representatives: bool,
    projection_tsne: bool,
    site_atoms: list[npt.NDArray[str]],
    subs: np.NDArray[float],
    seed: int,
    system_name: str,
) -> None:
    """
    Output PCA, silhouette and textual information
    """

    # make sure plt is in the default style and size
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = plt_figsize

    # create directory if it does not exist
    path = pathlib.Path(output_folder_path + "/output")
    path.mkdir(parents=True, exist_ok=True)
    path = str(path.resolve())

    # if energy is enabled, get the min_energy_list, else representatives
    if use_energy:
        min_energy_list = get_min_energy_list(c.labels[k], energies)
    else:
        representatives = get_representative_list(
            c.labels[k], c.centroids[k], data
        )

    # PCA =====================================================================
    plt.clf()
    pca = PCA(n_components=2).fit(data)
    pca_samples = pca.transform(data)
    pca_var = pca.explained_variance_ratio_
    plt.scatter(
        pca_samples[:, 0],
        pca_samples[:, 1],
        c=c.labels[k],
        cmap="gist_rainbow",
    )

    # Plot energies
    if use_energy:
        indices = [x[0] for x in min_energy_list]
        plt.scatter(
            pca_samples[indices, 0],
            pca_samples[indices, 1],
            marker="X",
            c="tab:gray",
            s=128,
        )
    # Plot representatives
    elif projection_representatives:
        plt.scatter(
            pca_samples[representatives, 0],
            pca_samples[representatives, 1],
            marker="X",
            c="tab:gray",
            s=128,
        )

    # Annotate IDs
    if projection_numbers:
        for i in range(len(pca_samples)):
            plt.annotate(i, (pca_samples[i, 0], pca_samples[i, 1]))
    plt.title(f"PCA of {system_name} (K={display_k})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    plt.savefig(
        path + f"/{treat_fname(system_name)}_PCA_{display_k}.pdf", dpi=300
    )

    # Clustering quality measures (only when there is more than one K)
    if len(c) > 1:
        x_axis = [i for i, x in enumerate(c.scores) if x is not None]
        y_axis = [y for y in c.scores if y is not None]
        plt.clf()
        plt.plot(x_axis, y_axis, linewidth=2)
        plt.scatter(x_axis, y_axis, c="r")
        plt.title(f"Silhouette Scores for {system_name}")
        plt.xlabel("K")
        plt.ylabel("Score")
        plt.grid(visible=True)
        plt.tight_layout()
        plt.savefig(
            path + f"/{treat_fname(system_name)}_silhouette_scores.pdf",
            dpi=300,
        )

        y_axis = [y for y in c.wcss if y is not None]
        plt.clf()
        plt.plot(x_axis, y_axis, linewidth=2)
        plt.scatter(x_axis, y_axis, c="r")
        plt.title(f"WCSS for {system_name}")
        plt.xlabel("K")
        plt.ylabel("WCSS")
        plt.grid(visible=True)
        plt.tight_layout()
        plt.savefig(path + f"/{treat_fname(system_name)}_wcss.pdf", dpi=300)

    # TSNE ====================================================================
    if projection_tsne:
        plt.clf()
        tsne_samples = TSNE(
            n_components=2,
            init="random",
            learning_rate="auto",
            random_state=seed,
            perplexity=min(30, len(data) / 4),
        ).fit_transform(data)
        plt.scatter(
            tsne_samples[:, 0],
            tsne_samples[:, 1],
            c=c.labels[k],
            cmap="gist_rainbow",
        )

        # Annotate IDs
        if projection_numbers:
            for i in range(len(pca_samples)):
                plt.annotate(i, (tsne_samples[i, 0], tsne_samples[i, 1]))
        plt.title(f"t-SNE of {system_name} (K={k})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.xticks(())
        plt.yticks(())
        plt.tight_layout()
        plt.savefig(
            path + f"/{treat_fname(system_name)}_TSNE_{display_k}.pdf", dpi=300
        )

    # Textual data ============================================================
    with open(path + f"/{treat_fname(system_name)}_output.txt", "w") as f:
        # Seed
        f.write(f"Seed: {seed}\n\n")

        # K
        f.write(f"Number K of clusters: {display_k}\n\n")

        # Silhouette score
        f.write(f"Silhouette Score: {c.scores[k]}\n\n")

        # WCSS
        f.write(f"Within-Cluster Sum of Squares (WCSS): {c.wcss[k]}\n\n")

        # Number of atoms
        f.write(f"Number of atoms in site: {n_atoms}\n\n")

        # Fixed substrate atomic number
        if fixed_substrate_atomic_number:
            f.write(
                "Fixed substrate atomic number: "
                f"{fixed_substrate_atomic_number}\n\n"
            )
        else:
            f.write("Fixed substrate atomic number: Disabled\n\n")

        # Coulomb Matrices exponents
        f.write(f"Coulomb Matrix exponents: z={z_exp:.6f} d={d_exp:.6f}\n\n")

        # Dataset scale information
        f.write(f"Scaled dataset: {'Yes' if scale_dataset else 'No'}\n\n")

        # site atoms data
        site_atoms_list = [" ".join(np.ndarray.tolist(x)) for x in site_atoms]

        # Labels
        f.write("Labels and Substrate Z Averages:")
        for i, file, label, avg, s_a in zip(
            range(len(c.labels[k])),
            xyz_file_names,
            c.labels[k],
            subs,
            site_atoms_list,
        ):
            f.write(
                f"\nID: {i} - File: {file} - Label: {label} - Z Avg: {avg:.6f}"
                f" - Site atoms: {s_a}"
            )

        # Energies or Centroids
        if use_energy:
            # Energies
            f.write("\n\nLowest energy elements:")
            for cluster, (index, energy) in enumerate(min_energy_list):
                f.write(f"\nCluster {cluster}: {xyz_file_names[index]}")
        else:
            # Representatives
            f.write("\n\nRepresentatives:")
            for cluster, r in enumerate(representatives):
                f.write(f"\nCluster {cluster}: {xyz_file_names[r]}")

        # Substrate information
        f.write("\n\nSubstrate information:")
        f.write(f"\nSubstrates Average of averages: {subs.mean():.6f}")
        f.write(f"\nSubstrates Std.Dev of averages: {subs.std():.6f}")

        # PCA Explained Variance Ratio
        f.write("\n\nPCA variance:")
        for v in pca_var:
            f.write(f" {v}")

        # End file with blank line
        f.write("\n")


def output_number_of_random_runs(
    k: int,
    display_k: int,
    output_folder_path: str,
    xyz_file_names: list[str],
    data: npt.NDArray[float],
    c: Cluster,
    k_scores: dict,
    nmi_table: npt.NDArray[npt.NDArray[float]],
    energies: npt.NDArray[float],
    n_atoms: int,
    fixed_substrate_atomic_number: int,
    z_exp: int | float,
    d_exp: int | float,
    use_energy: bool,
    scale_dataset: bool,
    projection_numbers: bool,
    projection_representatives: bool,
    projection_tsne: bool,
    site_atoms: list[npt.NDArray[str]],
    subs: np.NDArray[float],
    actual_seed: int,
    seed: int,
    number_of_random_runs: int,
    system_name: str,
) -> None:
    """
    Output PCA, silhouette and textual information
    """

    # make sure plt is in the default style and size
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = plt_figsize

    # create directory if it does not exist
    path = pathlib.Path(output_folder_path + "/output")
    path.mkdir(parents=True, exist_ok=True)
    path = str(path.resolve())

    # if energy is enabled, get the min_energy_list, else representatives
    if use_energy:
        min_energy_list = get_min_energy_list(c.labels[0], energies)
    else:
        representatives = get_representative_list(
            c.labels[0], c.centroids[0], data
        )

    # PCA =====================================================================
    plt.clf()
    pca = PCA(n_components=2).fit(data)
    pca_samples = pca.transform(data)
    pca_var = pca.explained_variance_ratio_
    plt.scatter(
        pca_samples[:, 0],
        pca_samples[:, 1],
        c=c.labels[0],
        cmap="gist_rainbow",
    )

    # Plot energies
    if use_energy:
        indices = [x[0] for x in min_energy_list]
        plt.scatter(
            pca_samples[indices, 0],
            pca_samples[indices, 1],
            marker="X",
            c="tab:gray",
            s=128,
        )
    # Plot representatives
    elif projection_representatives:
        plt.scatter(
            pca_samples[representatives, 0],
            pca_samples[representatives, 1],
            marker="X",
            c="tab:gray",
            s=128,
        )

    # Annotate IDs
    if projection_numbers:
        for i in range(len(pca_samples)):
            plt.annotate(i, (pca_samples[i, 0], pca_samples[i, 1]))
    plt.title(f"PCA of {system_name} (K={display_k})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    plt.savefig(path + f"/{treat_fname(system_name)}_PCA_{display_k}.pdf", dpi=300)

    # TSNE ====================================================================
    if projection_tsne:
        plt.clf()
        tsne_samples = TSNE(
            n_components=2,
            init="random",
            learning_rate="auto",
            random_state=actual_seed,
            perplexity=min(30, len(data) / 4),
        ).fit_transform(data)
        plt.scatter(
            tsne_samples[:, 0],
            tsne_samples[:, 1],
            c=c.labels[0],
            cmap="gist_rainbow",
        )

        # Annotate IDs
        if projection_numbers:
            for i in range(len(pca_samples)):
                plt.annotate(i, (tsne_samples[i, 0], tsne_samples[i, 1]))
        plt.title(f"t-SNE of {system_name} (K={display_k})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.xticks(())
        plt.yticks(())
        plt.tight_layout()
        plt.savefig(
            path + f"/{treat_fname(system_name)}_TSNE_{display_k}.pdf", dpi=300
        )

    # NMI table ===============================================================

    plt.clf()
    plt.imshow(nmi_table, interpolation="none", cmap="coolwarm")
    plt.colorbar()
    plt.title(f"NMI table for K={display_k}")
    plt.tight_layout()
    plt.savefig(
        path + f"/{treat_fname(system_name)}_NMI_{display_k}.pdf", dpi=300
    )

    # Textual data ============================================================
    with open(path + f"/{treat_fname(system_name)}_output.txt", "w") as f:
        # Seed
        f.write(f"Seed: {seed}\n\n")

        # number_of_random_runs
        f.write(f"Number of random runs: {number_of_random_runs}\n\n")

        # K
        f.write(f"Number K of clusters: {display_k}\n\n")

        # Silhouette score
        f.write(f"Silhouette Score: {c.scores[0]}\n\n")

        # WCSS
        f.write(f"Within-Cluster Sum of Squares (WCSS): {c.wcss[0]}\n\n")

        # Number of atoms
        f.write(f"Number of atoms in site: {n_atoms}\n\n")

        # Fixed substrate atomic number
        if fixed_substrate_atomic_number:
            f.write(
                "Fixed substrate atomic number: "
                f"{fixed_substrate_atomic_number}\n\n"
            )
        else:
            f.write("Fixed substrate atomic number: Disabled\n\n")

        # Coulomb Matrices exponents
        f.write(f"Coulomb Matrix exponents: z={z_exp:.6f} d={d_exp:.6f}\n\n")

        # Dataset scale information
        f.write(f"Scaled dataset: {'Yes' if scale_dataset else 'No'}")

        # Site atoms data
        site_atoms_list = [" ".join(np.ndarray.tolist(x)) for x in site_atoms]

        # Labels
        f.write("\n\nLabels and Substrate Z Averages:")
        for i, file, label, avg, s_a in zip(
            range(len(c.labels[0])),
            xyz_file_names,
            c.labels[0],
            subs,
            site_atoms_list,
        ):
            f.write(
                f"\nID: {i:03d} - File: {file} - Label: {label}"
                f" - Z Avg: {avg:.6f} - Site atoms: {s_a}"
            )

        # Energies or Centroids
        if use_energy:
            # Energies
            f.write("\n\nLowest energy elements:")
            for cluster, (index, energy) in enumerate(min_energy_list):
                f.write(f"\nCluster {cluster}: {xyz_file_names[index]}")
        else:
            # Representatives
            f.write("\n\nRepresentatives:")
            for cluster, r in enumerate(representatives):
                f.write(f"\nCluster {cluster:03d}: {xyz_file_names[r]}")

        # Substrate information
        f.write("\n\nSubstrate information:")
        f.write(f"\nSubstrates Average of averages: {subs.mean():.6f}")
        f.write(f"\nSubstrates Std.Dev of averages: {subs.std():.6f}")

        # PCA Explained Variance Ratio
        f.write("\n\nPCA variance:")
        for v in pca_var:
            f.write(f" {v}")

        # k_scores info
        f.write("\n\nInformation about automatic K selection")
        for ks in k_scores:
            f.write(
                f"\nk: {ks['k']:03d} - mean: {ks['mean']:.6f}"
                f" - var: {ks['var']:.6f}"
            )

        # End file with blank line
        f.write("\n")


def site_atoms_output(
    output_folder_path: str,
    system_name: str,
    xyz_file_names: list[str],
    site_atoms: list[npt.NDArray[str]],
) -> None:
    # get path
    path = pathlib.Path(output_folder_path + "/output")
    path = str(path.resolve())

    # structure data
    site_atoms_list = [" ".join(np.ndarray.tolist(x)) for x in site_atoms]
    data = list(zip(xyz_file_names, site_atoms_list))

    with open(path + f"/{treat_fname(system_name)}_site_atoms.txt", "w") as f:
        for d in data:
            f.write(f"{d[0]} - {d[1]}\n")


def cluster_xyz_filesystem(
    labels: npt.NDArray[int],
    xyz_file_names: list[str],
    output_folder_path: str,
) -> None:
    """
    Cluster XYZs in the file system
    """
    # create directory if it does not exist
    path = pathlib.Path(output_folder_path + "/xyz")
    path.mkdir(parents=True, exist_ok=True)
    path = str(path.resolve())

    # create all cluster directories
    for label in set(labels):
        path_clus = pathlib.Path(path + f"/{label}")
        path_clus.mkdir(parents=True, exist_ok=True)

    # copy files to correct directories
    for name, label in zip(xyz_file_names, labels):
        shutil.copyfile(name, path + f"/{label}/{pathlib.Path(name).name}")


def extract_best_k(
    clusters_data: list,
) -> tuple[int, float, dict, npt.NDArray[npt.NDArray[float]]]:
    """
    Get the best K available
    """
    k_scores = []

    # compute means and variances
    for i, c in enumerate(clusters_data):
        scores = []
        for d in c:
            scores.append(d.scores[0])
        k_scores.append(
            {"k": i + 2, "mean": np.mean(scores), "var": np.var(scores)}
        )

    # compute the best K
    best_k = [0]
    best_mean = [float("-inf")]

    for s in k_scores:
        # if there is a better value, keep only it
        if s["mean"] > best_mean[0]:
            best_mean = [s["mean"]]
            best_k = [s["k"]]
        # otherwise, add to the list
        elif s["mean"] == best_mean[0]:
            best_mean.append(s["mean"])
            best_k.append(s["k"])

    # if there are multiple ks with the same mean, take the rounded mean
    if len(best_k) > 1:
        best_k = int(np.round(np.mean(best_k)))
        best_mean = best_mean[0]
    # othewise, the best k is already the only one available
    else:
        best_k = best_k[0]
        best_mean = best_mean[0]

    # generate zeroed table of NMI
    nmi_table = np.zeros(
        shape=(len(clusters_data[best_k - 2]), len(clusters_data[best_k - 2]))
    )

    # populate table of NMI with the actual values
    for i, d1 in enumerate(clusters_data[best_k - 2]):
        for j, d2 in enumerate(clusters_data[best_k - 2]):
            nmi_table[i][j] = normalized_mutual_info_score(
                d1.labels[0], d2.labels[0]
            )

    return best_k, best_mean, k_scores, nmi_table


def pick_best_candidate(
    clusters_data: list[list[Cluster]], best_k: int
) -> Cluster:
    """
    Pick the best candidate from the clusterings that have the best K
    """

    best_cluster_data = clusters_data[best_k - 2]

    best_score = float("-inf")
    best_i = -1

    for i, bcd in enumerate(best_cluster_data):
        if bcd.scores[0] > best_score:
            best_score = bcd.scores[0]
            best_i = i

    return clusters_data[best_k - 2][best_i]


def pipeline(
    seed: int,
    system_name: str,
    input_folder_path: str,
    output_folder_path: str,
    method_of_k_selection: str,
    number_k_of_clusters: int,
    silhouette_range: npt.NDArray[int],
    number_of_random_runs: int,
    molecule_indices: npt.NDArray[int],
    site_metric: str,
    metric_val: int | float,
    fixed_substrate_atomic_number: int,
    z_exp: int | float,
    d_exp: int | float,
    use_energy: bool,
    scale_dataset: bool,
    projection_numbers: bool,
    projection_representatives: bool,
    projection_tsne: bool,
) -> None:
    """
    This function is the 'real' main function that should be called either from
    CLI or GUI.
    """

    print("[1/8] Gathering data...")
    xyz_names, xyz_data = get_data_from_path(input_folder_path, use_energy)

    print("[2/8] Validating molecules...")
    validate_molecules(xyz_names, xyz_data, molecule_indices)

    print("[3/8] Building the dataset...")
    # site_size
    if site_metric == "site_size":
        dataset, subs, energies, site_atoms = build_dataset_site_size(
            xyz_names,
            xyz_data,
            molecule_indices,
            metric_val,
            fixed_substrate_atomic_number,
            z_exp,
            d_exp,
            scale_dataset,
        )
        n_atoms = metric_val

    # site_radius
    else:
        (
            dataset,
            subs,
            energies,
            site_atoms,
            n_atoms,
        ) = build_dataset_site_radius(
            xyz_names,
            xyz_data,
            molecule_indices,
            metric_val,
            fixed_substrate_atomic_number,
            z_exp,
            d_exp,
            scale_dataset,
        )

    print("[4/8] Performing clustering...")

    # number_of_random_runs == 1 ==============================================
    if number_of_random_runs == 1:
        # Negative seed results in a random seed
        kmeans_seed = np.random.randint(999999) if seed < 0 else seed

        print("[5/8] Clustering data...")
        # K chosen by the user
        if method_of_k_selection == "user":
            clusters_data = perform_clustering_single_k(
                kmeans_seed, dataset, number_k_of_clusters
            )

            k = 0  # Only one cluster
            display_k = number_k_of_clusters

        # K chosen by silhouette
        else:
            clusters_data = perform_clustering(
                kmeans_seed, dataset, silhouette_range
            )

            k = clusters_data.best_k
            display_k = clusters_data.best_k

        print(f"[6/8] Generating final output for K={display_k}...")
        output(
            k,
            display_k,
            output_folder_path,
            xyz_names,
            dataset,
            clusters_data,
            energies,
            n_atoms,
            fixed_substrate_atomic_number,
            z_exp,
            d_exp,
            use_energy,
            scale_dataset,
            projection_numbers,
            projection_representatives,
            projection_tsne,
            site_atoms,
            subs,
            kmeans_seed,
            system_name,
        )

        print("[7/8] Clustering XYZs in the file system...")
        cluster_xyz_filesystem(
            clusters_data.labels[k], xyz_names, output_folder_path
        )

    # number_of_random_runs > 1 ===============================================
    else:
        print("[5/8] Trying different clustering configurations...")
        # Negative seed results in a random seed
        np.random.seed(np.random.randint(999999) if seed < 0 else seed)
        pseudo_seeds = [
            np.random.randint(999999) for i in range(number_of_random_runs)
        ]

        # Which K to display to the user
        display_k = None

        # K chosen by the user
        if method_of_k_selection == "user":
            clusters_data = perform_clustering_many_runs_single_k(
                pseudo_seeds, dataset, number_k_of_clusters
            )

            display_k = number_k_of_clusters

        # K chosen by silhouette
        else:
            clusters_data = perform_clustering_many_runs(
                pseudo_seeds, dataset, silhouette_range
            )

        best_k, best_mean, k_scores, nmi_table = extract_best_k(
            clusters_data
        )

        result = pick_best_candidate(clusters_data, best_k)

        # if no display_k has been assigned yet, assign it as the best_k
        if display_k is None:
            display_k = result.best_k

        print(f"[6/8] Generating final output for K={display_k}...")
        output_number_of_random_runs(
            best_k,
            display_k,
            output_folder_path,
            xyz_names,
            dataset,
            result,
            k_scores,
            nmi_table,
            energies,
            n_atoms,
            fixed_substrate_atomic_number,
            z_exp,
            d_exp,
            use_energy,
            scale_dataset,
            projection_numbers,
            projection_representatives,
            projection_tsne,
            site_atoms,
            subs,
            pseudo_seeds[0],
            seed,
            number_of_random_runs,
            system_name,
        )

        print("[7/8] Clustering XYZs in the file system...")
        cluster_xyz_filesystem(result.labels[0], xyz_names, output_folder_path)

    print("[8/8] Done!")


def main() -> None:
    print("Please run either the CLI or GUI program.")


if __name__ == "__main__":
    main()
