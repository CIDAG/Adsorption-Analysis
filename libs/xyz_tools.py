# Simple tools to work with xyz files

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class XYZData:
    """
    Data type to store XYZ file information. Includes basic operations.
    """

    atoms: npt.NDArray[str]
    coords: npt.NDArray[float]
    energy: float

    def __getitem__(self, ix: npt.NDArray[int] | int) -> XYZData:
        """Get item(s) from XYZ Data"""
        return XYZData(self.atoms[ix], self.coords[ix], self.energy)

    def __len__(self) -> int:
        """Return XYZ Data length"""
        return len(self.atoms)

    def __add__(self, other: XYZData) -> XYZData:
        """Concatenate XYZ Data"""

        # if the energies differ, raise an error
        if self.energy != other.energy:
            print("[ERROR] Different energy values for XYZDatas")
            exit()

        return XYZData(
            np.append(self.atoms, other.atoms),
            np.vstack([self.coords, other.coords]),
            self.energy
        )


def read_xyz(
    fname: str, use_energy: bool
) -> tuple[npt.NDArray[str], npt.NDArray[float], npt.NDArray[float]]:
    """
    Read .xyz file and return atoms and coords
    """
    with open(fname, "r") as f:
        file = f.readlines()

        # delete last line if it is empty
        if not file[-1].split():
            del(file[-1])

        # xyz data
        xyz = np.array([x.split() for x in file[2:]])

        if use_energy:
            # energy data
            if len(file[0].split()) > 1:
                energy = float(file[0].split()[1])
            else:
                print(f"[ERROR] Energy not found in file {fname}")
                exit()
        else:
            energy = None

        atoms = xyz[:, 0]
        coords = xyz[:, 1:].astype(float)

    return XYZData(atoms, coords, energy)


def coulomb_matrix(
    xyz: XYZData, z_exp: float = 2.4, d_exp: float = 1.0
) -> npt.NDArray[float]:
    """
    Calculate and return the coulomb matrix of give molecule
    """
    n_atoms = len(xyz)
    coulomb_m = np.zeros((n_atoms, n_atoms))
    charge = [atomic_num[symbol] for symbol in xyz.atoms]

    for i in range(n_atoms):
        # Diagonal term described by Potential energy of isolated atom
        coulomb_m[i, i] = 0.5 * charge[i] ** z_exp

        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(xyz.coords[i, :] - xyz.coords[j, :])
            # Pair-wise repulsion
            coulomb_m[j, i] = charge[i] * charge[j] / (dist ** d_exp)
            coulomb_m[i, j] = coulomb_m[j, i]

    return coulomb_m


def eigen_coulomb(
    xyz: XYZData, z_exp: float = 2.4, d_exp: float = 1.0
) -> npt.NDArray[float]:
    """
    Calculate and return the eigenvalues of the coulomb matrix
    """
    s_coulomb = coulomb_matrix(xyz, z_exp, d_exp)
    eigen_values = -np.sort(-np.linalg.eigvals(s_coulomb)).real

    return eigen_values[0 : len(xyz)]


def get_symbol(z: int) -> str:
    """
    Given an atomic number, return the corresponding symbol
    """
    return list(atomic_num.keys())[z - 1]


# Atomic numbers for all atom symbols
atomic_num = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Uub": 112,
    "Uut": 113,
    "Uuq": 114,
    "Uup": 115,
    "Uuh": 116,
    "Uus": 117,
    "Uuo": 118,
}
