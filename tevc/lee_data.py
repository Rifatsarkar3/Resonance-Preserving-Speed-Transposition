"""Lee, Kim & Kim (2024), Data in Brief 57:110940, multi-domain vibration dataset, subset 1 (deep groove ball bearing
6204): Mendeley Data 10.17632/53vtnjy6c6.1, CC BY 4.0. The pre-registered held-out dataset of the MSSP RPST-DA
programme (spec docs/superpowers/specs/2026-10-18-mssp-rpst-da-design.md §4).

Testbed: BLDC motor, shaft with rotor disk, two bearing housings; one uniaxial PCB 333D01 accelerometer on top of
the shaft-end housing. Files {rotor}_{bearing}_{fs kHz}_{model}_{rpm}.mat; used here: healthy rotor ('H_'), 16 kHz,
bearing H / B / IR / OR -> classes 0-3, nominal 600-1600 rpm; 80 s = 1,280,000 samples in field 'Data' (g).
6204: 8 balls, d = 7.94 mm, pitch diameter 34.57 mm, contact angle 0 (BPFO 3.08, BPFI 4.92 f_s).

Until the author approves the Lee protocol, only the 800-rpm library records may be read: record() refuses every
other speed while TARGETS_RELEASED is False, and the commit that freezes the approved protocol sets it to True.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

FS = 16000.0
ROOT = Path("data/raw/LEE")
SPEEDS_RPM = (600, 800, 1000, 1200, 1400, 1600)
LIBRARY_RPM = 800
BEARING = ("H", "B", "IR", "OR")
GEOMETRY = {"N_balls": 8, "d_mm": 7.94, "D_mm": 34.57, "alpha_deg": 0.0}
SPEED_BAND = (0.85, 1.05)       # frozen 2026-10-18 by the library-only rule (tevc.probes.lee_speed_band;
                                # outputs/tevc/lee_speed_band.json): library 1.001-1.004 x nominal, the default
TARGETS_RELEASED = False        # True only from the commit that freezes the author-approved Lee protocol
SHA256 = {                      # Mendeley's published manifest (public API, 2026-10-18)
    "H_H_16_6204_600.mat": "1508685ec732475044573bde40edde83cd7d0f6897d4df38e98d40ae88e740f6",
    "H_B_16_6204_600.mat": "e0efe1a08af9acee2d80adfc35cb2b612771b497b68b5d2fc0da9200633d144a",
    "H_IR_16_6204_600.mat": "953eff0fa4b85b40755df1eb7ce8f886113de1ab6ac60455ff26802d788f740d",
    "H_OR_16_6204_600.mat": "80a749a6ad9e47fdf09d2e1f127ae4b2ac8dcfd0d7c3a44c1f871290fa7ef98a",
    "H_H_16_6204_800.mat": "85ca770fe4fe8b40ca7b055aad265d80ee32ab1735c1256b873b01a508f49820",
    "H_B_16_6204_800.mat": "39db429c6624809da92f89874f8d5a90c4e6619ec5663ae9f19893495603c66d",
    "H_IR_16_6204_800.mat": "751b9635c4fc3d45142ec90bb439f3f6b11c3cbb06fbc4b6809cc12669e3d8ce",
    "H_OR_16_6204_800.mat": "422ad44804c23b9e738696640c2c2d31fa5d852d8bbcd340a41fe1fe823c0b02",
    "H_H_16_6204_1000.mat": "b42e3a48d22653f6d3eff11615bd983cbcf350cb6add1fe32488c28dfe03f149",
    "H_B_16_6204_1000.mat": "7a2667a2ed98711288de6a650a47035bfe48d5fca7f9cc990fc21c0bcc90bdf3",
    "H_IR_16_6204_1000.mat": "68af6fe3730e457980e73396fe6b53cfe161ff3be677f42e2b0bd18f61c54eb0",
    "H_OR_16_6204_1000.mat": "395779899cfa238984c252f8e2447791953c45bd7387912d30896196393914e9",
    "H_H_16_6204_1200.mat": "0d7c7cc93433cf2d9dfed1b3718ff1d0ef1a5699c08ce590178a18a3ca5d04bb",
    "H_B_16_6204_1200.mat": "fe5cddd3cda93ffd8ef1ba0493d1df8a571f6cd3ecf7bc836bfe20396e05381f",
    "H_IR_16_6204_1200.mat": "4c9509c11dcd98aafcd8903025a6a9458361cc58c1eef849c6b7623cbe38a7c8",
    "H_OR_16_6204_1200.mat": "42529532fe3785ae8521b60b43c510e0f26afe3411d60d77ede0c928a99fa24f",
    "H_H_16_6204_1400.mat": "b3bbfa641a61788820056deee40c8186de002311a5ab9a8be1df99986d92b11e",
    "H_B_16_6204_1400.mat": "4fb5dd11d03a0c45217e0cc8e9db827ca7ba99980d0da9ddf89784f99504b971",
    "H_IR_16_6204_1400.mat": "1c1f51e7ee1e3182a3494df795e702634ea00795dfdb62e186d576acf435936e",
    "H_OR_16_6204_1400.mat": "7595a402eaada09bd455e95cfba6d12275d5f73460be7730446800ecb0c01dcd",
    "H_H_16_6204_1600.mat": "3efeedb4d8f901b2fd6b610dfbec107cde3e5b1b9b6e2a229c37d79a50e935ca",
    "H_B_16_6204_1600.mat": "76e37ad2f40182c0294076dc10cfe2654399ea15a2e17088a37eea7a2ebe7ced",
    "H_IR_16_6204_1600.mat": "71bd7333369eb2c2dd395b6177ff45e7deef7dc7dea86970c5ff773ab31b1ca1",
    "H_OR_16_6204_1600.mat": "aa3f2f2631df101df68a4725ac9b4d6dc56fb206a7ff925e1e22f7d293ba3d17",
}


def record_path(cls, rpm, root=None):
    """The healthy-rotor 16 kHz record of bearing class cls (0-3) at nominal speed rpm."""
    return Path(root or ROOT) / f"H_{BEARING[cls]}_16_6204_{rpm}.mat"


def verify(root=None):
    """Names of manifest files that are missing under root or whose SHA-256 differs from the manifest."""
    bad = []
    for name, digest in sorted(SHA256.items()):
        path = Path(root or ROOT) / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    return bad


def _load(path):
    """Field 'Data' of a MAT file as float32: MAT v5 via scipy, MAT v7.3 (HDF5) via h5py."""
    import scipy.io as sio
    try:
        return np.asarray(sio.loadmat(path, variable_names=["Data"])["Data"], dtype=np.float32).ravel()
    except NotImplementedError:                       # v7.3
        import h5py
        with h5py.File(path, "r") as f:
            return np.asarray(f["Data"][()], dtype=np.float32).ravel()


@lru_cache(maxsize=None)
def record(cls, rpm):
    """Raw vibration (g) of one record; target speeds stay unread until the protocol is approved and frozen."""
    if rpm != LIBRARY_RPM and not TARGETS_RELEASED:
        raise PermissionError(f"{rpm} rpm is a target speed: Lee target records are read only after the author "
                              "approves the Lee protocol (spec §4)")
    return _load(record_path(cls, rpm))


class LEE:
    """Adapter with the window/cap attributes of tevc.datasets (spec §4): 0.5 s windows, hop 0.25 s; the library uses
    whole 800-rpm records, a target record gives U (first 40 s) and T (last 40 s)."""
    name, fs, win, hop = "lee", FS, 8000, 4000
    cap_fit, cap_eval = 80, 80
    n_classes = 4
    geometry = GEOMETRY
    _span = {"library": (0.0, 1.0), "U": (0.0, 0.5), "T": (0.5, 1.0)}

    @classmethod
    def entries(cls, role, rpm):
        lo, hi = cls._span[role]
        return [(c, record_path(c, rpm).stem, record(c, rpm), lo, hi) for c in range(cls.n_classes)]
