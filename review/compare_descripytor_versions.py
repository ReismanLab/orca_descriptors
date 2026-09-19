"""Compare descriptor output of the vendored DescriPyTor snapshot against an upstream checkout.

Usage (run from the repo root, in the gp_env environment):
    python review/compare_descripytor_versions.py vendored <folder_with_one_feather>
    python review/compare_descripytor_versions.py upstream <folder_with_one_feather> <path_to_upstream_clone>

Picks the first Ni atom (or atom 1 if no Ni) and its first two bonded neighbours, then
prints charges, Sterimol, bond lengths, vibrations, dipole and angle so the two runs can be
diffed. See review/DESCRIPYTOR_UPGRADE.md for the results.
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
which, folder = sys.argv[1], os.path.abspath(sys.argv[2])
if which == "upstream":
    sys.path.insert(0, os.path.abspath(sys.argv[3]))
    from M2_data_extractor.data_extractor import Molecules
else:
    root = os.path.abspath("DescriPyTor/MolFeatures")
    for p in [root, f"{root}/M2_data_extractor", f"{root}/utils"]:
        sys.path.insert(0, p)
    from data_extractor import Molecules
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    mols = Molecules(molecules_dir_name=folder)
out = {}
m = mols.molecules[0]
out["n_atoms"] = len(m.xyz_df)
out["atoms"] = list(m.xyz_df["atom"][:8])
# pick Ni and its first two neighbours from connectivity
ni_rows = m.xyz_df.index[m.xyz_df["atom"] == "Ni"]
ni = int(ni_rows[0]) + 1 if len(ni_rows) else 1
out["Ni_index"] = ni
bonds = m.bonds_df
nb = sorted(set(int(b) for a,b in zip(bonds.iloc[:,0], bonds.iloc[:,1]) if int(a)==ni) | set(int(a) for a,b in zip(bonds.iloc[:,0], bonds.iloc[:,1]) if int(b)==ni))
out["Ni_neighbours"] = nb
a1, a2 = nb[0], nb[1]
with contextlib.redirect_stdout(buf):
    ch = mols.get_charge_df_dict(atom_indices=[ni, a1, a2]); out["charges"] = ch.iloc[0].round(5).to_dict()
    st = mols.get_sterimol_dict(atom_indices=[[ni, a1]]); out["sterimol"] = st.iloc[0].round(4).to_dict()
    bl = mols.get_bond_length_dict(atom_pairs=[[ni, a1],[ni, a2]]); out["bond_length"] = bl.iloc[0].round(4).to_dict()
    vb = mols.get_stretch_vibration_dict(atom_pairs=[[ni, a1]]); out["stretch"] = vb.iloc[0].round(3).to_dict()
    bd = mols.get_bend_vibration_dict(atom_pairs=[[a1, a2]]); out["bend"] = bd.iloc[0].round(3).to_dict()
    dp = mols.get_dipole_dict(atom_indices=[[ni, a1, a2], ni, a1]); out["dipole"] = dp.iloc[0].round(4).to_dict()
    an = mols.get_bond_angle_dict(atom_indices=[[a1, ni, a2]]); out["angle"] = an.iloc[0].round(3).to_dict()
print(json.dumps({k: (v if not isinstance(v, dict) else {str(kk): (None if vv != vv else vv) for kk, vv in v.items()}) for k, v in out.items()}, indent=1, default=str))
