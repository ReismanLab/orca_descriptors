{
  description = "A Nix-flake-based Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    qchem.url = "github:Nix-QChem/NixOS-QChem";
  };
  
  outputs = {
    self,
    nixpkgs,
    qchem,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      overlays = [qchem.overlays.default];
      config = {
        allowUnfree = true;
        qchem-config = {
          allowEnv = true;
          optAVX = false;
        };
      };
      
    };
    

    python-with-pkgs = pkgs.python3.withPackages (ps: [
      ps.jupyter
      ps.jupyterlab
      ps.notebook

      ps.tqdm
      ps.rdkit
      ps.numpy
      ps.semantic-version

      ps.periodictable

      (ps.buildPythonPackage {
        pname = "morfeus-ml";
        version = "0.7.2";
        pyproject = true;
        build-system = [ps.setuptools];
        src = pkgs.fetchPypi {
          pname = "morfeus-ml";
          version = "0.7.2";
          hash = "sha256-izoQKayU+yLpgt9FptQc0lOon/+t6i/ZPWzuo34U5J0=";
        };
        nativeBuildInputs = [ps.setuptools];
        propagatedBuildInputs = [
          ps.numpy
          ps.scipy
          ps.fire
        ];
      })

      ps.pandas
      ps.matplotlib
      ps.scipy

      (ps.buildPythonPackage rec {
  pname = "cclib";
  version = "1.8.1";

  src = pkgs.fetchFromGitHub {
    owner = "cclib";
    repo = pname;
    rev = "v${version}";
    hash = "sha256-XYFVSJ11MGx2dq/yYa5jaC2XsrStZCT5WzwSCelEV3U=";
  };

  pyproject = true;

  postPatch = ''
    substituteInPlace pyproject.toml --replace-fail '"versioningit~=2.0"' ""
    sed -i "/versioningit>=/d" pyproject.toml
    sed -i '/^name =.*/a version = "${version}"' pyproject.toml
    sed -i "/dynamic =/d" pyproject.toml
    echo '__version__ = "${version}"' > cclib/_version.py
  '';

  build-system = [
    ps.setuptools
  ];

  dependencies = [
    ps.numpy
    ps.scipy
    ps.periodictable
    ps.pyyaml
    ps.ase
    ps.openbabel-bindings
    ps.h5py
    ps.pyscf
    ps.iodata
    ps.pandas
    ps.biopython
    #ps.pyquante
  ];}
)

    ]);
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      buildInputs = [
        python-with-pkgs
      ];

      shellHook = ''
        venv="$(cd $(dirname $(which python)); cd ..; pwd)"
        ln -Tsf "$venv" .venv
      '';
    };
  };
}
