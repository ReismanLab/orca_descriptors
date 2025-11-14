{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  # inputs.qchem.url = "github:markuskowa/NixOS-QChem/master";
  outputs = {
    self,
    nixpkgs,
    #qchem,
  }: let
    pkgs = import nixpkgs {
      #import qchem.inputs.nixpkgs {
      system = "x86_64-linux";
      #overlays = [qchem.overlays.default];
      config = {
        allowUnfree = true;
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
