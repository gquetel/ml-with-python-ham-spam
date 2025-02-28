let
  inputs = import ./npins;
  pkgs = import inputs.nixpkgs { };
  pythonEnv = pkgs.python312.withPackages (ps: [
    ps.spacy
    ps.spacy-models.en_core_web_sm
    ps.pandas
    ps.numpy
    ps.torch
    ps.scikit-learn
    ps.plotly
    ps.kaleido
  ]);
in
pkgs.mkShell rec {
  packages = [
    pythonEnv
  ];
  shellHook = ''
    export CUSTOM_INTERPRETER_PATH="${pythonEnv}/bin/python"
  '';
}
