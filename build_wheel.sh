#!/usr/bin/env bash
set -euo pipefail

target_dir="modulos_comprimidos"
old_dir="$target_dir/old"
timestamp="$(date +'%Y%m%d_%H%M%S')"

# Si ya hay .whl en modulos_comprimidos, moverlos a modulos_comprimidos/old renombrados con timestamp
if [ -d "$target_dir" ]; then
  shopt -s nullglob
  existing=( "$target_dir"/*.whl )
  if [ ${#existing[@]} -gt 0 ]; then
    mkdir -p "$old_dir"
    i=1
    for f in "${existing[@]}"; do
      ext="${f##*.}"
      mv -- "$f" "$old_dir/${timestamp}_$i.$ext"
      i=$((i+1))
    done
  fi
  shopt -u nullglob
fi

# Construir wheel
py -m build --wheel

# Crear carpeta destino y mover .whl
mkdir -p "$target_dir"
shopt -s nullglob
for f in dist/*.whl; do
  mv -v "$f" "$target_dir/"
done
shopt -u nullglob

# Eliminar artefactos solicitados
rm -rf dist build modelado_multidimensional_engagement.egg-info

echo "Listo. Los .whl están en: $target_dir"
```# filepath: c:\Users\brend\OneDrive\Escritorio\Estudios\Master\TFM\Modelado-Multidimensional-del-Engagement-en-E-commerce\build_wheel.sh
#!/usr/bin/env bash
set -euo pipefail

# Ir a la raíz del proyecto (ruta del script -> ../)
cd "$(dirname "$0")/.." || exit 1

target_dir="modulos_comprimidos"
old_dir="$target_dir/old"
timestamp="$(date +'%Y%m%d_%H%M%S')"

# Si ya hay .whl en modulos_comprimidos, moverlos a modulos_comprimidos/old renombrados con timestamp
if [ -d "$target_dir" ]; then
  shopt -s nullglob
  existing=( "$target_dir"/*.whl )
  if [ ${#existing[@]} -gt 0 ]; then
    mkdir -p "$old_dir"
    i=1
    for f in "${existing[@]}"; do
      ext="${f##*.}"
      mv -- "$f" "$old_dir/${timestamp}_$i.$ext"
      i=$((i+1))
    done
  fi
  shopt -u nullglob
fi

# Construir wheel
py -m build --wheel

# Crear carpeta destino y mover .whl
mkdir -p "$target_dir"
shopt -s nullglob
for f in dist/*.whl; do
  mv -v "$f" "$target_dir/"
done
shopt -u nullglob

# Eliminar artefactos solicitados
rm -rf dist build modelado_multidimensional_engagement.egg-info

echo "Listo. Los .whl están en: $target_dir"