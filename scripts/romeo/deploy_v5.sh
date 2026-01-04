#!/bin/bash
# Script de déploiement v5 Work Stealing sur Romeo

ROMEO_USER="nimarano"
ROMEO_HOST="romeo1.univ-reims.fr"
REMOTE_DIR="~/golomb"

echo "=== Déploiement v5 Work Stealing sur Romeo ==="
echo ""

# 1. Synchroniser les fichiers source
echo "1. Synchronisation des fichiers..."
rsync -avz --exclude='build/' --exclude='*.o' --exclude='results/' \
    /mnt/c/Users/nicol/Desktop/golomb/ \
    ${ROMEO_USER}@${ROMEO_HOST}:${REMOTE_DIR}/

# 2. Compiler sur Romeo
echo ""
echo "2. Compilation sur Romeo..."
ssh ${ROMEO_USER}@${ROMEO_HOST} << 'EOF'
cd ~/golomb
module load gcc/11.2.0
module load openmpi/4.1.2

echo "Compiling all versions..."
make clean
make sequential
make parallel

echo ""
echo "Versions compilées:"
ls -la build/
EOF

# 3. Test rapide
echo ""
echo "3. Test rapide de v5..."
ssh ${ROMEO_USER}@${ROMEO_HOST} << 'EOF'
cd ~/golomb
module load gcc/11.2.0
module load openmpi/4.1.2

echo "Test G9 avec 4 processus:"
mpirun -np 4 ./build/golomb_mpi_v5 9
EOF

echo ""
echo "=== Déploiement terminé ==="
echo ""
echo "Pour lancer les benchmarks:"
echo "  ssh ${ROMEO_USER}@${ROMEO_HOST}"
echo "  cd ~/golomb"
echo "  sbatch scripts/romeo/compare_all_versions.sh"
