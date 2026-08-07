# **************************************************************************
# *
# * Authors:     Izana Alcalde (izana.alcalde@alumnos.upm.es)
# *
# * Universidad Politécnica de Madrid
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# *************************************************************************
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from pwchem.objects import SmallMoleculesLibrary

# Neutralizes formal charges (protonated amines [NH+], carboxylates [O-], ...) that the
# priors reject; instantiated once and reused.
_UNCHARGER = rdMolStandardize.Uncharger()

# Element alphabet supported by the REINVENT priors (ChEMBL-based). Molecules containing any
# other element (e.g. I, P, B, Si) are rejected by REINVENT's tokenizer
SUPPORTED_ELEMENTS = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H', '*'}

REPLACEMENTS = [
    ('[C]', 'C'),
    ('[c]', 'c'),
    ('[N]', 'N'),
    ('[n]', 'n'),
    ('[N@]', 'N'),
    ('[N@@]', 'N'),
    ('[O]', 'O'),
    ('[o]', 'o')
]


def _clean_smi(smiOriginal):
    """Parse, desalt, neutralize and filter a single SMILES. Returns None if it can't be used."""
    mol = Chem.MolFromSmiles(smiOriginal)
    if mol is None:
        return None

    # Desalt: keep only the largest fragment. The REINVENT priors are trained on
    # single-component molecules, so salts/mixtures (e.g. '.Cl') break the tokenizer.
    hasWildcard = any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
    if not hasWildcard:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())

    # Neutralize formal charges
    mol = _UNCHARGER.uncharge(mol)

    # Skip molecules with elements outside the prior's supported alphabet
    if any(atom.GetSymbol() not in SUPPORTED_ELEMENTS for atom in mol.GetAtoms()):
        return None

    smiClean = Chem.MolToSmiles(mol)
    for old, new in REPLACEMENTS:
        smiClean = smiClean.replace(old, new)
    return smiClean


def extract_smiles_to_file(protocol, molInput, outputFilename):
    """Write one SMILES per line to outputFilename, reading from either a SmallMoleculesLibrary
    (its own flat SMI column) or a SetOfSmallMolecules (one .smi/.sdf/.mol2 file per molecule)."""
    outputPath = protocol._getPath(outputFilename)

    if isinstance(molInput, SmallMoleculesLibrary):
        smiIdx = molInput.getHeaders().index('SMI')
        with open(molInput.getFileName(), 'r') as fin, open(outputPath, 'w') as fout:
            for line in fin:
                row = line.rstrip('\n').split('\t')
                if len(row) > smiIdx and row[smiIdx].strip():
                    fout.write(row[smiIdx].strip() + '\n')
        return outputPath

    with open(outputPath, 'w') as fout:
        for mol in molInput:
            molFile = mol.getFileName()

            if molFile.endswith('.smi'):
                with open(molFile, 'r') as fin:
                    smiles = fin.readline().split()[0]
                    fout.write(smiles + '\n')

            elif molFile.endswith('.sdf'):
                for rdmol in Chem.SDMolSupplier(molFile):
                    if rdmol:
                        fout.write(Chem.MolToSmiles(rdmol) + '\n')

            elif molFile.endswith('.mol2'):
                rdmol = Chem.MolFromMol2File(molFile)
                if rdmol:
                    fout.write(Chem.MolToSmiles(rdmol) + '\n')

    return outputPath


def get_input_length(molInput):
    """Number of molecules in either a SmallMoleculesLibrary or a SetOfSmallMolecules."""
    if isinstance(molInput, SmallMoleculesLibrary):
        return molInput.getLength() or molInput.calculateLength()
    return len(molInput)


def preprocess_smi_file(protocol, input, output, separator=None):
    """Clean a SMILES file, one molecule per line.

    If separator is given (e.g. LinkInvent's '|'-joined warhead pairs), each line is split on it,
    every part is cleaned independently, and the line is only kept if all parts survive.
    """
    out_path = protocol._getPath(output)

    total = 0
    saved = 0

    try:
        with open(input, 'r') as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                smiOriginal = line.strip()
                if not smiOriginal:
                    continue
                total += 1

                if separator:
                    cleanedParts = [_clean_smi(part) for part in smiOriginal.split(separator)]
                    if any(part is None for part in cleanedParts):
                        continue
                    f_out.write(separator.join(cleanedParts) + '\n')
                else:
                    smiClean = _clean_smi(smiOriginal)
                    if smiClean is None:
                        continue
                    f_out.write(smiClean + '\n')

                saved += 1

        protocol.info("SMILES preprocessed. %d discarded (invalid or unsupported elements), %d saved."
                      % (total - saved, saved))
        return out_path

    except Exception as e:
        protocol.info("SMILES file can't be processed: %s" % e)
        return out_path
