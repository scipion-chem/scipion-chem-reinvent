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
    ('[O]', 'O'),
    ('[o]', 'o')
]


def preprocess_smi_file(protocol, input, output):
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

                mol = Chem.MolFromSmiles(smiOriginal)
                if mol is None:
                    continue

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
                    continue

                smiClean = Chem.MolToSmiles(mol)
                for old, new in REPLACEMENTS:
                    smiClean = smiClean.replace(old, new)

                f_out.write(smiClean + '\n')
                saved += 1

        protocol.info("SMILES preprocessed. %d discarded (invalid or unsupported elements), %d saved."
                      % (total - saved, saved))
        return out_path

    except Exception as e:
        protocol.info("SMILES file can't be processed: %s" % e)
        return out_path
