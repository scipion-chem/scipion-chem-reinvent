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


def preprocess_smi_file(protocol, input, output):
    out_path = protocol._getPath(output)

    forbidden_atoms = ['P']

    replacements = [
        ('[C]', 'C'),
        ('[c]', 'c'),
        ('[N]', 'N'),
        ('[n]', 'n'),
        ('[N@]', 'N'),
        ('[O]', 'O'),
        ('[o]', 'o')
    ]

    total = 0
    saved = 0

    try:
        with open(input, 'r') as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                smi_original = line.strip()
                if not smi_original or any(atom in smi_original for atom in forbidden_atoms):
                    continue
                total += 1

                mol = Chem.MolFromSmiles(smi_original)

                if mol is not None:
                    smi_limpio = Chem.MolToSmiles(mol)

                    for old, new in replacements:
                        smi_limpio = smi_limpio.replace(old, new)

                    f_out.write(smi_limpio + '\n')
                    saved += 1

        print (f"SMILES preprocessed. {total - saved} SMILES were discarded. {saved} SMILES were saved.")

        return out_path

    except Exception as e:
        print("Smiles file can't be processed")
        return out_path
