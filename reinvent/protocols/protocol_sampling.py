# -*- coding: utf-8 -*-
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
# **************************************************************************


"""
This protocol generates molecules from a trained model using the REINVENT4 software.
"""
import csv
import os

import toml

from pwem.protocols import EMProtocol
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import (PointerParam, IntParam, BooleanParam, EnumParam,
                                        PathParam, FloatParam)

from pwchem.objects import SetOfSmallMolecules, SmallMolecule, SmallMoleculesLibrary

from reinvent import Plugin
from reinvent.utils.smilesUtils import preprocess_smi_file

class ReinventSampling(EMProtocol):
    """
    Protocol to generate new molecules.
    """
    _label = 'Sampling'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        """
        form.addSection(label='Run Parameters')

        RunGroup = form.addGroup('Run Parameters')
        RunGroup.addParam('NumMols', IntParam, default=100,
                      label='Number of output molecules',
                      help='Number of molecules to generate. This number is multiplied per input SMILES.')
        
        RunGroup.addParam('UniqueMols', BooleanParam, default=True,
                      label='Remove duplicated SMILES?',
                      help='If TRUE returns unique canonicalized SMILES.')

        RunGroup.addParam('RandomSmi', BooleanParam, default=True,
                      label='Shuffle atoms randomly?',
                      help='If TRUE shuffle atoms in SMILES randomly.')

        GenGroup = form.addGroup('Molecule Generator')
        GenGroup.addParam('MolGenerator', EnumParam, choices=['Reinvent'], #'LibInvent',
                                                          #'LinkInvent', 'Mol2Mol'],
                      default=0,
                      label='Type of Molecule Generator',
                      help='Generative strategy to be used. Each generator requires a specific prior and input data.\n'
                            '- Reinvent: De novo sampling.\n'
                            '- LibInvent: Find R-groups for the given scaffolds.\n'
                            '- LinkInvent: Find a scaffold to link two fragments.\n'
                            '- Mol2Mol: Find molecules similar to provided SMILES.')

        GenGroup.addParam('ExtPrior', BooleanParam, default='False', expertLevel=LEVEL_ADVANCED,
                      label='Upload external prior file?',
                      help='Set to True to select a custom prior model.')

        GenGroup.addParam('PriorModel', PointerParam,
                      pointerClass='ReinventModel',
                      condition='ExtPrior==False',
                      allowsNull=True,
                      label='Prior model file',
                      help='Select a trained model. A Learning protocol should be run first.\n'
                            ' If left empty, the default prior for the selected generator will be used.')

        GenGroup.addParam('ExtPriorModel', PathParam,
                      condition='ExtPrior==True',
                      label='Prior model file',
                      help='Path to prior model file. Each generator requires a specific prior.')
        
        GenGroup.addParam('SmiFileLib', PathParam,
                      condition='MolGenerator==1',
                      label='Scaffold SMILES file',
                      help='One scaffold per line. Each scaffold must be annotated by 2 \'*\' to locate the attachment points.\n'
                            'Up to 4 attachments points are allowed.\n'
                            'Example:\n [*:0]Cc2ccc1cncc(C[*:1])c1c2')

        GenGroup.addParam('SmiFileLink', PathParam,
                      condition='MolGenerator==2',
                      label='Warheads SMILES file',
                      help='One warhead pair per line. Each warhead must be annotated with \'*\' to locate the attachment points.'
                            'The two warheads must be separated by the pipe symbol.\n'
                            'Example:\n Oc1cncc(*)c1|*c1ccoc1')
        
        GenGroup.addParam('SmiFileMol', PathParam,
                      condition='MolGenerator==3',
                      label='Compound SMILES file',
                      help='One compound per line.')

        GenGroup.addParam('SampleStrat', EnumParam, choices=['Beamsearch','Multinomial'],
                      condition='MolGenerator==3', expertLevel=LEVEL_ADVANCED,
                      default=0,
                      label='Sampling Strategy',
                      help='Multinomial: Fast random generation based on token probability distribution.\n'
                            'Beamsearch: Deterministic approach that ensures unique compounds '
                            'by selecting the highest probability sequences.(Recommended for sampling).')

        GenGroup.addParam('Temperature', FloatParam, default=1.0,
                      condition='MolGenerator==3 and SampleStrat==1', expertLevel=LEVEL_ADVANCED,
                      label='Temperature in Multinomial sampling',
                      help='Controls randomness. Lower values for more predictable molecules and higher values for more diverse structures.')

    # -------------------------- OTHER functions ----------------------
    def _getPriorFile(self):
        if self.ExtPrior.get() is True:
            return self.ExtPriorModel.get()
        if self.PriorModel.get() is not None:
            return self.PriorModel.get().getPath()
        else:
            priorModel = ['reinvent.prior', 'libinvent.prior', 'linkinvent.prior', 'mol2mol_scaffold_generic.prior']
            priorFile = priorModel[self.MolGenerator.get()]
            return Plugin.getPriorPath(priorFile)

    def _getSmilesPath(self):
        MolGenerator = self.MolGenerator.get()

        if MolGenerator == 1:
            smiles_file = self.SmiFileLib.get()
        elif MolGenerator == 2:
            smiles_file = self.SmiFileLink.get()
        elif MolGenerator == 3:
            smiles_file = self.SmiFileMol.get()
        else:
            smiles_file = None

        return smiles_file

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('createConfigFileStep')
        self._insertFunctionStep('runReinventStep')
        self._insertFunctionStep('createOutputStep')

    def createConfigFileStep(self):
        params = {  
                'model_file': self._getPriorFile(),
                'output_file': self.getPath('sampling.csv'),
                'num_smiles': self.NumMols.get(),
                'unique_molecules': self.UniqueMols.get(),
                'randomize_smiles': self.RandomSmi.get()
         }
        smiles_file = self._getSmilesPath()
        if smiles_file:
            new_smiles_file = preprocess_smi_file(self,smiles_file,'smiles_cleaned.smi')
            params['smiles_file'] = new_smiles_file

        SampleStrat = self.SampleStrat.get()

        if self.MolGenerator.get() == 3:
            if SampleStrat == 0:
                params['sample_strategy'] = 'beamsearch'
            if SampleStrat == 1:
                params['sample_strategy'] = 'multinomial'
                params['temperature'] = self.Temperature.get()

        config_params = {
                'run_type': 'sampling',
                'device': 'cpu',
                'tb_logdir': os.path.join(self._getExtraPath(), 'TB_logs'),
                'json_out_config':self._getTmpPath('_sampling.json'),

                'parameters': params
        }

        self.configPath = self._getPath('sampling_config.toml')
        with open(self.configPath, 'w') as f:
            toml.dump(config_params, f)

    def runReinventStep(self):
        CondaInit = Plugin.getCondaActivationCmd()
        EnvActivation = Plugin.getEnvActivation()
        Executable = Plugin.getProgram()
        
        parts=[CondaInit, f'{EnvActivation} &&', Executable]
        full_command =" ".join(parts)
        self.runJob(full_command, self.configPath, env=Plugin.getEnviron())

    def createOutputStep(self):
        path_csv = self._getPath('sampling.csv')
        smi_out = self._getPath('sampling.smi')

        with open(path_csv, 'r') as f_in, open(smi_out, 'w') as f_out:
            reader = csv.reader(f_in)
            headers = next(reader)
            skip_idx = headers.index('SMILES_state')

            for i, row in enumerate(reader):
                if row and row[0].strip():
                    filtered_row = [v for j, v in enumerate(row) if j != skip_idx]
                    name = f'MOL_{str(i + 1).zfill(3)}'
                    f_out.write(f'{filtered_row[0]}\t{name}\t{filtered_row[1]}\n')

        headers = ['SMI', 'molName', 'NLL']

        outputLib = SmallMoleculesLibrary(libraryFilename=smi_out, headers=headers)
        outputLib.calculateLength()
        self._defineOutputs(outputLibrary=outputLib)


    # --------------------------- INFO functions -----------------------------------
    def _validate(self):
        errors = []

        if self.NumMols.get() <= 0:
            errors.append("Number of molecules to generate must be greater than 0.")

        if self.MolGenerator.get() != 0:
            smiles_path = self._getSmilesPath()
            if not smiles_path:
                errors.append("SMILES file must not be empty.")
            elif not os.path.isfile(smiles_path):
                errors.append(f"SMILES file is not a valid path: {smiles_path}")

        return errors


    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []
        summary.append(f"Generator type: {self.getEnumText('MolGenerator')}")
        summary.append(f"Molecules generated: {self.NumMols.get()} (per input SMILES)")

        return summary