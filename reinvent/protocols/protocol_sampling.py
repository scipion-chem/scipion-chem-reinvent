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

        runGroup = form.addGroup('Run Parameters')
        runGroup.addParam('numMols', IntParam, default=100,
                      label='Number of output molecules',
                      help='Number of molecules to generate. This number is multiplied per input SMILES.')

        runGroup.addParam('uniqueMols', BooleanParam, default=True,
                      label='Remove duplicated SMILES?',
                      help='If TRUE returns unique canonicalized SMILES.')

        runGroup.addParam('randomSmi', BooleanParam, default=True,
                      label='Shuffle atoms randomly?',
                      help='If TRUE shuffle atoms in SMILES randomly.')

        genGroup = form.addGroup('Molecule Generator')
        genGroup.addParam('molGenerator', EnumParam, choices=['Reinvent'], #, 'LibInvent', 'LinkInvent', 'Mol2Mol'],
                      default=0,
                      label='Type of Molecule Generator',
                      help='Generative strategy to be used. Each generator requires a specific prior and input data.\n'
                            '- Reinvent: De novo sampling.\n'
                            '- LibInvent: Find R-groups for the given scaffolds.\n'
                            '- LinkInvent: Find a scaffold to link two fragments.\n'
                            '- Mol2Mol: Find molecules similar to provided SMILES.')

        genGroup.addParam('extPrior', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Upload external prior file?',
                      help='Set to True to select a custom prior model.')

        genGroup.addParam('priorModel', PointerParam,
                      pointerClass='ReinventModel',
                      condition='extPrior==False',
                      allowsNull=True,
                      label='Prior model file',
                      help='Select a trained model. A Learning protocol should be run first.\n'
                            ' If left empty, the default prior for the selected generator will be used.')

        genGroup.addParam('extPriorModel', PathParam,
                      condition='extPrior==True',
                      label='Prior model file',
                      help='Path to prior model file. Each generator requires a specific prior.')

        genGroup.addParam('smiFileLib', PathParam,
                      condition='molGenerator==1',
                      label='Scaffold SMILES file',
                      help='One scaffold per line. Each scaffold must be annotated by 2 \'*\' to locate the attachment points.\n'
                            'Up to 4 attachments points are allowed.\n'
                            'Example:\n [*:0]Cc2ccc1cncc(C[*:1])c1c2')

        genGroup.addParam('smiFileLink', PathParam,
                      condition='molGenerator==2',
                      label='Warheads SMILES file',
                      help='One warhead pair per line. Each warhead must be annotated with \'*\' to locate the attachment points.'
                            'The two warheads must be separated by the pipe symbol.\n'
                            'Example:\n Oc1cncc(*)c1|*c1ccoc1')

        genGroup.addParam('smiFileMol', PathParam,
                      condition='molGenerator==3',
                      label='Compound SMILES file',
                      help='One compound per line.')

        genGroup.addParam('sampleStrat', EnumParam, choices=['Beamsearch','Multinomial'],
                      condition='molGenerator==3', expertLevel=LEVEL_ADVANCED,
                      default=0,
                      label='Sampling Strategy',
                      help='Multinomial: Fast random generation based on token probability distribution.\n'
                            'Beamsearch: Deterministic approach that ensures unique compounds '
                            'by selecting the highest probability sequences.(Recommended for sampling).')

        genGroup.addParam('temperature', FloatParam, default=1.0,
                      condition='molGenerator==3 and sampleStrat==1', expertLevel=LEVEL_ADVANCED,
                      label='Temperature in Multinomial sampling',
                      help='Controls randomness. Lower values for more predictable molecules and higher values for more diverse structures.')

    # -------------------------- OTHER functions ----------------------
    def _getPriorFile(self):
        if self.extPrior.get() is True:
            return self.extPriorModel.get()
        if self.priorModel.get() is not None:
            return self.priorModel.get().getPath()
        else:
            priorModel = ['reinvent.prior', 'libinvent.prior', 'linkinvent.prior', 'mol2mol_scaffold_generic.prior']
            priorFile = priorModel[self.molGenerator.get()]
            return Plugin.getPriorPath(priorFile)

    def _getSmilesPath(self):
        molGenerator = self.molGenerator.get()

        if molGenerator == 1:
            smilesFile = self.smiFileLib.get()
        elif molGenerator == 2:
            smilesFile = self.smiFileLink.get()
        elif molGenerator == 3:
            smilesFile = self.smiFileMol.get()
        else:
            smilesFile = None

        return smilesFile

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.createConfigFileStep)
        self._insertFunctionStep(self.runReinventStep)
        self._insertFunctionStep(self.createOutputStep)

    def createConfigFileStep(self):
        params = {
                'model_file': self._getPriorFile(),
                'output_file': self._getPath('sampling.csv'),
                'num_smiles': self.numMols.get(),
                'unique_molecules': self.uniqueMols.get(),
                'randomize_smiles': self.randomSmi.get()
         }
        smilesFile = self._getSmilesPath()
        if smilesFile:
            newSmilesFile = preprocess_smi_file(self, smilesFile, 'smiles_cleaned.smi')
            params['smiles_file'] = newSmilesFile

        sampleStrat = self.sampleStrat.get()

        if self.molGenerator.get() == 3:
            if sampleStrat == 0:
                params['sample_strategy'] = 'beamsearch'
            if sampleStrat == 1:
                params['sample_strategy'] = 'multinomial'
                params['temperature'] = self.temperature.get()

        configParams = {
                'run_type': 'sampling',
                'device': 'cpu',
                'tb_logdir': os.path.join(self._getExtraPath(), 'TB_logs'),
                'json_out_config': self._getTmpPath('_sampling.json'),

                'parameters': params
        }

        self.configPath = self._getPath('sampling_config.toml')
        with open(self.configPath, 'w') as f:
            toml.dump(configParams, f)

    def runReinventStep(self):
        condaInit = Plugin.getCondaActivationCmd()
        envActivation = Plugin.getEnvActivation()
        executable = Plugin.getProgram()

        parts = [condaInit, f'{envActivation} &&', executable]
        fullCommand = " ".join(parts)
        self.runJob(fullCommand, self.configPath, env=Plugin.getEnviron())

    def createOutputStep(self):
        pathCsv = self._getPath('sampling.csv')
        smiOut = self._getPath('sampling.smi')

        with open(pathCsv, 'r') as fIn, open(smiOut, 'w') as fOut:
            reader = csv.reader(fIn)
            headers = next(reader)
            skipIdx = headers.index('SMILES_state')

            for i, row in enumerate(reader):
                if row and row[0].strip():
                    filteredRow = [v for j, v in enumerate(row) if j != skipIdx]
                    name = f'MOL_{str(i + 1).zfill(3)}'
                    fOut.write(f'{filteredRow[0]}\t{name}\t{filteredRow[1]}\n')

        headers = ['SMI', 'molName', 'NLL']

        outputLib = SmallMoleculesLibrary(libraryFilename=smiOut, headers=headers)
        outputLib.calculateLength()
        self._defineOutputs(outputLibrary=outputLib)


    # --------------------------- INFO functions -----------------------------------
    def _validate(self):
        errors = []

        if self.numMols.get() <= 0:
            errors.append("Number of molecules to generate must be greater than 0.")

        if self.molGenerator.get() != 0:
            smilesPath = self._getSmilesPath()
            if not smilesPath:
                errors.append("SMILES file must not be empty.")
            elif not os.path.isfile(smilesPath):
                errors.append(f"SMILES file is not a valid path: {smilesPath}")

        return errors


    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []
        summary.append(f"Generator type: {self.getEnumText('molGenerator')}")
        summary.append(f"Molecules generated: {self.numMols.get()} (per input SMILES)")

        return summary
