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
from reinvent.utils.smilesUtils import preprocess_smi_file, extract_smiles_to_file, get_input_length

LINK_INVENT = 'molGenerator==2'

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
        genGroup.addParam('molGenerator', EnumParam, choices=['Reinvent', 'LibInvent', 'LinkInvent', 'Mol2Mol'],
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

        genGroup.addParam('smiFileLib', PointerParam,
                      pointerClass='SetOfSmallMolecules,SmallMoleculesLibrary',
                      condition='molGenerator==1',
                      label='Scaffold SMILES set',
                      help='Set of scaffold molecules; only their SMILES is used, one scaffold per molecule.\n'
                            'Each scaffold must be annotated by 2 \'*\' to locate the attachment points.\n'
                            'Up to 4 attachments points are allowed.\n'
                            'Example:\n [*:0]Cc2ccc1cncc(C[*:1])c1c2')

        genGroup.addParam('smiFileLinkA', PointerParam,
                      pointerClass='SetOfSmallMolecules,SmallMoleculesLibrary',
                      condition=LINK_INVENT,
                      label='Warhead set 1',
                      help='First warhead of each pair. Each warhead must be annotated with \'*\' to locate the attachment point.\n'
                            'Paired by order with "Warhead set 2" (1st with 1st, 2nd with 2nd...); '
                            'both sets must have the same number of molecules.')

        genGroup.addParam('smiFileLinkB', PointerParam,
                      pointerClass='SetOfSmallMolecules,SmallMoleculesLibrary',
                      condition=LINK_INVENT,
                      label='Warhead set 2',
                      help='Second warhead of each pair, paired by order with "Warhead set 1".')

        genGroup.addParam('smiFileMol', PointerParam,
                      pointerClass='SetOfSmallMolecules,SmallMoleculesLibrary',
                      condition='molGenerator==3',
                      label='Compound SMILES set',
                      help='Set of reference molecules; only their SMILES is used, one compound per molecule.')

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

    def _extractPairedSmilesToFile(self, molSetA, molSetB, outputFilename):
        pathA = extract_smiles_to_file(self, molSetA, 'warheads_A_raw.smi')
        pathB = extract_smiles_to_file(self, molSetB, 'warheads_B_raw.smi')

        outputPath = self._getPath(outputFilename)
        with open(pathA) as fA, open(pathB) as fB, open(outputPath, 'w') as fout:
            for lineA, lineB in zip(fA, fB):
                smiA, smiB = lineA.strip(), lineB.strip()
                if smiA and smiB:
                    fout.write(f'{smiA}|{smiB}\n')

        return outputPath

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
        molGenerator = self.molGenerator.get()
        if molGenerator == 1:
            rawSmilesFile = extract_smiles_to_file(self, self.smiFileLib.get(), 'smiles_raw.smi')
            params['smiles_file'] = preprocess_smi_file(self, rawSmilesFile, 'smiles_cleaned.smi')
        elif molGenerator == 2:
            rawSmilesFile = self._extractPairedSmilesToFile(self.smiFileLinkA.get(), self.smiFileLinkB.get(),
                                                             'warheads_raw.smi')
            params['smiles_file'] = preprocess_smi_file(self, rawSmilesFile, 'smiles_cleaned.smi', separator='|')
        elif molGenerator == 3:
            rawSmilesFile = extract_smiles_to_file(self, self.smiFileMol.get(), 'smiles_raw.smi')
            params['smiles_file'] = preprocess_smi_file(self, rawSmilesFile, 'smiles_cleaned.smi')

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
            csvHeaders = next(reader)
            skipIdx = csvHeaders.index('SMILES_state')
            # Columns besides SMILES/SMILES_state depend on the generator: Reinvent only adds
            # NLL, LibInvent/LinkInvent add Scaffold|Warheads + R-groups|Linker + NLL, Mol2Mol
            # adds Input_SMILES + Tanimoto + NLL. Keep whatever REINVENT reports instead of
            # assuming a fixed column count.
            extraHeaders = [h for j, h in enumerate(csvHeaders) if j not in (0, skipIdx)]

            for i, row in enumerate(reader):
                if row and row[0].strip():
                    filteredRow = [v for j, v in enumerate(row) if j != skipIdx]
                    name = f'MOL_{str(i + 1).zfill(3)}'
                    fOut.write('\t'.join([filteredRow[0], name] + filteredRow[1:]) + '\n')

        headers = ['SMI', 'molName'] + extraHeaders

        outputLib = SmallMoleculesLibrary(libraryFilename=smiOut, headers=headers)
        outputLib.calculateLength()
        self._defineOutputs(outputLibrary=outputLib)


    # --------------------------- INFO functions -----------------------------------
    def _validate(self):
        errors = []

        if self.numMols.get() <= 0:
            errors.append("Number of molecules to generate must be greater than 0.")

        molGenerator = self.molGenerator.get()
        if molGenerator == 1 and self.smiFileLib.get() is None:
            errors.append("Scaffold SMILES set must be added.")

        elif molGenerator == 2:
            setA, setB = self.smiFileLinkA.get(), self.smiFileLinkB.get()
            if setA is None or setB is None:
                errors.append("Both warhead sets must be added.")
            elif get_input_length(setA) != get_input_length(setB):
                errors.append("Warhead set 1 and Warhead set 2 must have the same number of molecules.")

        elif molGenerator == 3 and self.smiFileMol.get() is None:
            errors.append("Compound SMILES set must be added.")

        return errors


    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []
        summary.append(f"Generator type: {self.getEnumText('molGenerator')}")
        summary.append(f"Molecules generated: {self.numMols.get()} (per input SMILES)")

        return summary
