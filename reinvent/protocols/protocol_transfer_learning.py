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
This protocol uses REINVENT4's transfer learning to adapt a prior model using an input set of molecules.
"""
import os

import toml
import random
from pwem.protocols import EMProtocol
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import (IntParam, EnumParam, PathParam, FloatParam, BooleanParam, PointerParam)
from rdkit import Chem

from reinvent import Plugin
from reinvent.objects import ReinventModel
from reinvent.utils.smilesUtils import preprocess_smi_file

MOL2MOL = 'molGenerator==1'

class ReinventTransferLearning(EMProtocol):
    """
    Protocol to specialize a general prior by training it into a specific set of input SMILES.
    """
    _label = 'Transfer Learning'

   # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used."""
        form.addSection(label='Run Parameters')
        priorGroup = form.addGroup('Prior Model')
        priorGroup.addParam('molGenerator', EnumParam, choices=['Reinvent', 'Mol2Mol', 'LibInvent', 'LinkInvent'],
                      default=0,
                      label='Type of Molecule Generator',
                      help='Select generative strategy. Each type requires a specific prior and input data.\n'
                           '- Reinvent: De novo sampling. Learns molecule style from input dataset.\n'
                           '- Mol2Mol: Find molecules similar to provided SMILES.\n'
                           '- LibInvent: Find R-groups for the given scaffolds.\n'
                           '- LinkInvent: Find a scaffold to link two fragments.\n'
                           'Transfer learning is less effective for Lib/LinkInvent as they are already constrained by fixed scaffolds/warheads.'
                           'Basic prior is usually sufficient for sampling.')

        priorGroup.addParam('extPrior', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Upload external prior file?',
                      help='Set to True to select a custom prior model.')

        priorGroup.addParam('priorModel', PathParam,
                      condition='extPrior==True',
                      label='External prior model file',
                      help='Select prior model file. Each generator requires a specific prior.')

        priorGroup.addParam('smiFileReinvent', PointerParam,
                            pointerClass='SetOfSmallMolecules',
                            condition='molGenerator==0',
                            label='SMILES file',
                            help='Only reads first column. One compound per line.')

        priorGroup.addParam('smiFileMol', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition=MOL2MOL,
                      label='SMILES file',
                      help='Only reads first column.One compound per line.')

        priorGroup.addParam('smiFileLib', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='molGenerator==2',
                      label='SMILES file',
                      help='Only reads 2 first columns. One scaffold per line.\n'
                           'Column 1: Scaffold with attachment point(s).\n'
                           'Column 2: R-group separated by "." (if multiple). \n'
                           'Example:\n'
                           '[*:0]Cc2ccc1cncc(C[*:1])c1c2    CCC.[H]')

        priorGroup.addParam('smiFileLink', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='molGenerator==3',
                      label='SMILES file',
                      help='Only reads 2 frist columns. One warhead pair per line.\n'
                           'Column 1: Two warheads with attachment point(s) separated by "." (if multiple).\n'
                           'Column 2: Linker connecting them.\n'
                           'Example:\n'
                           'Oc1cncc(*)c1.*c1ccoc1   CC(C)CC')

        priorGroup.addParam('validation', BooleanParam, default=False,
                      label='Use validation set?',
                      help='If TRUE, a portion of the input SMILES will be used for validation')

        priorGroup.addParam('validationSize', FloatParam, default=0.2, expertLevel=LEVEL_ADVANCED,
                      condition='validation==True',
                      label='Validation fraction',
                      help='Proportion of the data to use for validation (e.g., 0.2 for 20%)')

        priorGroup.addParam('pairsType', EnumParam, choices=['tanimoto'], expertLevel=LEVEL_ADVANCED,
                      condition=MOL2MOL,
                      default=0,
                      label='Similarity type',
                      help='Choose metric to calculate similarity between molecular pairs.')

        priorGroup.addParam('pairsUpper', FloatParam, default=1.0, expertLevel=LEVEL_ADVANCED,
                      condition=MOL2MOL,
                      label='Upper similarity threshold',
                      help='Maximum similarity score allowed for a molecular pair to be considered valid.')

        priorGroup.addParam('pairsLower', FloatParam, default=0.7, expertLevel=LEVEL_ADVANCED,
                      condition=MOL2MOL,
                      label='Lower similarity threshold',
                      help='Minimum similarity score required to ensure generated molecules remain related to reference')

        priorGroup.addParam('pairsMinCard', IntParam, default=1, expertLevel=LEVEL_ADVANCED,
                      condition=MOL2MOL,
                      label='Minimum cardinality',
                      help='Minimum number of similar neighbors required for a molecule to be included in train set.')

        priorGroup.addParam('pairsMaxCard', IntParam, default=199, expertLevel=LEVEL_ADVANCED,
                      condition=MOL2MOL,
                      label='Maximum cardinality',
                      help='Maximum number of compounds that can be compared with a certain one.')

        runGroup = form.addGroup('Run Parameters')
        runGroup.addParam('numEpochs', IntParam, default=10,
                      label='Number of epochs',
                      help='Number of steps to train the prior model')

        runGroup.addParam('saveChkpt', IntParam, default=5,
                      label='Save checkpoint',
                      help='Save checkpoint model file every N epochs')

        runGroup.addParam('batchSize', IntParam, default=128, expertLevel=LEVEL_ADVANCED,
                      label='Batch size',
                      help='Number of training molecules processed in each epoch')

        runGroup.addParam('numRefs', IntParam, default=0, expertLevel=LEVEL_ADVANCED,
                      label='Number of reference molecules',
                      help='Number of reference molecules randomly chosen to calculate similarity in each epoch \n'
                            'Value = 0 (Recommended): Uses the entire dataset')

        runGroup.addParam('sampleBatchSize', IntParam, default=100, expertLevel=LEVEL_ADVANCED,
                      label='Sample batch size',
                      help='Number of generated molecules to compute sample loss')

    # -------------------------- OTHER functions ----------------------
    def _getPriorFile(self):
        priorModel = ['reinvent.prior','mol2mol_scaffold_generic.prior','libinvent.prior','linkinvent.prior']
        priorFile = priorModel[self.molGenerator.get()]
        return Plugin.getPriorPath(priorFile)

    def _getSmilesSet(self):
        molGenerator = self.molGenerator.get()
        if molGenerator == 0:
            return self.smiFileReinvent.get()
        elif molGenerator == 1:
            return self.smiFileMol.get()
        elif molGenerator == 2:
            return self.smiFileLib.get()
        elif molGenerator == 3:
            return self.smiFileLink.get()

    def _extractSmilesToFile(self, molSet, outputFilename):
        outputPath = self._getPath(outputFilename)

        with open(outputPath, 'w') as fout:
            for mol in molSet:
                molFile = mol.getFileName()

                if molFile.endswith('.smi'):
                    with open(molFile, 'r') as fin:
                        smiles = fin.readline().split()[0]
                        fout.write(smiles + '\n')

                elif molFile.endswith('.sdf'):
                    supplier = Chem.SDMolSupplier(molFile)
                    for rdmol in supplier:
                        if rdmol:
                            fout.write(Chem.MolToSmiles(rdmol) + '\n')

                elif molFile.endswith('.mol2'):
                    rdmol = Chem.MolFromMol2File(molFile)
                    if rdmol:
                        fout.write(Chem.MolToSmiles(rdmol) + '\n')

        return outputPath

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.splitSmilesStep)
        self._insertFunctionStep(self.createConfigFileStep)
        self._insertFunctionStep(self.runReinventStep)
        self._insertFunctionStep(self.createOutputStep)

    def splitSmilesStep(self):
        molSet = self._getSmilesSet()
        rawSmiPath = self._extractSmilesToFile(molSet, 'smiles_raw.txt')
        cleanInput = preprocess_smi_file(self, rawSmiPath, 'smiles_cleaned.txt')

        if self.validation.get():
            trainPath = self._getPath('train_split.smi')
            valPath = self._getPath('val_split.smi')

            with open(cleanInput, 'r') as f:
                lines = f.readlines()

            random.shuffle(lines)
            splitIdx = int(len(lines) * (1 - self.validationSize.get()))

            with open(trainPath, 'w') as f:
                f.writelines(lines[:splitIdx])
            with open(valPath, 'w') as f:
                f.writelines(lines[splitIdx:])

            self.info("SMILES split: %d training, %d validation" % (splitIdx, len(lines) - splitIdx))
        else:
            self.info('Using all SMILES for training (no validation).')



    def createConfigFileStep(self):

        filename = f"TL_{self.getEnumText('molGenerator')}.model"

        if self.extPrior.get():
            priorPath = self.priorModel.get()
        else:
            priorPath = self._getPriorFile()

        params = {
            'num_epochs': self.numEpochs.get(),
            'save_every_n_epochs': self.saveChkpt.get(),
            'batch_size': self.batchSize.get(),
            'num_refs': self.numRefs.get(),
            'sample_batch_size': self.sampleBatchSize.get(),
            'input_model_file': priorPath,
            'output_model_file': self._getPath(filename)
        }
        if self.validation.get():
            params['validation_smiles_file'] = self._getPath('val_split.smi')
            params['smiles_file'] = self._getPath('train_split.smi')
        else:
            params['smiles_file'] = self._getPath('smiles_cleaned.txt')

        if self.molGenerator.get() == 1:
            pairsParams = {
                'type': self.getEnumText('pairsType'),
                'upper_threshold': self.pairsUpper.get(),
                'lower_threshold': self.pairsLower.get(),
                'min_cardinality': self.pairsMinCard.get(),
                'max_cardinality': self.pairsMaxCard.get()
            }
            params['pairs'] = pairsParams

        configParams = {
            'run_type': 'transfer_learning',
            'device': 'cpu',
            'tb_logdir': os.path.join(self._getExtraPath(), 'TB_logs'),
            'json_out_config': self._getTmpPath('_transfer_learning.json'),
            'parameters': params
            }

        self.configPath = self._getPath('TL_config.toml')
        with open(self.configPath, 'w') as f:
            toml.dump(configParams, f)
        self.info("TOML config file created succesfully")

    def runReinventStep(self):
        condaInit = Plugin.getCondaActivationCmd()
        envActivation = Plugin.getEnvActivation()
        executable = Plugin.getProgram()

        parts = [condaInit, f'{envActivation} &&', executable]
        fullCommand = " ".join(parts)
        self.runJob(fullCommand, self.configPath,
                    env=Plugin.getEnviron())
        self.info("REINVENT execution completed.")


    def createOutputStep(self):

        filename = f"TL_{self.getEnumText('molGenerator')}.model"
        pathModel = self._getPath(filename)
        trainedModel = ReinventModel(path=pathModel)
        trainedModel.setObjLabel((f"Transfer learning model ({self.getEnumText('molGenerator')})"))
        self._defineOutputs(TL_TrainedModel=trainedModel)

        numEpochs = self.numEpochs.get()
        saveEvery = self.saveChkpt.get()
        chkptEpochs = range(saveEvery, numEpochs + 1, saveEvery)

        outputs = {}
        for epoch in chkptEpochs:
            chkptPath = self._getPath(f"{filename}.{epoch}.chkpt")
            if os.path.exists(chkptPath):
                chkptModel = ReinventModel(path=chkptPath)
                chkptModel.setObjLabel(f"Transfer learning chkpt {epoch} ({self.getEnumText('molGenerator')})")
                outputs[f"TL_chkpt_{epoch}"] = chkptModel

        if outputs:
            self._defineOutputs(**outputs)
        self.info("Outputs created succesfully")

# --------------------------- INFO functions -----------------------------------
    def _validate(self):
        errors = []

        if self.extPrior.get() is True and self.priorModel.get() is None:
            errors.append("External prior file must be added")

        smilesfile = self._getSmilesSet()
        if smilesfile is None:
            errors.append("SMILES file must be added")

        if self.validation.get() is True and not (0.1 <= self.validationSize.get() <= 0.9):
            errors.append("Validation fraction must be in range [0.1,0.9]")

        if not (0 <= self.pairsUpper.get() <= 1) or not (0 <= self.pairsLower.get() <= 1):
            errors.append("Similarity thresholds must be in range [0,1]")

        if self.pairsLower.get() > self.pairsUpper.get():
            errors.append("Lower similarity threshold must be smaller than upper similarity threshold")

        if self.pairsMinCard.get() > self.pairsMaxCard.get():
            errors.append("Minimum cardinality must be smaller than maximum cardinality")

        if self.numEpochs.get() < 1:
            errors.append("Number of epochs should be minimum 1.")

        if self.saveChkpt.get() < 1:
            errors.append("Save chekpoint should be minimum 1.")

        if self.saveChkpt.get() > self.numEpochs.get():
            errors.append("Save checkpoint must be smaller than number of epochs.")

        return errors




    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []
        summary.append(f"Generator type: {self.getEnumText('molGenerator')}")

        return summary
