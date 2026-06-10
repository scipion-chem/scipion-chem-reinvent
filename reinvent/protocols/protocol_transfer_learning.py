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



class ReinventTransferLearning(EMProtocol):
    """
    Protocol to specialize a general prior by training it into a specific set of input SMILES.
    """
    _label = 'Transfer Learning'

   # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used."""
        form.addSection(label='Run Parameters')
        PriorGroup = form.addGroup('Prior Model')
        PriorGroup.addParam('MolGenerator', EnumParam, choices=['Reinvent'],# 'Mol2Mol',
                                                          #'LibInvent', 'LinkInvent'],
                      default=0,
                      label='Type of Molecule Generator',
                      help='Select generative strategy. Each type requires a specific prior and input data.\n'
                           '- Reinvent: De novo sampling. Learns molecule style from input dataset.\n'
                           '- Mol2Mol: Find molecules similar to provided SMILES.\n'
                           '- LibInvent: Find R-groups for the given scaffolds.\n'
                           '- LinkInvent: Find a scaffold to link two fragments.\n'
                           'Transfer learning is less effective for Lib/LinkInvent as they are already constrained by fixed scaffolds/warheads.'
                           'Basic prior is usually sufficient for sampling.')

        PriorGroup.addParam('ExtPrior', BooleanParam, default='False', expertLevel=LEVEL_ADVANCED,
                      label='Upload external prior file?',
                      help='Set to True to select a custom prior model.')

        PriorGroup.addParam('PriorModel', PathParam,
                      condition='ExtPrior==True',
                      label='External prior model file',
                      help='Select prior model file. Each generator requires a specific prior.')

        PriorGroup.addParam('SmiFileReinvent', PointerParam,
                            pointerClass='SetOfSmallMolecules',
                            condition='MolGenerator==0',
                            label='SMILES file',
                            help='Only reads first column. One compound per line.')

        PriorGroup.addParam('SmiFileMol', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='MolGenerator==1',
                      label='SMILES file',
                      help='Only reads first column.One compound per line.')

        PriorGroup.addParam('SmiFileLib', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='MolGenerator==2',
                      label='SMILES file',
                      help='Only reads 2 first columns. One scaffold per line.\n'
                           'Column 1: Scaffold with attachment point(s).\n'
                           'Column 2: R-group separated by "." (if multiple). \n'
                           'Example:\n'
                           '[*:0]Cc2ccc1cncc(C[*:1])c1c2    CCC.[H]')

        PriorGroup.addParam('SmiFileLink', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='MolGenerator==3',
                      label='SMILES file',
                      help='Only reads 2 frist columns. One warhead pair per line.\n'
                           'Column 1: Two warheads with attachment point(s) separated by "." (if multiple).\n'
                           'Column 2: Linker connecting them.\n'
                           'Example:\n'
                           'Oc1cncc(*)c1.*c1ccoc1   CC(C)CC')

        PriorGroup.addParam('Validation', BooleanParam, default=False,
                      label='Use validation set?',
                      help='If TRUE, a portion of the input SMILES will be used for validation')

        PriorGroup.addParam('ValidationSize', FloatParam, default=0.2, expertLevel=LEVEL_ADVANCED,
                      condition='Validation==True',
                      label='Validation fraction',
                      help='Proportion of the data to use for validation (e.g., 0.2 for 20%)')

        PriorGroup.addParam('PairsType', EnumParam, choices=['tanimoto'], expertLevel=LEVEL_ADVANCED,
                      condition='MolGenerator==1',
                      default=0,
                      label='Similarity type',
                      help='Choose metric to calculate similarity between molecular pairs.')

        PriorGroup.addParam('PairsUpper', FloatParam, default=1.0, expertLevel=LEVEL_ADVANCED,
                      condition='MolGenerator==1',
                      label='Upper similarity threshold',
                      help='Maximum similarity score allowed for a molecular pair to be considered valid.')

        PriorGroup.addParam('PairsLower', FloatParam, default=0.7, expertLevel=LEVEL_ADVANCED,
                      condition='MolGenerator==1',
                      label='Lower similarity threshold',
                      help='Minimum similarity score required to ensure generated molecules remain related to reference')

        PriorGroup.addParam('PairsMinCard', IntParam, default=1, expertLevel=LEVEL_ADVANCED,
                      condition='MolGenerator==1',
                      label='Minimum cardinality',
                      help='Minimum number of similar neighbors required for a molecule to be included in train set.')

        PriorGroup.addParam('PairsMaxCard', IntParam, default=199, expertLevel=LEVEL_ADVANCED,
                      condition='MolGenerator==1',
                      label='Maximum cardinality',
                      help='Maximum number of compounds that can be compared with a certain one.')

        RunGroup = form.addGroup('Run Parameters')
        RunGroup.addParam('NumEpochs', IntParam, default=10,
                      label='Number of epochs',
                      help='Number of steps to train the prior model')

        RunGroup.addParam('SaveChkpt', IntParam, default=5,
                      label='Save checkpoint',
                      help='Save checkpoint model file every N epochs')

        RunGroup.addParam('BatchSize', IntParam, default=128, expertLevel=LEVEL_ADVANCED,
                      label='Batch size',
                      help='Number of training molecules processed in each epoch')

        RunGroup.addParam('NumRefs', IntParam, default=0, expertLevel=LEVEL_ADVANCED,
                      label='Number of reference molecules',
                      help='Number of reference molecules randomly chosen to calculate similarity in each epoch \n'
                            'Value = 0 (Recommended): Uses the entire dataset')

        RunGroup.addParam('SampleBatchSize', IntParam, default=100, expertLevel=LEVEL_ADVANCED,
                      label='Sample batch size',
                      help='Number of generated molecules to compute sample loss')

    # -------------------------- OTHER functions ----------------------
    def _getPriorFile(self):
        priorModel = ['reinvent.prior','mol2mol_scaffold_generic.prior','libinvent.prior','linkinvent.prior']
        priorFile = priorModel[self.MolGenerator.get()]
        return Plugin.getPriorPath(priorFile)

    def _getSmilesSet(self):
        MolGenerator = self.MolGenerator.get()
        if MolGenerator == 0:
            return self.SmiFileReinvent.get()
        elif MolGenerator == 1:
            return self.SmiFileMol.get()
        elif MolGenerator == 2:
            return self.SmiFileLib.get()
        elif MolGenerator == 3:
            return self.SmiFileLink.get()

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
        self._insertFunctionStep('splitSmilesStep')
        self._insertFunctionStep('createConfigFileStep')
        self._insertFunctionStep('runReinventStep')
        self._insertFunctionStep('createOutputStep')

    def splitSmilesStep(self):
        molSet = self._getSmilesSet()
        raw_smi_path = self._extractSmilesToFile(molSet, 'smiles_raw.txt')
        clean_input = preprocess_smi_file(self, raw_smi_path, 'smiles_cleaned.txt')

        if self.Validation.get():
            train_path = self._getPath('train_split.smi')
            val_path = self._getPath('val_split.smi')

            with open(clean_input, 'r') as f:
                lines=f.readlines()

            random.shuffle(lines)
            split_idx=int(len(lines)*(1-self.ValidationSize.get()))

            with open(train_path, 'w') as f:
                f.writelines(lines[:split_idx])
            with open(val_path, 'w') as f:
                f.writelines(lines[split_idx:])

            self.info("SMILES split: %d training, %d validation" % (split_idx, len(lines)-split_idx))
        else:
            self.info('Using all SMILES for training (no validation).')



    def createConfigFileStep(self):

        filename = f"TL_{self.getEnumText('MolGenerator')}.model"

        if self.ExtPrior.get():
            prior_path = self.PriorModel.get()
        else:
            prior_path = self._getPriorFile()

        params = {
            'num_epochs': self.NumEpochs.get(),
            'save_every_n_epochs': self.SaveChkpt.get(),
            'batch_size': self.BatchSize.get(),
            'num_refs': self.NumRefs.get(),
            'sample_batch_size': self.SampleBatchSize.get(),
            'input_model_file': prior_path,
            'output_model_file': self.getPath(filename)
        }
        if self.Validation.get():
            params['validation_smiles_file'] = self._getPath('val_split.smi')
            params['smiles_file'] = self._getPath('train_split.smi')
        else:
            params['smiles_file'] = self._getPath('smiles_cleaned.txt')

        if self.MolGenerator.get() == 1:
            pairs_params = {
                'type': self.getEnumText('PairsType'),
                'upper_threshold': self.PairsUpper.get(),
                'lower_threshold': self.PairsLower.get(),
                'min_cardinality': self.PairsMinCard.get(),
                'max_cardinality': self.PairsMaxCard.get()
            }
            params['pairs'] = pairs_params

        config_params = {
            'run_type': 'transfer_learning',
            'device': 'cpu',
            'json_out_config': self._getTmpPath('_transfer_learning.json'),
            'parameters': params
            }

        self.configPath = self._getPath('TL_config.toml')
        with open(self.configPath, 'w') as f:
            toml.dump(config_params, f)
        self.info("TOML config file created succesfully")

    def runReinventStep(self):
        CondaInit = Plugin.getCondaActivationCmd()
        EnvActivation = Plugin.getEnvActivation()
        Executable = Plugin.getProgram()

        parts = [CondaInit, f'{EnvActivation} &&', Executable]
        full_command = " ".join(parts)
        self.runJob(full_command, self.configPath,
                    env=Plugin.getEnviron())
        self.info("REINVENT execution completed.")


    def createOutputStep(self):

        filename = f"TL_{self.getEnumText('MolGenerator')}.model"
        path_model = self._getPath(filename)
        trained_model = ReinventModel(path=path_model)
        trained_model.setObjLabel((f"Transfer learning model ({self.getEnumText('MolGenerator')})"))
        self._defineOutputs(TL_TrainedModel=trained_model)

        num_epochs = self.NumEpochs.get()
        save_every = self.SaveChkpt.get()
        chkpt_epochs = range (save_every, num_epochs + 1, save_every)

        outputs = {}
        for epoch in chkpt_epochs:
            chkpt_path = self._getPath(f"{filename}.{epoch}.chkpt")
            if os.path.exists(chkpt_path):
                chkpt_model = ReinventModel(path=chkpt_path)
                chkpt_model.setObjLabel(f"Transfer learning chkpt {epoch} ({self.getEnumText('MolGenerator')})")
                outputs[f"TL_chkpt_{epoch}"] = chkpt_model

        if outputs:
            self._defineOutputs(**outputs)
        self.info("Outputs created succesfully")

# --------------------------- INFO functions -----------------------------------
    def _validate(self):
        errors = []

        if self.ExtPrior.get() is True and self.PriorModel.get() is None:
            errors.append("External prior file must be added")

        smilesfile = self._getSmilesSet()
        if smilesfile is None:
            errors.append("SMILES file must be added")

        if self.Validation.get() is True and not (0.1 <= self.ValidationSize.get() <= 0.9):
            errors.append("Validation fraction must be in range [0.1,0.9]")

        if not (0 <= self.PairsUpper.get() <= 1) or not (0 <= self.PairsLower.get() <= 1):
            errors.append("Similarity thresholds must be in range [0,1]")

        if self.PairsLower.get() > self.PairsUpper.get():
            errors.append("Lower similarity threshold must be smaller than upper similarity threshold")

        if self.PairsMinCard.get() > self.PairsMaxCard.get():
            errors.append("Minimum cardinality must be smaller than maximum cardinality")

        if self.NumEpochs.get() < 1:
            errors.append("Number of epochs should be minimum 1.")

        if self.SaveChkpt.get() < 1:
            errors.append("Save chekpoint should be minimum 1.")

        if self.SaveChkpt.get() > self.NumEpochs.get():
            errors.append("Save checkpoint must be smaller than number of epochs.")

        return errors




    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []
        summary.append(f"Generator type: {self.getEnumText('MolGenerator')}")

        return summary