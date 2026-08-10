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

from pyworkflow.tests import BaseTest, setupTestProject, DataSet

from pwchem.protocols import ProtChemImportSmallMolecules
from pwchem.utils import assertHandle

from reinvent.protocols import ReinventSampling, ReinventTransferLearning, ReinventStagedLearning


class TestReinventSampling(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testSampling(self):
        print('\nSampling molecules from the default Reinvent prior')
        protSampling = self.newProtocol(
            ReinventSampling,
            molGenerator=0,   # Reinvent (de novo)
            numMols=16)
        self.proj.launchProtocol(protSampling, wait=False)
        self._waitOutput(protSampling, 'outputLibrary', sleepTime=10)

        outLib = getattr(protSampling, 'outputLibrary', None)
        assertHandle(self.assertIsNotNone, outLib, cwd=protSampling.getWorkingDir())
        # The generated library must contain at least one molecule
        assertHandle(self.assertGreater, outLib.getLength(), 0,
                     cwd=protSampling.getWorkingDir())


class TestReinventTransferLearning(BaseTest):
    @classmethod
    def setUpClass(cls):
        cls.dsLig = DataSet.getDataSet('smallMolecules')
        setupTestProject(cls)
        cls._runImportSmallMols()
        cls._waitOutput(cls.protImportSmallMols, 'outputSmallMolecules', sleepTime=5)

    @classmethod
    def _runImportSmallMols(cls):
        cls.protImportSmallMols = cls.newProtocol(
            ProtChemImportSmallMolecules,
            filesPath=cls.dsLig.getFile('mol2'),
            filesPattern='*.mol2')
        cls.proj.launchProtocol(cls.protImportSmallMols, wait=False)

    def testTransferLearning(self):
        print('\nTransfer learning: adapting the Reinvent prior to the input molecules')
        protTL = self.newProtocol(
            ReinventTransferLearning,
            molGenerator=0,   # Reinvent
            smiFileReinvent=self.protImportSmallMols.outputSmallMolecules,
            numEpochs=2, saveChkpt=2, batchSize=4)
        self.proj.launchProtocol(protTL, wait=False)
        self._waitOutput(protTL, 'TL_TrainedModel', sleepTime=15)

        model = getattr(protTL, 'TL_TrainedModel', None)
        assertHandle(self.assertIsNotNone, model, cwd=protTL.getWorkingDir())


class TestReinventStagedLearning(BaseTest):
    # A single stage optimizing QED (already in [0, 1], so no transform needed);
    # this is what the Add-Stage/Add-Component wizards would store in 'workFlowSteps'.
    STAGE = {
        'termination': 'simple',
        'max_score': 0.7,
        'min_steps': 1,
        'max_steps': 2,
        'scoring': {
            'type': 'geometric_mean',
            'component': [{'QED': {'endpoint': [{'name': 'QED', 'weight': 1.0}]}}]
        }
    }

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testStagedLearning(self):
        print('\nStaged learning: one QED-optimizing stage from the Reinvent prior')
        protSL = self.newProtocol(
            ReinventStagedLearning,
            molGenerator=0,        # Reinvent (de novo, no input molecules needed)
            divFilter=False,
            batchSize=8,
            workFlowSteps=str(self.STAGE) + '\n')
        self.proj.launchProtocol(protSL, wait=False)
        self._waitOutput(protSL, 'SL1_TrainedModel', sleepTime=15)

        model = getattr(protSL, 'SL1_TrainedModel', None)
        assertHandle(self.assertIsNotNone, model, cwd=protSL.getWorkingDir())
