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
This protocol trains a prior model with multiple selection filters, using REINVENT4's staged learning.
Guides generation toward desired molecular profiles.

"""
import os
import shutil

import toml
import ast
from pwem.protocols import EMProtocol
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import (PointerParam, IntParam, BooleanParam, EnumParam,
                                        PathParam, FloatParam, TextParam)

from rdkit import Chem

from reinvent import Plugin
from reinvent.objects import ReinventModel
from reinvent.utils.smilesUtils import preprocess_smi_file



class ReinventStagedLearning(EMProtocol):
    """
    This protocol will incrementally adapt the model to your requirements.
    """
    _label = 'Staged Learning'

    # -------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        """ Define the input parameters that will be used."""
        form.addSection(label='Molecule Generator')
        priorGroup = form.addGroup('Prior Model')
        priorGroup.addParam('molGenerator', EnumParam, choices=['Reinvent', 'LibInvent', 'LinkInvent', 'Mol2Mol'],
                      default=0,
                      label='Type of Molecule Generator',
                      help='Generative strategy to be used. Each generator requires a specific prior and input data.\n'
                           '- Reinvent: De novo sampling.\n'
                           '- LibInvent: Find R-groups for the given scaffolds.\n'
                           '- LinkInvent: Find a scaffold to link two fragments.\n'
                           '- Mol2Mol: Find molecules similar to provided SMILES.')

        priorGroup.addParam('extPrior', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Upload external prior file?',
                      help='Set to True to select a custom prior model.')

        priorGroup.addParam('priorModel', PointerParam,
                      pointerClass='ReinventModel',
                      condition='extPrior==False',
                      allowsNull=True,
                      label='Prior model file',
                      help='Select a trained model. A Transfer Learning protocol should be run first.\n'
                           ' If left empty, the default prior for the selected generator will be used.')

        priorGroup.addParam('extPriorModel', PathParam,
                      condition='extPrior==True',
                      label='Prior model file',
                      help='Path to prior model file. Each generator requires a specific prior.')

        priorGroup.addParam('smiFileLib', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='molGenerator==1',
                      label='Scaffold SMILES file',
                      help='One scaffold per line. Each scaffold must be annotated by 2 \'*\' to locate the attachment points.\n'
                           'Up to 4 attachments points are allowed.\n'
                           'Example:\n [*:0]Cc2ccc1cncc(C[*:1])c1c2')

        priorGroup.addParam('smiFileLink', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='molGenerator==2',
                      label='Warheads SMILES file',
                      help='One warhead pair per line. Each warhead must be annotated with \'*\' to locate the attachment points.'
                           'The two warheads must be separated by the pipe symbol.\n'
                           'Example:\n Oc1cncc(*)c1|*c1ccoc1')

        priorGroup.addParam('smiFileMol', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='molGenerator==3',
                      label='Compound SMILES file',
                      help='One compound per line.')

        priorGroup.addParam('inception', BooleanParam, default=False,
                      condition='molGenerator==0',
                      label='Activate Inception?',
                      help='Guide learning into a list of molecules provided.')

        priorGroup.addParam('inceptSmi', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='molGenerator==0 and inception==True',
                      label='Inception SMILES file',
                      help='One molecule per line.')

        priorGroup.addParam('memSize', IntParam, default=100, expertLevel=LEVEL_ADVANCED,
                      condition='molGenerator==0 and inception==True',
                      label='SMILES held in memory',
                      help='Top N scored molecules. As the learning progresses, the initial molecules are removed '
                           'and replaced by those with higher scores')

        priorGroup.addParam('sampSize', IntParam, default=10, expertLevel=LEVEL_ADVANCED,
                      condition='molGenerator==0 and inception==True',
                      label='SMILES chosen per epoch',
                      help='Number of randomly sampled molecules to be used in computing inception loss.')

        priorGroup.addParam('sampleStrat', EnumParam, choices=['Beamsearch', 'Multinomial'],
                      condition='molGenerator==3', expertLevel=LEVEL_ADVANCED,
                      default=0,
                      label='Sampling Strategy',
                      help='Multinomial: Fast random generation based on token probability distribution.\n'
                           'Beamsearch: Deterministic approach that ensures unique compounds '
                           'by selecting the highest probability sequences.')

        priorGroup.addParam('temperature', FloatParam, default=1.0, expertLevel=LEVEL_ADVANCED,
                      condition='molGenerator==3 and sampleStrat==1',
                      label='Temperature in Multinomial sampling',
                      help='Controls randomness. Lower values for more predictable molecules and higher values for more diverse structures.')

        priorGroup.addParam('disThres', IntParam, default=100,
                      condition='molGenerator==3', expertLevel=LEVEL_ADVANCED,
                      label='Distance Threshold',
                      help='Maximum limit on how much the generated molecule can structurally deviate from the source. '
                           'Higher values increase diversity.')

        runGroup = form.addGroup('Run Mode')

        runGroup.addParam('useCkpt', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Use checkpoint?',
                      help='If TRUE use diversity filter from agent file.')

        runGroup.addParam('purgeMem', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Purge Memories?',
                      help='Controls if memory is cleared between training stages.'
                           'If FALSE, the model maintains memory across all stages '
                           'ensuring it does not repeat previously found structures.')

        runGroup.addParam('batchSize', IntParam, default=128, expertLevel=LEVEL_ADVANCED,
                      label='Batch size',
                      help='Number of molecules generated per run (epoch).')

        runGroup.addParam('randomSmi', BooleanParam, default=True,
                      label='Shuffle atoms randomly?',
                      help='If TRUE shuffle atoms in SMILES randomly.')

        learnGroup = form.addGroup('Learning Strategy', expertLevel=LEVEL_ADVANCED,)
        learnGroup.addParam('learnType', EnumParam, choices=['dap'],
                      default=0, expertLevel=LEVEL_ADVANCED,
                      label='Type of Learning Strategy',
                      help='DAP recommended.\n'
                           'Use of default values is also recommended. \n'
                           'If the learning is too slow, you can increase the learning rate.')

        learnGroup.addParam('sigma', IntParam, default=128, expertLevel=LEVEL_ADVANCED,
                      label='Sigma of the reward function',
                      help='Determines closeness between Agent and Prior. Higher values push the model to convert'
                           ' faster toward objective.')

        learnGroup.addParam('learningRate', FloatParam, default=0.0001, expertLevel=LEVEL_ADVANCED,
                      label='Learning rate',
                      help="Controls how much the model's parameters change in response to estimated error.")

        divGroup = form.addGroup('Diversity Filter')
        divGroup.addParam('divFilter', BooleanParam, default=True,
                      label='Activate Diversity Filter?',
                      help='If TRUE activates memory of agent file that remembers good molecules '
                           'and penalizes the model if it repeats them. Forces to find different structures.')


        divGroup.addParam('divType', EnumParam, choices=['IdenticalMurckoScaffold', 'IdenticalTopologicalScaffold',
                                                     'ScaffoldSimilarity', 'PenalizeSameSmiles'],
                      condition='divFilter==True',
                      default=0,
                      label='Diversity Filter type',
                      help='How to group similar molecules.\n'
                            '- IdenticalMurckoScaffold: Groups molecules by their Murcko scaffold (rings and linking chains). '
                                                        'Most common for structural diversity.\n'
                            '- IdenticalTopologicalScaffold: Similar to Murcko but considers atom types and bond orders as generic.\n'
                            '- ScaffoldSimilarity: Penalizes molecules with similar scaffolds.\n'
                            '- PenalizeSameSmiles: Only penalizes exact molecular matches.')

        divGroup.addParam('bucketSize', IntParam, default=25, expertLevel=LEVEL_ADVANCED,
                      condition='divFilter==True',
                      label='Bucket size',
                      help='Number of compounds per bucket. Each bucket holds the same scaffold.')

        divGroup.addParam('minScore', FloatParam, default=0.4, expertLevel=LEVEL_ADVANCED,
                      condition='divFilter==True',
                      label='Minimum score',
                      help='Memorize those compounds that have a score value equal or higher to this minimum value.')

        divGroup.addParam('minSim', FloatParam, default=0.4, expertLevel=LEVEL_ADVANCED,
                      condition='divFilter==True and divType==2',
                      label='Minimum similarity',
                      help='Sets similarity threshold above which a new scaffold is considered redundant compared to stored ones.')

        divGroup.addParam('penalty', FloatParam, default=0.5, expertLevel=LEVEL_ADVANCED,
                      condition='divFilter==True and divType==3',
                      label='Penalty factor',
                      help='Penalty applied to score of repeated molecule.')

        form.addSection(label='Stage Parameters')
        stageGroup = form.addGroup('Stage Parameters')

        stageGroup.addParam('maxScore', FloatParam, default=0.7,
                      label='Maximum score',
                      help='Success score. The stage ends when the generated molecules reach this quality.')

        stageGroup.addParam('minSteps', IntParam, default=10,
                      label='Minimum number of steps',
                      help='Minimum number of epochs to run before checking termination criteria.')

        stageGroup.addParam('maxSteps', IntParam, default=100,
                      label='Maximum number of steps',
                      help='Maximum number of steps allowed. When reached, the run terminates')

        stageGroup.addParam('scoreFunct', EnumParam, choices=['geometric_mean', 'arithmetic_mean'],
                      default=0,
                      label='Score function type',
                      help='Aggregation function used to combine all component scores into a single one.\n'
                           'Use geometric for strict multi-parameter balance or arithmetic for flexible average performance.')

        stageGroup.addParam('insertStep', IntParam,
                          default=1,
                          label='Insert step index',
                          help='Index where to insert the new stage (wizard).')

        component_choices = [
            'SlogP', 'MolecularWeight', 'TPSA', 'GraphLength',
            'NumAtomStereoCenters', 'HBondAcceptors', 'HBondDonors',
            'NumRotBond', 'Csp3', 'numsp', 'numsp2', 'numsp3',
            'NumHeavyAtoms', 'NumHeteroAtoms', 'NumRings', 'NumAromaticRings',
            'NumAliphaticRings', 'PMI', 'MolVolume', 'QED', 'SASCore',
            'FragmentMolecularWeight', 'FragmentGraphLength', 'FragmentHBondDonors',
            'FragmentHBondAcceptors', 'FragmentNumRotBond', 'Fragmentnumsp',
            'Fragmentnumsp2', 'Fragmentnumsp3', 'FragmentNumRings',
            'FragmentNumAromaticRings', 'FragmentNumAliphaticRings'
        ]
        comp_help = """Select a molecular descriptor to guide the drug design process.

                            --- PHYSICOCHEMICAL ---
                            * SlogP: Measures lipophilicity. Logarithm of the octanol-water partition coefficient.
                            * MolecularWeight: Total mass of the molecule in Daltons.
                            * TPSA: Topological Polar Surface Area.
                            * MolVolume: Molecular volume occupied in space.

                            --- DRUG-LIKENESS ---
                            * QED: Quantitative Quantitative Estimate of Drug-likeness (0 to 1).
                            * SASCore: Synthetic Accessibility Score. Estimates the ease of synthesis in a lab (1 = very easy, 10 = very difficult).

                            --- ATOMIC STRUCTURE & CONNECTIVITY ---
                            * GraphLength: Measures the diameter of the molecular graph (longest distance between two atoms).
                            * NumAtomStereoCenters: Number of chiral centers.
                            * NumHeavyAtoms: Count of all atoms excluding Hydrogen.
                            * NumHeteroAtoms: Count of atoms other than Carbon and Hydrogen.

                            --- DONORS & ACCEPTORS ---
                            * HBondAcceptors: Number of atoms capable of accepting a hydrogen bond.
                            * HBondDonors: Number of atoms capable of donating a hydrogen bond.

                            --- FLEXIBILITY & HYBRIDIZATION ---
                             * NumRotBond: Flexibility. Number of single rotatable bonds.
                             * Csp3: Fraction of sp3 carbons. Higher values indicate a more 3D structure.
                             * numsp/numsp2/numsp3: Individual counts of atoms with sp (linear), sp2 (planar), or sp3 (tetrahedral) hybridization.

                             --- RINGS ---
                             * NumRings: Total count of rings in the molecular structure.
                             * NumAromaticRings: Total number of aromatic rings.
                             * NumAliphaticRings: Number of non-aromatic rings.

                             --- SHAPE DESCRIPTORS (3D) ---
                             * PMI: Principal Moments of Inertia. Used to characterize the overall shape of the molecule (spherical, rod-like, or disc-like).

                             --- FRAGMENT-BASED PROPERTIES ---
                             (These components calculate the same metrics as above but restricted only to a specific sub-structure or fragment of the molecule)
                             * FragmentMolecularWeight: Mass of the analyzed fragment.
                             * FragmentGraphLength: Longest path within the fragment.
                             * FragmentHBondDonors / Acceptors: Hydrogen bonding capacity localized to the fragment.
                             * FragmentNumRotBond: Flexibility localized within the fragment.
                             * Fragmentnumsp / sp2 / sp3: Hybridization of the atoms within the fragment.
                             * FragmentNumRings / Aromatic / Aliphatic: Ring counts within the fragment sub-structure.

                            """
        compGroup = form.addGroup('Scoring component')
        compGroup.addParam('compType', EnumParam, choices=component_choices,
                       default=0,
                       label='Select scoring component',
                       help=comp_help)

        compGroup.addParam('weight', FloatParam, default=1.0,
                       label='Weight',
                       help='Relative importance of this component in the final score.')

        compGroup.addParam('trans', BooleanParam, default=True,
                       label='Apply transformer function?',
                       help='Enable this to normalize the raw chemical value into a score between 0 and 1.')

        compGroup.addParam('transFunc', EnumParam, choices=['Sigmoid', 'Reverse_Sigmoid', 'Double_Sigmoid',
                                                               'Right_Step', 'Left_Step', 'Step', 'value_mapping'],
                        default=0,
                        condition='trans',
                        label='Transformer type',
                        help="""Select the mathematical function used to map raw values to scores:
                    Select SIGMOID transformers for smooth optimization and STEP functions for rigid threshold.

                    - Sigmoid: A continuous S-curve. Use when 'higher is better'.
                    - Reverse Sigmoid: A decreasing S-curve. Use when 'lower is better'
                    - Double Sigmoid: Best for keeping values within a specific range with smooth penalization in edges.
                    - Right Step: Values above the threshold get a score of 1.0, everything below gets 0.0.
                    - Left Step: Values below the threshold get a score of 1.0, everything above gets 0.0.
                    - Step: Only values between the thresholds receive a score of 1.0.
                    - Value Mapping: Assigns discrete scores to specific categories. (Only recommended for MMP).""")

        compGroup.addParam('low', FloatParam, default=50.0,
                           condition='trans and transFunc in [0,1,2,4,5]',
                           label='Lower threshold',)

        compGroup.addParam('up', FloatParam, default=100.0,
                           condition='trans and transFunc in [0,1,2,3,5]',
                           label='Upper threshold')

        compGroup.addParam('scoreMatch', FloatParam, default=100.0,
                           condition='trans and transFunc == 6',
                           label='Score if matches')

        compGroup.addParam('scoreNoMatch', FloatParam, default=50.0,
                           condition='trans and transFunc == 6',
                           label='Score if no match')

        compGroup.addParam('insertComponent', IntParam,
                          default=1,
                          label='Insert component in step index',
                          help='Index of the stage where to insert the new component (wizard).')


        sumGroup = form.addGroup('Summary')
        sumGroup.addParam('workFlowSteps', TextParam, condition=False,
                               default='',
                               label='Workflow steps',
                               help='Internal workflow steps (used by wizard).')

        sumGroup.addParam('summarySteps', TextParam,
                               default='',
                               label='Summary steps',
                               help='Protocol summary steps.')

        sumGroup.addParam('delStage', IntParam,
                          default='0', allowsNull=True,
                          label='Delete stage',
                          help='Index of the stage to delete')

        sumGroup.addParam('delComponent', IntParam,
                          default='0', allowsNull=True,
                          label='Delete component',
                          help='Index of the component to delete. The stage index is also needed.')

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

    def _getSmilesSet(self):
        molGenerator = self.molGenerator.get()
        if molGenerator == 1:
            return self.smiFileLib.get()
        elif molGenerator == 2:
            return self.smiFileLink.get()
        elif molGenerator == 3:
            return self.smiFileMol.get()
        else:
            return None

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
                    for rdmol in Chem.SDMolSupplier(molFile):
                        if rdmol:
                            fout.write(Chem.MolToSmiles(rdmol) + '\n')
                elif molFile.endswith('.mol2'):
                    rdmol = Chem.MolFromMol2File(molFile)
                    if rdmol:
                        fout.write(Chem.MolToSmiles(rdmol) + '\n')
        return outputPath

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.createConfigFileStep)
        self._insertFunctionStep(self.runReinventStep)
        self._insertFunctionStep(self.createOutputStep)

    def createConfigFileStep(self):

        priorPath = self._getPriorFile()

        params = {
            'summary_csv_prefix': 'staged_learning',
            'use_checkpoint': self.useCkpt.get(),
            'purge_memories': self.purgeMem.get(),
            'batch_size': self.batchSize.get(),
            'randomize_smiles': self.randomSmi.get(),
            'prior_file': priorPath,
            'agent_file': priorPath,
        }

        molSet = self._getSmilesSet()
        if molSet is not None:
            rawSmiPath = self._extractSmilesToFile(molSet, 'smiles_raw.smi')
            newSmilesFile = preprocess_smi_file(self, rawSmiPath, 'smiles_cleaned.smi')
            params['smiles_file'] = newSmilesFile


        sampleStrat = self.sampleStrat.get()
        if self.molGenerator.get() == 3:
            if sampleStrat == 0:
                params['sample_strategy'] = 'beamsearch'
            if sampleStrat == 1:
                params['sample_strategy'] = 'multinomial'
                params['temperature'] = self.temperature.get()
            params['distance_threshold'] = self.disThres.get()

        configParams = {
            'run_type': 'staged_learning',
            'device': 'cpu',
            'json_out_config': self._getTmpPath('_staged_learning.json'),
            'parameters': params,
            'learning_strategy': {
                'type': self.getEnumText('learnType'),
                'sigma': self.sigma.get(),
                'rate': self.learningRate.get()
            }
        }

        if self.inception.get() is True:
            inceptionRaw = self._extractSmilesToFile(self.inceptSmi.get(), 'inception_smiles_raw.smi')
            inceptionSmiPath = preprocess_smi_file(self, inceptionRaw, 'inception_smiles.smi')
            configParams['inception'] = {
                'smiles_file': inceptionSmiPath,
                'memory_size': self.memSize.get(),
                'sample_size': self.sampSize.get()
            }

        if self.divFilter.get() is True:
            diversityFilter = {
                'type': self.getEnumText('divType'),
                'bucket_size': self.bucketSize.get(),
                'minscore': self.minScore.get()
            }
            divType = self.divType.get()
            if divType == 2:
                diversityFilter['minsimilarity'] = self.minSim.get()
            if divType == 3:
                diversityFilter['penalty_multiplier'] = self.penalty.get()
            configParams['diversity_filter'] = diversityFilter

        stepsText = self.workFlowSteps.get().strip()
        rawStages = [ast.literal_eval(line) for line in stepsText.split('\n') if line.strip()]
        stagesConPath = []
        for i, stageDict in enumerate(rawStages):
            stageNumber = i + 1

            stageDict['chkpt_file'] = self._getPath('SL_S%d.chkpt' % stageNumber)
            stagesConPath.append(stageDict)

        with open(self._getPath('SL_config.toml'), 'w') as f:
            toml.dump(configParams, f)
            if stagesConPath:
                f.write('\n')
                toml.dump({'stage': stagesConPath}, f)

    def runReinventStep(self):
        condaInit = Plugin.getCondaActivationCmd()
        envActivation = Plugin.getEnvActivation()
        executable = Plugin.getProgram()

        parts = [condaInit, f'{envActivation} &&', executable]
        fullCommand = " ".join(parts)
        self.runJob(fullCommand, self._getPath('SL_config.toml'),
                    env=Plugin.getEnviron())

    def createOutputStep(self):
        stepsText = self.workFlowSteps.get().strip()
        numStages = len([l for l in stepsText.split('\n') if l.strip()])
        outputs = {}

        for i in range(numStages):
            stageNum = i + 1
            modelPath = self._getPath('SL_S%d.chkpt' % stageNum)
            csvSrc = self.getProject().getPath('staged_learning_%d.csv' % stageNum)
            csvDest = self._getPath('SL_S%d.csv' % stageNum)

            if os.path.exists(modelPath):
                modelObj = ReinventModel(path=modelPath)
                modelObj.setObjLabel(f"Staged learning model S%d ({self.getEnumText('molGenerator')})" % stageNum)
                outputs['SL%d_TrainedModel' % stageNum] = modelObj

            if os.path.exists(csvSrc):
                shutil.move(csvSrc, csvDest)

        self._defineOutputs(**outputs)


# --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []
        summary.append(f"Generator type: {self.getEnumText('molGenerator')}\n")

        summary.append(f"Configured steps: \n{self.summarySteps.get()}")

        return summary

    def _validate(self):
        errors = []

        if self.extPrior.get() is True and self.priorModel.get() is None:
            errors.append("External prior file must be added.")

        if self.inception.get() is True and self.inceptSmi.get() is None:
            errors.append("Inception molecule set must be added.")

        smilesfile = self._getSmilesSet()
        if smilesfile is None and self.getEnumText('molGenerator') != 'Reinvent':
            errors.append("SMILES molecule set must be added")

        steps = self.workFlowSteps.get()
        lines = [line.strip() for line in steps.split('\n') if line.strip()]
        if not lines:
            errors.append("There must be minimun one STAGE added to the workflow.")

        for index, line in enumerate(lines):
            stageNum = index + 1
            stageDict = ast.literal_eval(line)

            scoringSect = stageDict.get('scoring', {})
            components = scoringSect.get('component', [])

            if not components:
                errors.append(f"STAGE {stageNum} has no components. Minimum one must be added.")
                continue

            totalWeight = 0
            for comp in components:
                for name, content in comp.items():
                    endpoints = content.get('endpoint', [])
                    if endpoints:
                        totalWeight += endpoints[0].get('weight', 1.0)
            if abs(totalWeight - 1.0) > 1e-4:
                errors.append(f"In STAGE {stageNum}, weights sum {totalWeight}. Total sum has to be 1.0.")

        return errors


# --------------------------- LISTING functions -----------------------------

    def countSteps(self):
        stepsStr = self.summarySteps.get() if self.summarySteps.get() is not None else ''
        steps = stepsStr.split('\n')
        return len(steps) - 1

    def _updateSummary(self):
        rawText = self.workFlowSteps.get() or ''
        lines = [l for l in rawText.strip().split('\n') if l.strip()]

        summaryLines = []
        for i, line in enumerate(lines):

            d = ast.literal_eval(line)
            steps = "%s-%s" % (d.get('min_steps', '?'), d.get('max_steps', '?'))

            comps = d.get('scoring', {}).get('component', [])
            compInfo = []
            for c in comps:
                name = list(c.keys())[0]
                weight = c[name]['endpoint'][0].get('weight', 1.0)
                compInfo.append("%s (w:%0.1f)" % (name, weight))

            compStr = " + ".join(compInfo) if compInfo else "EMPTY"

            summaryLines.append("STAGE %d [%s steps]: %s" %
                                 (i + 1, steps, compStr))

        newSummary = "\n".join(summaryLines)
        self.summarySteps.set(newSummary)
        return newSummary
