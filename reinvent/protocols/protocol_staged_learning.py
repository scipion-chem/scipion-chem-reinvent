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
        PriorGroup = form.addGroup('Prior Model')
        PriorGroup.addParam('MolGenerator', EnumParam, choices=['Reinvent', 'LibInvent',
                                                          'LinkInvent', 'Mol2Mol'],
                      default=0,
                      label='Type of Molecule Generator',
                      help='Generative strategy to be used. Each generator requires a specific prior and input data.\n'
                           '- Reinvent: De novo sampling.\n'
                           '- LibInvent: Find R-groups for the given scaffolds.\n'
                           '- LinkInvent: Find a scaffold to link two fragments.\n'
                           '- Mol2Mol: Find molecules similar to provided SMILES.')

        PriorGroup.addParam('ExtPrior', BooleanParam, default='False', expertLevel=LEVEL_ADVANCED,
                      label='Upload external prior file?',
                      help='Set to True to select a custom prior model.')

        PriorGroup.addParam('PriorModel', PointerParam,
                      pointerClass='ReinventModel',
                      condition='ExtPrior==False',
                      allowsNull=True,
                      label='Prior model file',
                      help='Select a trained model. A Transfer Learning protocol should be run first.\n'
                           ' If left empty, the default prior for the selected generator will be used.')

        PriorGroup.addParam('ExtPriorModel', PathParam,
                      condition='ExtPrior==True',
                      label='Prior model file',
                      help='Path to prior model file. Each generator requires a specific prior.')

        PriorGroup.addParam('SmiFileLib', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='MolGenerator==1',
                      label='Scaffold SMILES file',
                      help='One scaffold per line. Each scaffold must be annotated by 2 \'*\' to locate the attachment points.\n'
                           'Up to 4 attachments points are allowed.\n'
                           'Example:\n [*:0]Cc2ccc1cncc(C[*:1])c1c2')

        PriorGroup.addParam('SmiFileLink', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='MolGenerator==2',
                      label='Warheads SMILES file',
                      help='One warhead pair per line. Each warhead must be annotated with \'*\' to locate the attachment points.'
                           'The two warheads must be separated by the pipe symbol.\n'
                           'Example:\n Oc1cncc(*)c1|*c1ccoc1')

        PriorGroup.addParam('SmiFileMol', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='MolGenerator==3',
                      label='Compound SMILES file',
                      help='One compound per line.')

        PriorGroup.addParam('Inception', BooleanParam, default='False',
                      condition='MolGenerator==0',
                      label='Activate Inception?',
                      help='Guide learning into a list of molecules provided.')

        PriorGroup.addParam('InceptSmi', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      condition='MolGenerator==0 and Inception==True',
                      label='Inception SMILES file',
                      help='One molecule per line.')

        PriorGroup.addParam('MemSize', IntParam, default=100, expertLevel=LEVEL_ADVANCED,
                      condition='MolGenerator==0 and Inception==True',
                      label='SMILES held in memory',
                      help='Top N scored molecules. As the learning progresses, the initial molecules are removed '
                           'and replaced by those with higher scores')

        PriorGroup.addParam('SampSize', IntParam, default=10, expertLevel=LEVEL_ADVANCED,
                      condition='MolGenerator==0 and Inception==True',
                      label='SMILES chosen per epoch',
                      help='Number of randomly sampled molecules to be used in computing inception loss.')

        PriorGroup.addParam('SampleStrat', EnumParam, choices=['Beamsearch', 'Multinomial'],
                      condition='MolGenerator==3', expertLevel=LEVEL_ADVANCED,
                      default=0,
                      label='Sampling Strategy',
                      help='Multinomial: Fast random generation based on token probability distribution.\n'
                           'Beamsearch: Deterministic approach that ensures unique compounds '
                           'by selecting the highest probability sequences.')

        PriorGroup.addParam('Temperature', FloatParam, default=1.0, expertLevel=LEVEL_ADVANCED,
                      condition='MolGenerator==3 and SampleStrat==1',
                      label='Temperature in Multinomial sampling',
                      help='Controls randomness. Lower values for more predictable molecules and higher values for more diverse structures.')

        PriorGroup.addParam('DisThres', IntParam, default=100,
                      condition='MolGenerator==3', expertLevel=LEVEL_ADVANCED,
                      label='Distance Threshold',
                      help='Maximum limit on how much the generated molecule can structurally deviate from the source. '
                           'Higher values increase diversity.')

        RunGroup = form.addGroup('Run Mode')

        RunGroup.addParam('UseCkpt', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Use checkpoint?',
                      help='If TRUE use diversity filter from agent file.')

        RunGroup.addParam('PurgeMem', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Purge Memories?',
                      help='Controls if memory is cleared between training stages.'
                           'If FALSE, the model maintains memory across all stages '
                           'ensuring it does not repeat previously found structures.')

        RunGroup.addParam('BatchSize', IntParam, default=128, expertLevel=LEVEL_ADVANCED,
                      label='Batch size',
                      help='Number of molecules generated per run (epoch).')

        RunGroup.addParam('RandomSmi', BooleanParam, default=True,
                      label='Shuffle atoms randomly?',
                      help='If TRUE shuffle atoms in SMILES randomly.')

        LearnGroup = form.addGroup('Learning Strategy', expertLevel=LEVEL_ADVANCED,)
        LearnGroup.addParam('LearnType', EnumParam, choices=['dap'],
                      default=0, expertLevel=LEVEL_ADVANCED,
                      label='Type of Learning Strategy',
                      help='DAP recommended.\n'
                           'Use of default values is also recommended. \n'
                           'If the learning is too slow, you can increase the learning rate.')

        LearnGroup.addParam('Sigma', IntParam, default=128, expertLevel=LEVEL_ADVANCED,
                      label='Sigma of the reward function',
                      help='Determines closeness between Agent and Prior. Higher values push the model to convert'
                           ' faster toward objective.')

        LearnGroup.addParam('LearningRate', FloatParam, default=0.0001, expertLevel=LEVEL_ADVANCED,
                      label='Learning rate',
                      help="Controls how much the model's parameters change in response to estimated error.")

        DivGroup = form.addGroup('Diversity Filter')
        DivGroup.addParam('DivFilter', BooleanParam, default=True,
                      label='Activate Diversity Filter?',
                      help='If TRUE activates memory of agent file that remembers good molecules '
                           'and penalizes the model if it repeats them. Forces to find different structures.')
            

        DivGroup.addParam('DivType', EnumParam, choices=['IdenticalMurckoScaffold', 'IdenticalTopologicalScaffold',
                                                     'ScaffoldSimilarity', 'PenalizeSameSmiles'],
                      condition='DivFilter==True',
                      default=0,
                      label='Diversity Filter type',
                      help='How to group similar molecules.\n'
                            '- IdenticalMurckoScaffold: Groups molecules by their Murcko scaffold (rings and linking chains). '
                                                        'Most common for structural diversity.\n'
                            '- IdenticalTopologicalScaffold: Similar to Murcko but considers atom types and bond orders as generic.\n'
                            '- ScaffoldSimilarity: Penalizes molecules with similar scaffolds.\n'
                            '- PenalizeSameSmiles: Only penalizes exact molecular matches.')

        DivGroup.addParam('BucketSize', IntParam, default=25, expertLevel=LEVEL_ADVANCED,
                      condition='DivFilter==True',
                      label='Bucket size',
                      help='Number of compounds per bucket. Each bucket holds the same scaffold.')

        DivGroup.addParam('MinScore', FloatParam, default=0.4, expertLevel=LEVEL_ADVANCED,
                      condition='DivFilter==True',
                      label='Minimum score',
                      help='Memorize those compounds that have a score value equal or higher to this minimum value.')

        DivGroup.addParam('MinSim', FloatParam, default=0.4, expertLevel=LEVEL_ADVANCED,
                      condition='DivFilter==True and DivType==2',
                      label='Minimum similarity',
                      help='Sets similarity threshold above which a new scaffold is considered redundant compared to stored ones.')

        DivGroup.addParam('Penalty', FloatParam, default=0.5, expertLevel=LEVEL_ADVANCED,
                      condition='DivFilter==True and DivType==3',
                      label='Penalty factor',
                      help='Penalty applied to score of repeated molecule.')

        form.addSection(label='Stage Parameters')
        StageGroup = form.addGroup('Stage Parameters')

        StageGroup.addParam('MaxScore', FloatParam, default=0.7,
                      label='Maximum score',
                      help='Success score. The stage ends when the generated molecules reach this quality.')

        StageGroup.addParam('MinSteps', IntParam, default=10,
                      label='Minimum number of steps',
                      help='Minimum number of epochs to run before checking termination criteria.')

        StageGroup.addParam('MaxSteps', IntParam, default=100,
                      label='Maximum number of steps',
                      help='Maximum number of steps allowed. When reached, the run terminates')

        StageGroup.addParam('ScoreFunct', EnumParam, choices=['geometric_mean', 'arithmetic_mean'],
                      default=0,
                      label='Score function type',
                      help='Aggregation function used to combine all component scores into a single one.\n'
                           'Use geometric for strict multi-parameter balance or arithmetic for flexible average performance.')

        StageGroup.addParam('insertStep', IntParam,
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
        CompGroup = form.addGroup('Scoring component')
        CompGroup.addParam('CompType', EnumParam, choices=component_choices,
                       default=0,
                       label='Select scoring component',
                       help=comp_help)

        CompGroup.addParam('Weight', FloatParam, default=1.0,
                       label='Weight',
                       help='Relative importance of this component in the final score.')

        CompGroup.addParam('Trans', BooleanParam, default=True,
                       label='Apply transformer function?',
                       help='Enable this to normalize the raw chemical value into a score between 0 and 1.')

        CompGroup.addParam('TransFunc', EnumParam, choices=['Sigmoid', 'Reverse_Sigmoid', 'Double_Sigmoid',
                                                               'Right_Step', 'Left_Step', 'Step', 'value_mapping'],
                        default=0,
                        condition='Trans',
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

        CompGroup.addParam('Low', FloatParam, default=50.0,
                           condition='Trans and TransFunc in [0,1,2,4,5]',
                           label='Lower threshold',)

        CompGroup.addParam('Up', FloatParam, default=100.0,
                           condition='Trans and TransFunc in [0,1,2,3,5]',
                           label='Upper threshold')

        CompGroup.addParam('ScoreMatch', FloatParam, default=100.0,
                           condition='Trans and TransFunc == 6',
                           label='Score if matches')

        CompGroup.addParam('ScoreNoMatch', FloatParam, default=50.0,
                           condition='Trans and TransFunc == 6',
                           label='Score if no match')

        CompGroup.addParam('insertComponent', IntParam,
                          default=1,
                          label='Insert component in step index',
                          help='Index of the stage where to insert the new component (wizard).')


        SumGroup = form.addGroup('Summary')
        SumGroup.addParam('workFlowSteps', TextParam, expertLevel=LEVEL_ADVANCED,
                               default='',
                               label='Workflow steps',
                               help='Internal workflow steps (used by wizard).')

        SumGroup.addParam('summarySteps', TextParam,
                               default='',
                               label='Summary steps',
                               help='Protocol summary steps.')

        SumGroup.addParam('delStage', IntParam,
                          default='0', allowsNull=True,
                          label='Delete stage',
                          help='Index of the stage to delete')

        SumGroup.addParam('delComponent', IntParam,
                          default='0', allowsNull=True,
                          label='Delete component',
                          help='Index of the component to delete. The stage index is also needed.')

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

    def _getSmilesSet(self):
        MolGenerator = self.MolGenerator.get()
        if MolGenerator == 1:
            return self.SmiFileLib.get()
        elif MolGenerator == 2:
            return self.SmiFileLink.get()
        elif MolGenerator == 3:
            return self.SmiFileMol.get()
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
        self._insertFunctionStep('createConfigFileStep')
        self._insertFunctionStep('runReinventStep')
        self._insertFunctionStep('createOutputStep')

    def createConfigFileStep(self):

        prior_path=self._getPriorFile()

        params = {
            'summary_csv_prefix': 'staged_learning',
            'use_checkpoint': self.UseCkpt.get(),
            'purge_memories': self.PurgeMem.get(),
            'batch_size': self.BatchSize.get(),
            'unique_sequences': 'true',
            'randomize_smiles': self.RandomSmi.get(),
            'prior_file': prior_path,
            'agent_file': prior_path,
        }

        molSet = self._getSmilesSet()
        if molSet is not None:
            raw_smi_path = self._extractSmilesToFile(molSet, 'smiles_raw.smi')
            new_smiles_file = preprocess_smi_file(self, raw_smi_path, 'smiles_cleaned.smi')
            params['smiles_file'] = new_smiles_file


        SampleStrat = self.SampleStrat.get()
        if self.MolGenerator.get() == 3:
            if SampleStrat == 0:
                params['sample_strategy'] = 'beamsearch'
            if SampleStrat == 1:
                params['sample_strategy'] = 'multinomial'
                params['temperature'] = self.Temperature.get()
            params['distance_threshold'] = self.DisThres.get()

        config_params = {
            'run_type': 'staged_learning',
            'device': 'cpu',
            'json_out_config': self._getTmpPath('_staged_learning.json'),
            'parameters': params,
            'learning_strategy': {
                'type': self.getEnumText('LearnType'),
                'sigma': self.Sigma.get(),
                'rate': self.LearningRate.get()
            }
        }

        if self.Inception.get() is True:
            inception_raw = self._extractSmilesToFile(self.InceptSmi.get(), 'inception_smiles_raw.smi')
            inception_smi_path = preprocess_smi_file(self, inception_raw, 'inception_smiles.smi')
            config_params['inception'] = {
                'smiles_file': inception_smi_path,
                'memory_size': self.MemSize.get(),
                'sample_size': self.SampSize.get()
            }

        if self.DivFilter.get() is True:
            diversity_filter = {
                'type': self.getEnumText('DivType'),
                'bucket_size': self.BucketSize.get(),
                'minscore': self.MinScore.get()
            }
            DivType = self.DivType.get()
            if DivType == 2:
                diversity_filter['minsimilarity'] = self.MinSim.get()
            if DivType == 3:
                diversity_filter['penalty_multiplier'] = self.Penalty.get()
            config_params['diversity_filter'] = diversity_filter

        steps_text = self.workFlowSteps.get().strip()
        raw_stages = [ast.literal_eval(line) for line in steps_text.split('\n') if line.strip()]
        stages_con_path = []
        for i, stage_dict in enumerate(raw_stages):
            stage_number = i + 1

            stage_dict['chkpt_file'] = self._getPath('SL_S%d.chkpt' % stage_number)
            stages_con_path.append(stage_dict)

        with open(self._getPath('SL_config.toml'), 'w') as f:
            toml.dump(config_params, f)
            if stages_con_path:
                f.write('\n')
                toml.dump({'stage': stages_con_path}, f)

    def runReinventStep(self):
        CondaInit = Plugin.getCondaActivationCmd()
        EnvActivation = Plugin.getEnvActivation()
        Executable = Plugin.getProgram()

        parts = [CondaInit, f'{EnvActivation} &&', Executable]
        full_command = " ".join(parts)
        self.runJob(full_command, self._getPath('SL_config.toml'),
                    env=Plugin.getEnviron())

    def createOutputStep(self):
        steps_text = self.workFlowSteps.get().strip()
        num_stages = len([l for l in steps_text.split('\n') if l.strip()])
        outputs={}

        for i in range(num_stages):
            stage_num = i + 1
            model_path = self._getPath('SL_S%d.chkpt' % stage_num)
            csv_src = self.getProject().getPath('staged_learning_%d.csv' % stage_num)
            csv_dest = self._getPath('SL_S%d.csv' % stage_num)

            if os.path.exists(model_path):
                model_obj = ReinventModel(path=model_path)
                model_obj.setObjLabel(f"Staged learning model S%d ({self.getEnumText('MolGenerator')})" % stage_num)
                outputs['SL%d_TrainedModel' % stage_num] = model_obj

            if os.path.exists(csv_src):
                shutil.move(csv_src, csv_dest)

        self._defineOutputs(**outputs)


# --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []
        summary.append(f"Generator type: {self.getEnumText('MolGenerator')}\n")

        summary.append(f"Configured steps: \n{self.summarySteps.get()}")

        return summary

    def _validate(self):
        errors = []

        if self.ExtPrior.get() is True and self.PriorModel.get() is None:
            errors.append("External prior file must be added.")

        if self.Inception.get() is True and self.InceptSmi.get() is None:
            errors.append("Inception molecule set must be added.")

        smilesfile = self._getSmilesSet()
        if smilesfile is None and self.getEnumText('MolGenerator') != 'Reinvent':
            errors.append("SMILES molecule set must be added")

        steps = self.workFlowSteps.get()
        lines = [line.strip() for line in steps.split('\n') if line.strip()]
        if not lines:
            errors.append("There must be minimun one STAGE added to the workflow.")

        for index, line in enumerate(lines):
            stage_num = index + 1
            stage_dict = ast.literal_eval(line)

            scoring_sect = stage_dict.get('scoring', {})
            components = scoring_sect.get('component', [])

            if not components:
                errors.append(f"STAGE {stage_num} has no components. Minimum one must be added.")
                continue

            total_weight = 0
            for comp in components:
                for name, content in comp.items():
                    endpoints = content.get('endpoint', [])
                    if endpoints:
                        total_weight += endpoints[0].get('weight', 1.0)
            if round(total_weight, 4) != 1.0:
                errors.append(f"In STAGE {stage_num}, weights sum {total_weight}. Total sum has to be 1.0.")

        return errors


# --------------------------- LISTING functions --------    ---------------------------

    def countSteps(self):
        stepsStr = self.summarySteps.get() if self.summarySteps.get() is not None else ''
        steps = stepsStr.split('\n')
        return len(steps) - 1

    def _updateSummary(self):
        import ast
        raw_text = self.workFlowSteps.get() or ''
        lines = [l for l in raw_text.strip().split('\n') if l.strip()]

        summary_lines = []
        for i, line in enumerate(lines):

            d = ast.literal_eval(line)
            steps = "%s-%s" % (d.get('min_steps', '?'), d.get('max_steps', '?'))

            comps = d.get('scoring', {}).get('component', [])
            comp_info = []
            for c in comps:
                name = list(c.keys())[0]
                weight = c[name]['endpoint'][0].get('weight', 1.0)
                comp_info.append("%s (w:%0.1f)" % (name, weight))

            comp_str = " + ".join(comp_info) if comp_info else "EMPTY"

            summary_lines.append("STAGE %d [%s steps]: %s" %
                                 (i + 1, steps, comp_str))

        new_summary = "\n".join(summary_lines)
        self.summarySteps.set(new_summary)
        return new_summary