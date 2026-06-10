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
from pwchem.wizards import VariableWizard
from reinvent.protocols import ReinventStagedLearning
import ast


class AddStageWizard(VariableWizard):
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        inputParam, outputParam = self.getInputOutput(form)
        protocol = form.protocol

        if getattr(protocol, inputParam[0]).get() != '':
            index = int(getattr(protocol, inputParam[0]).get())

        msjDic = {
            'termination': "simple",
            'max_score': protocol.MaxScore.get(),
            'min_steps': protocol.MinSteps.get(),
            'max_steps': protocol.MaxSteps.get(),
            'scoring': {'type': protocol.getEnumText('ScoreFunct'),
                        'component': []}
        }

        current_text = getattr(protocol, outputParam[0]).get() or ''
        workSteps = [line for line in current_text.strip().split('\n') if line.strip()]

        if index > len(workSteps) or index <= 0:
            new_text = current_text.strip() + '\n' + str(msjDic) + '\n'
        else:
            workSteps.insert(index - 1, str(msjDic))
            new_text = '\n'.join(workSteps) + '\n'

        form.setVar(outputParam[0], new_text.lstrip())

        new_summary = protocol._updateSummary()
        form.setVar('summarySteps', new_summary)


class AddScoringComponentWizard(VariableWizard):
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        inputParam, outputParam = self.getInputOutput(form)
        protocol = form.protocol

        raw_index = getattr(protocol, inputParam[0]).get()
        target_idx = int(raw_index) - 1 if raw_index and str(raw_index).strip() != '' else 0
        comp_name = protocol.getEnumText('CompType')
        endpoint_data = {
            'name': comp_name,
            'weight': protocol.Weight.get()
        }

        if protocol.Trans.get() is True:
            trans_type_index = protocol.TransFunc.get()
            endpoint_data['transform.type'] = protocol.getEnumText('TransFunc')

            if trans_type_index in [0, 1, 2, 4, 5]:
                endpoint_data['transform.low'] = protocol.Low.get()

            if trans_type_index in [0, 1, 2, 3, 5]:
                endpoint_data['transform.high'] = protocol.Up.get()

            if trans_type_index == 2:
                endpoint_data['transform.coef_div'] = 100.00
                endpoint_data['transform.coef_si'] = 10.00
                endpoint_data['transform.coef_se'] = 10.00

            if trans_type_index == 6:
                endpoint_data['transform.mapping'] = {
                    comp_name: protocol.ScoreMatch.get(),
                    "No %s" % comp_name: protocol.ScoreNoMatch.get()
                }

        comp_dict = {
            comp_name: {
                'endpoint': [endpoint_data]
            }
        }
        current_text = getattr(protocol, outputParam[0]).get() or ''
        workSteps = [line for line in current_text.strip().split('\n') if line.strip()]

        if 0 <= target_idx < len(workSteps):
            stage_data = ast.literal_eval(workSteps[target_idx])

            if 'component' not in stage_data['scoring']:
                stage_data['scoring']['component'] = []

            stage_data['scoring']['component'].append(comp_dict)

            workSteps[target_idx] = str(stage_data)
            new_text = '\n'.join(workSteps) + '\n'

            protocol.workFlowSteps.set(new_text)
            form.setVar(outputParam[0], new_text.lstrip())

            new_summary = protocol._updateSummary()
            form.setVar('summarySteps', new_summary)


class DeleteStageWizard(VariableWizard):
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        inputParam, outputParam = self.getInputOutput(form)
        protocol = form.protocol
        try:
            index = int(getattr(protocol, inputParam[0]).get())
            current_text = getattr(protocol, outputParam[0]).get() or ''
            workSteps = [line for line in current_text.strip().split('\n') if line.strip()]
            if 1 <= index <= len(workSteps):
                del workSteps[index - 1]
                if workSteps:
                    new_text = '\n'.join(workSteps) + '\n'
                else:
                    new_text = ''
                form.setVar(outputParam[0], new_text)
                newSum = protocol._updateSummary()
                form.setVar('summarySteps', newSum)
        except:
            print('Incorrect index')

class DeleteComponentWizard(VariableWizard):
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        protocol = form.protocol
        try:
            s_idx = int(protocol.delStage.get())
            c_idx = int(protocol.delComponent.get())

            current_text = protocol.workFlowSteps.get() or ''
            workSteps = [l for l in current_text.strip().split('\n') if l.strip()]

            if 1 <= s_idx <= len(workSteps):
                stage_data = ast.literal_eval(workSteps[s_idx - 1])

                components = stage_data.get('scoring', {}).get('component', [])

                if 1 <= c_idx <= len(components):
                    del components[c_idx - 1]

                    workSteps[s_idx - 1] = str(stage_data)
                    new_text = '\n'.join(workSteps) + '\n'

                    protocol.workFlowSteps.set(new_text)
                    form.setVar('workFlowSteps', new_text)

                    if hasattr(protocol, '_updateSummary'):
                        new_sum = protocol._updateSummary()
                        form.setVar('summarySteps', new_sum)

        except:
            print('Incorrect index')

AddStageWizard().addTarget(protocol=ReinventStagedLearning,
                           targets=['insertStep'],
                           inputs=['insertStep'],
                           outputs=['workFlowSteps', 'summarySteps'])

AddScoringComponentWizard().addTarget(protocol=ReinventStagedLearning,
                           targets=['insertComponent'],
                           inputs=['insertComponent'],
                           outputs=['workFlowSteps', 'summarySteps'])

DeleteStageWizard().addTarget(protocol=ReinventStagedLearning,
                              targets=['delStage'],
                              inputs=['delStage'],
                              outputs=['workFlowSteps', 'summarySteps'])

DeleteComponentWizard().addTarget(protocol=ReinventStagedLearning,
                                  targets=['delComponent'],
                                  inputs=['delComponent', 'delStage'],
                                  outputs=['workFlowSteps', 'summarySteps'])