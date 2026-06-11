=======================
REINVENT scipion plugin
=======================

This is a **Scipion** plugin that integrates **REINVENT4**, a molecular generation tool
for *de novo* drug design with target profiles.

Therefore, this plugin allows to use REINVENT4 workflows within the Scipion framework,
making it easier to automate generative chemistry pipelines alongside other chemoinformatics tools.

Current protocols implemented:

    - Transfer Learning
    - Staged Learning
    - Sampling

The full documentation to the plugin can be found in the `official documentation page <https://github.com/MolecularAI/REINVENT4>`_.

==========================
Install this plugin
==========================

You will need to use `Scipion3 <https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html>`_ to run these protocols.

1. **Install the plugin in Scipion**

- **Install the stable version (Not available yet)**

    Through the plugin manager GUI by launching Scipion and following **Configuration** >> **Plugins**

    or

.. code-block::

    scipion3 installp -p scipion-chem-reinvent4


- **Developer's version**

    1. **Download repository**:

    .. code-block::

        git clone https://github.com/scipion-chem/scipion-chem-reinvent.git

2. **Switch to the desired branch** (master or devel):

    Scipion-chem-reinvent is constantly under development and including new features.
    If you want a relatively older and more stable version, use master branch (default).
    If you want the latest changes and developments, use devel branch.

    .. code-block::

                cd scipion-chem-reinvent
                git checkout devel

    3. **Install**:

    .. code-block::

        scipion3 installp -p path_to_scipion-chem-reinvent --devel
