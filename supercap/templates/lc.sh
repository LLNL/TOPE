{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --job-name="{{ id }}"
#SBATCH --nodes=1
#SBATCH --partition=pbatch
#SBATCH --account=suprcap
#SBATCH -t 20:00:00
#SBATCH --ntasks=1
{% if partition %}
#SBATCH --partition={{ partition }}
{% endif %}
{% if walltime %}
#SBATCH -t {{ walltime|format_timedelta }}
{% endif %}
{% if job_output %}
#SBATCH --output={{ job_output }}
#SBATCH --error={{ job_output }}
{% endif %}
{% endblock %}

{% block body %}
source /usr/workspace/$USER/firedrake_install/activate.sh
{% for operation in operations %}
{{ operation.cmd }}
{% endfor %}
{% endblock %}
