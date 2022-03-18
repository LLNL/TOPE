{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --job-name="{{ id }}"
#SBATCH --partition=pdebug
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --account=suprcap
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
module load opt mesa
module load paraview/5.10.0
{% endblock %}


