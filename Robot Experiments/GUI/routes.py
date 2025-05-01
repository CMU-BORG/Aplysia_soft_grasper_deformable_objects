from flask import Blueprint, render_template

main = Blueprint("main", __name__)

@main.route("/")
def index():
    return render_template("index.jinja")










###### individual pages ######
@main.route("/landing-PROCTOR")
def landing_proc():
    return render_template("landing-PROCTOR.jinja")

@main.route("/landing-PARTICIPANT")
def landing_part():
    return render_template("survey-link-PARTICIPANT.jinja")

@main.route("/soft-grasper-PROCTOR")
def soft_grasper_proc():
    return render_template("soft-grasper-PROCTOR.jinja")

@main.route("/soft-grasper-PARTICIPANT")
def soft_grasper_part():
    return render_template("soft-grasper-PARTICIPANT.jinja")

@main.route("/dual-grasper-PROCTOR")
def dual_grasper_proc():
    return render_template("dual-grasper-PROCTOR.jinja")

@main.route("/dual-grasper-PARTICIPANT")
def dual_grasper_part():
    return render_template("dual-grasper-PARTICIPANT.jinja")
