from resume_screener.extraction import guess_name


def test_name_split_over_two_lines():
    lines = ["Prathamesh", "Patil", "+91 73490 41840 | Bengaluru"]
    assert guess_name(lines, "prathameshpatil330@gmail.com", "c.pdf") == "Prathamesh Patil"


def test_name_glued_to_email():
    lines = ["Sumaiya Sultana Shaiksultanasumaiya623@gmail.com — +91 97015 28467"]
    assert guess_name(lines, "sultanasumaiya623@gmail.com", "c.pdf") == "Sumaiya Sultana Shaik"


def test_plain_name_unchanged():
    assert guess_name(["ASHA RAO", "asha@x.com"], "asha@x.com", "c.pdf") == "Asha Rao"


def test_skills_line_is_not_a_name():
    # Mirrors a resume whose first lines are a summary and a skills list:
    # no real name line exists, so we must fall back to the email, not pick
    # "Programming Language" or "English Tamil" as the name.
    lines = [
        "database management, and software development life cycle.",
        "Programming Language: Python",
        "Web Development: HTML, CSS",
        "Database: SQL",
        "English",
        "Tamil",
    ]
    assert guess_name(lines, "priyaraguraman29@gmail.com", "c.pdf") == "Priyaraguraman"