def prettytext(text):
    text_size = len(text)
    border = "*" * text_size
    result = border + "<br>" + text + "<br>" + border
    return result

def even_number(number):
    if number % 2 == 1:
        return f"{number} is odd"
    else:
        return f"{number} is even"

def hasdiabetes(age, weight, glucose):
    # imagine that this condition is based on a ML model
    # THIS IS NOT A REAL TEST. LEARNING PURPOSE ONLY
    if (age > 40) & (weight > 100) & (glucose > 120):
        return True
    else:
        return False