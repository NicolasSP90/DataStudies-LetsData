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
