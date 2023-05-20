from gtts import gTTS

# language = 'en'
language = 'hi'

# mytext = 'this is an example in english!'
# mytext = 'यह हिंदी में एक उदाहरण है'

with open("selcted_lines_hin.txt") as fp:
    lines = fp.readlines()
    i=0
    for mytext in lines:
        myobj = gTTS(text=mytext, lang=language, tld='co.in', slow=False)
        myobj.save(f"data/hindi/{i}_female_hin.wav")
        print(f"{i}/{len(lines)} done...")
        i += 1
