
## Your Code ##

# I want to go to the ...
# I want to get to the ...
# I'd like to go to the ...
# I plan on going to the ...
# im going to the ...
# I will be going to the ...
# I'll go to the ...
# do you know the way to go to the ...
# do you know where the party is
# do you know how to get to the party

# destination plan bot

import re
import random

def check(param, option):
    try:
        if option != 'choice':
            if option == 'dest':
                pattern = r'\b(.*[D|d]?o you know.*the|.*[I|i]?.*[to]*.*the) (\w+)\b'

            elif option == 'valid':
                pattern = r'\b(^[Y|y](e*|(es*))|^[N|n]o*)\b'

            if re.match(pattern, param):
                return re.match(pattern, param)
            raise Exception

        elif option == 'choice':
            match message:
                case 1:
                    print("bot: Here are your plans")
                    for i in range(len(ls)):
                        print(f"    {i+1}. {ls[i]}")
                    return 1
                case 2:
                    print("bot: Where would you like to go?")
                    return 2
                case 3:
                    print("bot: Thank you for using our service")
                    return 3
                case _:
                    pass

    except:
        print("bot: Sorry, I didn't understand that.")

print("bot: Hello, I'm a destination plan bot. \nbot: Where would you like to go?")

perm = 0
ls = []

while 1:
    message = input('user: ')
    dest = check(message, 'dest').group(2)
    disc = random.randint(1, 100)
    print("bot: " + dest + f"? Alright, I know where that is. \nbot: It's going to be {disc} km away and that would cost about {disc*10} baht. \nbot: Would you like to go there? (y/n)")
    message = input('user: ')
    valid = check(message, 'valid').group(1)
    if re.match('^[Y|y]', valid):
        while 1:
            if perm == 0:
                perm = 1
                ls.append({'location': dest, 'distance': disc, 'cost': disc*10})
                print("bot: Great! I'll add the location in your plan.")

            print("bot: press the number in order to access the following options 1.(see all plans) 2.(add another plan) 3.(exit)")
            message = int(input('user: '))
            choice = check(message, 'choice')
            if choice == 1:
                continue
            elif choice == 2:
                perm = 0
                break
            elif choice == 3:
                perm = -1
                break

        if perm == -1:
            break
        continue
    else:
        print("bot: I understood, anywhere else would you like to go?")
        continue