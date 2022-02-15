import sys
import hashlib
import time
import json


x=['a','b','c','d','e','f','g','h','i']
y=[1,2,3,4,5,6,7,8,9]




class Block():
    def __init__(self, index, timestamp, data):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previousHash = 0
        self.hash = self.calHash()

    def calHash(self):
        return hashlib.sha256(str(self.index).encode() + str(self.data).encode() +
                              str(self.timestamp).encode() + str(self.previousHash).encode()).hexdigest()

class BlockChain:
    def __init__(self):
        self.chain = []
        self.createGenesis()

    def createGenesis(self):
        self.chain.append(Block(0, time.time(), 'Genesis'))

    def addBlock(self, nBlock):
        nBlock.previousHash = self.chain[len(self.chain)-1].hash
        nBlock.hash = nBlock.calHash()
        self.chain.append(nBlock)

    def isValid(self):
        i = 1
        while(i<len(self.chain)):
            if(self.chain[i].hash != self.chain[i].calHash()):
                return False
            if(self.chain[i].previousHash != self.chain[i-1].hash):
                return False
            i += 1
        return True

def check_the_contact_level(contact_level):
    print("Result : ")
    print("============================================================================================================")
    if (contact_level == 0):
        print("Contact level : F")
        print("It seems that you didn't contact any infections.")
        level= "F"
    elif (contact_level < 1):
        print("Contact level : D")
        print("You might contact to few infected people. If you have any symptoms, please visit a hospital.")
        level= "D"
    elif (contact_level < 2):
        print("Contact level : C")
        print("You've contacted with infected people. It is better to do a medical examination.")
        level= "C"
    elif (contact_level < 3):
        print("Contact level : B")
        print("You've contacted with many infected people. Please visit a hospital.")
        level= "B"
    else:
        print("Contact level : A")
        print("You need to isolate yourself and go to a hospital ASAP.")
        level= "A"
    print(
        "============================================================================================================")
    return level

def check_axis(axis):
    a=0
    for k in range(0,9):
        if(axis[0]==x[k]):
            a=a+1
        if(str(axis[1])==str(y[k])):
            a=a+1

    if(a!=2):
        return "error"
    else:
        return "good"

def alp_to_num(a):
    j=0
    for k in x:
        if (a == x[j]):
            return j+1
        j = j + 1

def same_diff(start_axis,end_axis):
    start_x = start_axis[0]
    start_y = int(start_axis[1])
    end_x = end_axis[0]
    end_y = int(end_axis[1])
    diff_x = alp_to_num(start_x) - alp_to_num(end_x)
    diff_y = start_y - end_y
    if (diff_x == 1):
        list_of_grid.append(start_axis)
        list_of_grid.append(end_axis)
    else:
        # 둘다 작거나 클경우
        if ((diff_x * diff_y) > 0):
            if (diff_x > 0):
                temp_axis = start_axis
                start_axis = end_axis
                end_axis = temp_axis
            for k in range(-1, abs(diff_x)):
                list_of_grid.append(x[alp_to_num(start_axis[0]) + k] + str(y[int(start_axis[1]) + k]))
        # 하나만 작거나 클경우
        else:
            if (diff_x > 0):
                temp_axis = start_axis
                start_axis = end_axis
                end_axis = temp_axis
            list_of_grid.append(start_axis)
            for k in range(0, abs(diff_x)):
                list_of_grid.append(x[alp_to_num(start_axis[0]) + k] + str(y[int(start_axis[1]) - k - 2]))
    return


def connect_with_the_line(start_axis,end_axis):

    start_x = start_axis[0]
    start_y = int(start_axis[1])
    end_x = end_axis[0]
    end_y = int(end_axis[1])
    diff_x = alp_to_num(start_x) - alp_to_num(end_x)
    diff_y = start_y - end_y
    # case 1 : 같은좌표
    if (start_axis == end_axis):
        list_of_grid.append(start_axis)
        return
    # case 2 : 직선좌표
    # 처음이 같음
    if (start_axis[0]==end_axis[0]):
        if(int(start_axis[1])-int(end_axis[1]))<0:
            list_of_grid.append(start_axis)
            for k in range(0,abs(int(start_axis[1])-int(end_axis[1]))):
                list_of_grid.append(start_axis[0]+str(y[int(start_axis[1])+k]))
        else:
            list_of_grid.append(end_axis)
            for k in range(0,abs(int(start_axis[1])-int(end_axis[1]))):
                list_of_grid.append(start_axis[0]+str(y[int(end_axis[1])+k]))
        return
    # 뒤가같음
    elif (start_axis[1]==end_axis[1]):
        if (alp_to_num(start_axis[0]) - alp_to_num(end_axis[0])) < 0:
            list_of_grid.append(start_axis)
            for k in range(0, abs(alp_to_num(start_axis[0]) - alp_to_num(end_axis[0]))):
                list_of_grid.append(x[alp_to_num(start_axis[0])+k] + end_axis[1])
        else:
            list_of_grid.append(end_axis)
            for k in range(0, abs(alp_to_num(start_axis[0]) - alp_to_num(end_axis[0]))):
                list_of_grid.append(x[alp_to_num(end_axis[0])+k]+ end_axis[1])
        return
    # case 3 : 일반좌표 - 차이가 양쪽 같다 (대각선 연결 종료)
    if (abs(diff_x)==abs(diff_y)):
        same_diff(start_axis,end_axis)
        return

    # case 4 : 일반좌표 - 차이가 다르다
    else:
        new_start_axis=start_axis
        new_end_axis=end_axis
        #처음 차이가 크다
        while 1:
            if(abs(diff_x)>abs(diff_y)):
                if (diff_x > 0):
                    temp_axis = new_start_axis
                    new_start_axis = new_end_axis
                    new_end_axis = temp_axis
                # 큰값 하나 줄이기

                list_of_grid.append(new_start_axis)
                new_start_axis=x[alp_to_num(new_start_axis[0])]+new_start_axis[1]
                list_of_grid.append(new_end_axis)
                new_end_axis = x[alp_to_num(new_end_axis[0]) - 2] + new_end_axis[1]
            #뒷자리 차이가 크다
            else:
                if (diff_y > 0):
                    temp_axis = new_start_axis
                    new_start_axis = new_end_axis
                    new_end_axis = temp_axis

                list_of_grid.append(new_start_axis)
                new_start_axis=new_start_axis[0]+ str(y[int(new_start_axis[1])])
                list_of_grid.append(new_end_axis)
                new_end_axis = new_end_axis[0] + str(y[int(new_end_axis[1])-2])

                # 큰값 하나 줄이기


            new_start_x = new_start_axis[0]
            new_start_y = int(new_start_axis[1])
            new_end_x = new_end_axis[0]
            new_end_y = int(new_end_axis[1])
            diff_x = alp_to_num(new_start_x) - alp_to_num(new_end_x)
            diff_y = new_start_y - new_end_y

            # 대각선인지 비교
            # 대각선이다
            if(abs(diff_x) == abs(diff_y)):
                same_diff(new_start_axis, new_end_axis)
                break
            elif(new_start_x==new_end_x):
                if (int(new_start_axis[1]) - int(new_end_axis[1])) < 0:
                    list_of_grid.append(new_start_axis)
                    for k in range(0, abs(int(new_start_axis[1]) - int(new_end_axis[1]))):
                        list_of_grid.append(new_start_axis[0] + str(y[int(new_start_axis[1]) + k]))
                else:
                    list_of_grid.append(new_end_axis)
                    for k in range(0, abs(int(new_start_axis[1]) - int(new_end_axis[1]))):
                        list_of_grid.append(new_start_axis[0] + str(y[int(new_end_axis[1]) + k]))

                break
            elif(new_start_y==new_end_y):
                if (alp_to_num(new_start_axis[0]) - alp_to_num(new_end_axis[0])) < 0:
                    list_of_grid.append(new_start_axis)
                    for k in range(0, abs(alp_to_num(new_start_axis[0]) - alp_to_num(new_end_axis[0]))):
                        list_of_grid.append(x[alp_to_num(new_start_axis[0]) + k] + new_end_axis[1])
                else:
                    list_of_grid.append(end_axis)
                    for k in range(0, abs(alp_to_num(new_start_axis[0]) - alp_to_num(new_end_axis[0]))):
                        list_of_grid.append(x[alp_to_num(new_end_axis[0]) + k] + new_end_axis[1])
                break

    return


f=open('infected.txt','r')
list_of_time = []
list_of_spot = []
list_of_chain = []
line_count=0
while True:
    line = f.readline()
    if not line: break
    if(line_count==0):
        patient_num=line[:-1]
        line_count=1
        print("============================================================================================================")
        print("Information of New case")
        print("============================================================================================================")
        print("Patient ID : " + patient_num)
    else:
        list_of_time.append(line.split(',')[0])
        list_of_spot.append(line.split(',')[1][:-1])
f.close()

for num in range(0,len(list_of_spot) - 1):
    print("time : " + list_of_time[num] + " / " + "visit : " + list_of_spot[num].upper()+"-"+list_of_spot[num+1].upper())

print("time : " + list_of_time[len(list_of_time)-1] + " / " + "visit : " + list_of_spot[len(list_of_spot)-1].upper())
print("============================================================================================================")


if (input("Do you want to link this information to blockchain? (Y/N) ")=="y"):
    autentication_key = hashlib.sha256("this_person_is_infected".encode()).hexdigest()
    print("============================================================================================================")
    print("Autentication Key :  " +autentication_key)
    contact_tracing = BlockChain()
    for k in range(0,len(list_of_spot) - 1):
        # print(list_of_spot[k]+"-"+list_of_spot[k+1],end=' ')
        list_of_grid = []
        connect_with_the_line(list_of_spot[k],list_of_spot[k+1])
        list_of_grid.sort()
        for j in list_of_grid:
            hash_value=hashlib.sha256((list_of_time[k]+":"+j).encode())
            print(list_of_time[k]+":"+j + "   =====>   " + hash_value.hexdigest())
            contact_tracing.addBlock(Block(len(contact_tracing.chain), time.time(), {"patient_id": patient_num,"value": hash_value.hexdigest(),"Authentication Key : ": autentication_key}))
            list_of_chain.append((hash_value.hexdigest()))
    # last spot
    hash_value=hashlib.sha256((list_of_time[len(list_of_time)-1]+":" +list_of_spot[len(list_of_spot)-1]).encode())
    print(list_of_time[len(list_of_time)-1]+":" +list_of_spot[len(list_of_spot)-1]+ "   =====>   " + hash_value.hexdigest())
    list_of_chain.append((hash_value.hexdigest()))
    contact_tracing.addBlock(Block(len(contact_tracing.chain), time.time(), {"patient_id": patient_num, "value": hash_value.hexdigest(),"Authentication Key : ": autentication_key}))
    print("============================================================================================================")

    if (input("Do you want to check the detail of blockchain? (Y/N) ") == "y"):
        print("============================================================================================================")
        for block in contact_tracing.chain:
            print(json.dumps(vars(block), indent=4))
            print("============================================================================================================")

while 1:
    case_num=input("Contact tracing : Person A => not_infected / Person B => infected ( 1 / 2 / exit ) : ")

    list_of_time = []
    list_of_spot = []
    list_of_compare = []
    line_count = 0

    if( case_num == "1"):
        print("============================================================================================================")
        f = open('compare_case1.txt', 'r')
        while True:
            line = f.readline()
            if not line: break
            if (line_count == 0):
                patient_num = line[:-1]
                line_count = 1
                print("Information for contact tracing")
                print(
                    "============================================================================================================")
                print("Patient ID : " + patient_num)
            else:
                list_of_time.append(line.split(',')[0])
                list_of_spot.append(line.split(',')[1][:-1])
        f.close()

        for num in range(0, len(list_of_spot) - 1):
            print("time : " + list_of_time[num] + " / " + "visit : " + list_of_spot[num].upper() + "-" + list_of_spot[
                num + 1].upper())

        print("time : " + list_of_time[len(list_of_time) - 1] + " / " + "visit : " + list_of_spot[
            len(list_of_spot) - 1].upper())
        print(
            "============================================================================================================")
        if (input("Do you want to contact tracing with this information? (Y/N) ") == "y"):
            print(
                "============================================================================================================")
            autentication_key = hashlib.sha256("this_person_is_not_infected".encode()).hexdigest()
            contact_level=0
            print("Autentication Key :  " + autentication_key)
            contact_tracing = BlockChain()
            for k in range(0, len(list_of_spot) - 1):
                # print(list_of_spot[k]+"-"+list_of_spot[k+1],end=' ')
                list_of_grid = []
                connect_with_the_line(list_of_spot[k], list_of_spot[k + 1])
                list_of_grid.sort()
                for j in list_of_grid:
                    hash_value = hashlib.sha256((list_of_time[k] + ":" + j).encode())
                    list_of_compare.append(hash_value.hexdigest())
                    print(list_of_time[k] + ":" + j + "   =====>   " + hash_value.hexdigest())
                    contact_tracing.addBlock(Block(len(contact_tracing.chain), time.time(),
                                                   {"patient_id": patient_num, "value": hash_value.hexdigest(),
                                                    "Authentication Key : ": autentication_key}))

            # last spot
            hash_value = hashlib.sha256(
                (list_of_time[len(list_of_time) - 1] + ":" + list_of_spot[len(list_of_spot) - 1]).encode())
            print(list_of_time[len(list_of_time) - 1] + ":" + list_of_spot[
                len(list_of_spot) - 1] + "   =====>   " + hash_value.hexdigest())
            list_of_compare.append(hash_value.hexdigest())
            contact_tracing.addBlock(Block(len(contact_tracing.chain), time.time(),
                                           {"patient_id": patient_num, "value": hash_value.hexdigest(),
                                            "Authentication Key : ": autentication_key}))
            print(
                "============================================================================================================")
            for list1 in list_of_chain:
                for list2 in list_of_compare:
                    if list1==list2:
                        contact_level=contact_level+1
            level=check_the_contact_level(contact_level)

    elif(case_num=="2"):
        print(
            "============================================================================================================")
        f = open('compare_case2.txt', 'r')
        list_of_time = []
        list_of_spot = []
        line_count = 0
        while True:
            line = f.readline()
            if not line: break
            if (line_count == 0):
                patient_num = line[:-1]
                line_count = 1
                print("Information for contact tracing")
                print(
                    "============================================================================================================")
                print("Patient ID : " + patient_num)
            else:
                list_of_time.append(line.split(',')[0])
                list_of_spot.append(line.split(',')[1][:-1])
        f.close()

        for num in range(0, len(list_of_spot) - 1):
            print("time : " + list_of_time[num] + " / " + "visit : " + list_of_spot[num].upper() + "-" + list_of_spot[
                num + 1].upper())

        print("time : " + list_of_time[len(list_of_time) - 1] + " / " + "visit : " + list_of_spot[
            len(list_of_spot) - 1].upper())
        print(
            "============================================================================================================")
        if (input("Do you want to contact tracing with this information? (Y/N) ") == "y"):
            print(
                "============================================================================================================")
            autentication_key = hashlib.sha256("this_person_is_not_infected".encode()).hexdigest()
            contact_level = 0
            print("Autentication Key :  " + autentication_key)
            contact_tracing = BlockChain()
            for k in range(0, len(list_of_spot) - 1):
                # print(list_of_spot[k]+"-"+list_of_spot[k+1],end=' ')
                list_of_grid = []
                connect_with_the_line(list_of_spot[k], list_of_spot[k + 1])
                list_of_grid.sort()
                for j in list_of_grid:
                    hash_value = hashlib.sha256((list_of_time[k] + ":" + j).encode())
                    list_of_compare.append(hash_value.hexdigest())
                    print(list_of_time[k] + ":" + j + "   =====>   " + hash_value.hexdigest())
                    contact_tracing.addBlock(Block(len(contact_tracing.chain), time.time(),
                                                   {"patient_id": patient_num, "value": hash_value.hexdigest(),
                                                    "Authentication Key : ": autentication_key}))

            # last spot
            hash_value = hashlib.sha256(
                (list_of_time[len(list_of_time) - 1] + ":" + list_of_spot[len(list_of_spot) - 1]).encode())
            print(list_of_time[len(list_of_time) - 1] + ":" + list_of_spot[
                len(list_of_spot) - 1] + "   =====>   " + hash_value.hexdigest())
            list_of_compare.append(hash_value.hexdigest())
            contact_tracing.addBlock(Block(len(contact_tracing.chain), time.time(),
                                           {"patient_id": patient_num, "value": hash_value.hexdigest(),
                                            "Authentication Key : ": autentication_key}))
            print(
                "============================================================================================================")
            for list1 in list_of_chain:
                for list2 in list_of_compare:
                    if list1 == list2:
                        contact_level = contact_level + 1
            level=check_the_contact_level(contact_level)
            if ((level=="A")or(level=="B")):
                print(patient_num +"----->  infected !!!")
                if (input("Do you want to link this information to blockchain? (Y/N) ") == "y"):
                    autentication_key = hashlib.sha256("this_person_is_infected".encode()).hexdigest()
                    print(
                        "============================================================================================================")
                    print("Autentication Key :  " + autentication_key)
                    for k in range(0, len(list_of_spot) - 1):
                        # print(list_of_spot[k]+"-"+list_of_spot[k+1],end=' ')
                        list_of_grid = []
                        connect_with_the_line(list_of_spot[k], list_of_spot[k + 1])
                        list_of_grid.sort()
                        for j in list_of_grid:
                            hash_value = hashlib.sha256((list_of_time[k] + ":" + j).encode())
                            print(list_of_time[k] + ":" + j + "   =====>   " + hash_value.hexdigest())
                            contact_tracing.addBlock(Block(len(contact_tracing.chain), time.time(),
                                                           {"patient_id": patient_num, "value": hash_value.hexdigest(),
                                                            "Authentication Key : ": autentication_key}))
                            list_of_chain.append((hash_value.hexdigest()))
                    # last spot
                    hash_value = hashlib.sha256(
                        (list_of_time[len(list_of_time) - 1] + ":" + list_of_spot[len(list_of_spot) - 1]).encode())
                    print(list_of_time[len(list_of_time) - 1] + ":" + list_of_spot[
                        len(list_of_spot) - 1] + "   =====>   " + hash_value.hexdigest())
                    list_of_chain.append((hash_value.hexdigest()))
                    contact_tracing.addBlock(Block(len(contact_tracing.chain), time.time(),
                                                   {"patient_id": patient_num, "value": hash_value.hexdigest(),
                                                    "Authentication Key : ": autentication_key}))
                    print(
                        "============================================================================================================")

                    if (input("Do you want to check the detail of blockchain? (Y/N) ") == "y"):
                        print(
                            "============================================================================================================")
                        for block in contact_tracing.chain:
                            print(json.dumps(vars(block), indent=4))
                            print(
                                "============================================================================================================")
    else:
        print(
            "============================== Fin. ========================================================================")
        sys.exit()









