
def derive(equation,variable):
    addEquation=equation.split("+")
    splitEquation=[]

    # Handling +/- symbols in the equation
    for equation in addEquation:
        negatedEquation = equation.split("-")
        for i in range(0,len(negatedEquation)):
            if i != 0 and len(negatedEquation[i])!=0 :
                splitEquation.append("-"+negatedEquation[i])
            elif len(negatedEquation[i])!=0 :
                splitEquation.append(negatedEquation[i])
    if len(splitEquation) == 1 :
        if variable in equation:
            # Handling Individual Derivation in the equation
            # Handle multiplication
            derivedValue=''
            multiplicationSplits=equation.split("*")
            count=[]
            count += [1 if variable in multiplicationSplit else 0 for multiplicationSplit in multiplicationSplits ]
            if(sum(count)>1):
                # TODO:Do Deriation for U*V in it
                derivedValue=equation
                print("Contains totally "+str(count))
            else :
                # Choose the split term to multiply
                multiplier=''
                for multiplicationSplit in multiplicationSplits:
                    if(variable in multiplicationSplit):
                        # Derive the value for variable
                        # Handling Powers
                        if multiplicationSplit.index(variable)+1 < len(multiplicationSplit) and "^" in multiplicationSplit[multiplicationSplit.index(variable)+1]:
                            powerValue=multiplicationSplit[(multiplicationSplit.index(variable)+2):]
                            if powerValue.isdigit():
                                powerValue = int(powerValue)
                                powerValue -=1
                                powerValue = str(powerValue)
                            else:
                                powerValue ="("+powerValue+"-1)"
                            multiplicationSplit=multiplicationSplit[:(multiplicationSplit.index(variable)+2)]+str(powerValue)
                        else:
                            multiplicationSplit=multiplicationSplit.replace(variable,"1")
                    derivedValue+=multiplier+multiplicationSplit
                    multiplier='*'
            return derivedValue
        elif equation.startswith("-"):
            return "-"+str(1)
        else :
            return str(1)  
    equationString=''
    add=''
    for subEquation in splitEquation:
        derivation = derive(subEquation,variable)
#        print("Derivation gave "+derivation)
        if (derivation[0].startswith("+") or derivation[0].startswith("-") ) :
            equationString+=derivation
        else :
            equationString+=add+derivation
            add='+'
    return equationString
    
    
print(derive("2*X^2-y+b-z*X^r","X"))