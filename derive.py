
def derive(equation,variable):
    splitEquationList=splitEquation(equation,variable)
    
    if len(splitEquationList) == 1 :
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
            else :
                multiplier=''
                # Choose the split term to multiply
                for multiplicationSplit in multiplicationSplits:
                    if(variable in multiplicationSplit):
                        # Derive the value for variable
                        # Handling Powers
                        multiplicationSplit=derivePower(multiplicationSplit,variable)
                    derivedValue+=multiplier+multiplicationSplit
                    multiplier='*'
            return derivedValue
        elif equation.startswith("-"):
            return "-"+str(1)
        else :
            return str(1)  
    return mergeEquation(splitEquationList,variable)
    
def mergeEquation(splitEquation,variable):
    equationString=''
    add=''
    for subEquation in splitEquation:
        derivation = derive(subEquation,variable)
        if (derivation[0].startswith("+") or derivation[0].startswith("-") ) :
            equationString+=derivation
        else :
            equationString+=add+derivation
            add='+'
    return equationString

def splitEquation(equation,variable):
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
    return splitEquation

def derivePower(terms,variable):
    if terms.index(variable)+1 < len(terms) and "^" in terms[terms.index(variable)+1]:
        powerValue=terms[(terms.index(variable)+2):]
        if powerValue.isdigit():
            powerValue = int(powerValue)
            powerValue -=1
            powerValue = str(powerValue)
        else:
            powerValue ="("+powerValue+"-1)"
        terms=terms[:(terms.index(variable)+2)]+str(powerValue)
    else:
        terms=terms.replace(variable,"1")
    return terms


print(derive("2*X^2-y+b-z*X^r","X"))