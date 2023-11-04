from PraNet_Res2Net import PraNet

model = PraNet()
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.6f" % (total))