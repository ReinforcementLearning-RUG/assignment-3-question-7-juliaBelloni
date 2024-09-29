from util import create_mdp, create_policy_1, create_policy_2
from model_free_prediction.monte_carlo_evaluator import MCEvaluator
from model_free_prediction.td_evaluator import TDEvaluator
from model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator
def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    
    mdp = create_mdp()
    policy1 = create_policy_1()
    policy2 = create_policy_2()
    episodes = 1000

    # MC
    mc = MCEvaluator(mdp)
    value_mc_1 = mc.evaluate(policy1, episodes)
    value_mc_2 = mc.evaluate(policy2, episodes)
    print(f"MC values for policy1: {value_mc_1}, policy2: {value_mc_2}")

    # TD(0)
    td = TDEvaluator(mdp, 0.5)
    value_td_1 = td.evaluate(policy1, 1000)
    value_td_2 = td.evaluate(policy2, 1000)
    print(f"TD(0) values for policy1: {value_td_1}, policy2: {value_td_2}")

    # TD(gamma)
    td_lambda = TDLambdaEvaluator(mdp, 0.5, 0.1)
    value_td_lambda_1 = td.evaluate(policy1, 1000)
    value_td_lambda_2 = td.evaluate(policy2, 1000)
    print(f"TD(lamda) values for policy1: {value_td_lambda_1}, policy2: {value_td_lambda_2}")




if __name__ == "__main__":
    main()
