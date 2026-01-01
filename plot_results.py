import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("navigation_results.csv")

plt.figure()
df["trust"].plot(title="Trust Over Time")
plt.xlabel("Step")
plt.ylabel("Trust")
plt.grid()
plt.savefig("trust_plot.png")

plt.figure()
df["learned_lambda"].plot(title="Lambda Evolution")
plt.xlabel("Step")
plt.ylabel("Lambda")
plt.grid()
plt.savefig("lambda_plot.png")

plt.figure()
df["drift_threshold"].plot(label="Drift Threshold")
df["failure_threshold"].plot(label="Failure Threshold")
plt.title("Threshold Adaptation")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig("thresholds_plot.png")

print("Plots Saved:")
print("- trust_plot.png")
print("- lambda_plot.png")
print("- thresholds_plot.png")
