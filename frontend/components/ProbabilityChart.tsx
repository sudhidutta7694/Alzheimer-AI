"use client";

import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

type ProbItem = { index: number; label: string; probability: number };

export default function ProbabilityChart({ data }: { data: ProbItem[] }) {
  const labels = data.map((d) => d.label);
  const probabilities = data.map((d) => Math.round(d.probability * 10000) / 100);

  const chartData = {
    labels,
    datasets: [
      {
        label: "Probability (%)",
        data: probabilities,
        backgroundColor: "rgba(0, 188, 212, 0.6)",
        borderColor: "rgba(0, 188, 212, 1)",
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: { color: "#e6f0f7" },
        grid: { color: "rgba(255,255,255,0.1)" },
      },
      x: {
        ticks: { color: "#e6f0f7" },
        grid: { color: "rgba(255,255,255,0.1)" },
      },
    },
    plugins: {
      legend: { labels: { color: "#e6f0f7" } },
      tooltip: {
        callbacks: {
          label: (ctx: any) => `${ctx.parsed.y.toFixed(2)}%`,
        },
      },
    },
  } as const;

  return <Bar data={chartData} options={options} />;
}
