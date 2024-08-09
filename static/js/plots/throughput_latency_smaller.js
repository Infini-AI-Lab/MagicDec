document.addEventListener('DOMContentLoaded', function() {
    // First chart
    const ctx1 = document.getElementById('chart1').getContext('2d');
    ctx1.canvas.width = 300;  // Set the width of the first chart
    ctx1.canvas.height = 200;  // Set the height of the first chart

    const data1 = {
        datasets: [
            {
                label: '(AutoReg) Prefill 8000',
                data: [
                    {x: 16.53, y: 1936},
                    {x: 21.45, y: 2237},
                    {x: 31.49, y: 2237},
                    {x: 49.89, y: 2565}
                ],
                borderColor: 'rgb(153, 102, 51)',
                backgroundColor: 'rgb(153, 102, 51)',
                fill: false,
                tension: 0.1,
                pointStyle: 'rect',
                pointRadius: 6,
                borderWidth: 2,
                showLine: true
            },
            {
                label: '(SpecDec) Prefill 8000',
                data: [
                    {x: 12.83, y: 2493},
                    {x: 15.26, y: 3146},
                    {x: 20.09, y: 3186},
                    {x: 30.41, y: 4209}
                ],
                borderColor: 'rgb(102, 51, 0)',
                backgroundColor: 'rgb(102, 51, 0)',
                fill: false,
                tension: 0.1,
                pointStyle: 'circle',
                pointRadius: 6,
                borderWidth: 2,
                showLine: true
            }
        ]
    };

    const options1 = {
        scales: {
            y: {
                type: 'linear',
                title: {
                    display: true,
                    text: 'Max Throughput (tokens/s)',
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Avg. Tokenwise Latency (ms)'
                },
                type: 'linear',
                position: 'bottom',
            }
        }
    };

    const config1 = {
        type: 'scatter',
        data: data1,
        options: options1
    };

    new Chart(ctx1, config1);

    // Second chart
    const ctx2 = document.getElementById('chart2').getContext('2d');
    ctx2.canvas.width = 375;  // Set the width of the second chart
    ctx2.canvas.height = 250;  // Set the height of the second chart

    const data2 = {
        datasets: [
            {
                label: 'Prefill 2048',
                data: [{x: 9.19, y: 1.03}, {x: 10.27, y: 1.06}, {x: 12.19, y: 1.14}, {x: 16.96, y: 1.13}, {x: 26.9, y: 1.13}],
                borderColor: 'rgb(153, 102, 51)',
                backgroundColor: 'rgb(153, 102, 51)',
                fill: false,
                tension: 0.1,
                pointStyle: 'rect',
                pointRadius: 6,
                borderWidth: 2,
                showLine: true
            },
            {
                label: 'Prefill 4000',
                data: [{x: 10.25, y: 1.16}, {x: 11.93, y: 1.21}, {x: 14.86, y: 1.3}, {x: 21.54, y: 1.34}, {x: 34.85, y: 1.4}],
                borderColor: 'rgb(102, 51, 0)',
                backgroundColor: 'rgb(102, 51, 0)',
                fill: false,
                tension: 0.1,
                pointStyle: 'circle',
                pointRadius: 6,
                borderWidth: 2,
                showLine: true
            },
            {
                label: 'Prefill 8000',
                data: [{x: 12.83, y: 1.29}, {x: 15.26, y: 1.41}, {x: 20.09, y: 1.57}, {x: 30.41, y: 1.64}],
                borderColor: 'rgb(0, 51, 102)',
                backgroundColor: 'rgb(0, 51, 102)',
                fill: false,
                tension: 0.1,
                pointStyle: 'triangle',
                pointRadius: 6,
                borderWidth: 2,
                showLine: true
            },
            {
                label: 'Prefill 16000',
                data: [{x: 16.74, y: 1.57}, {x: 21.26, y: 1.69}, {x: 29.79, y: 1.85}],
                borderColor: 'rgb(0, 102, 153)',
                backgroundColor: 'rgb(0, 102, 153)',
                fill: false,
                tension: 0.1,
                pointStyle: 'rectRot',
                pointRadius: 6,
                borderWidth: 2,
                showLine: true
            },
            {
                label: 'Prefill 24000',
                data: [{x: 20.76, y: 1.72}, {x: 26.95, y: 1.86}],
                borderColor: 'rgb(0, 153, 204)',
                backgroundColor: 'rgb(0, 153, 204)',
                fill: false,
                tension: 0.1,
                pointStyle: 'circle',
                pointRadius: 6,
                borderWidth: 2,
                showLine: true
            },
            {
                label: 'Prefill 32000',
                data: [{x: 24.04, y: 1.87}],
                borderColor: 'rgb(0, 204, 255)',
                backgroundColor: 'rgb(0, 204, 255)',
                fill: false,
                tension: 0.1,
                pointStyle: 'triangle',
                pointRadius: 6,
                borderWidth: 2,
                showLine: true
            }
        ]
    };

    const options2 = {
        scales: {
            y: {
                type: 'linear',
                title: {
                    display: true,
                    text: 'Throughput Ratio (SpecDec/AutoReg)'
                }
            },
            x: {
                type: 'linear',
                position: 'bottom',
                title: {
                    display: true,
                    text: 'Avg. Tokenwise Latency (ms)'
                }
            }
        }
    };

    const config2 = {
        type: 'scatter',
        data: data2,
        options: options2
    };

    new Chart(ctx2, config2);
});
