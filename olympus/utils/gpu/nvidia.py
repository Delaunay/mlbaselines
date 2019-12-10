import subprocess
import torch.cuda

from olympus.utils.stat import StatStream


nvidia_smi = 'nvidia-smi'
metrics = [
    'index',
    'temperature.gpu',
    'utilization.gpu',
    'utilization.memory',
    'memory.total',
    'memory.free',
    'memory.used'
]
query = '--query-gpu=' + ','.join(metrics)


class NvGpuMonitor:
    def __init__(self, loop_interval, device_id):
        self.options = ['--format=csv', '--loop-ms=' + str(loop_interval), '--id=' + str(device_id)]
        self.n = len(metrics)
        self.process = None
        self.running = True
        self.n_gpu = torch.cuda.device_count()
        self.dispatcher = {
            'name': self.process_ignore,
            'temperature.gpu': self.process_value,
            'utilization.gpu': self.process_percentage,
            'utilization.memory': self.process_percentage,
            'memory.total': self.process_memory,
            'memory.free': self.process_memory,
            'memory.used': self.process_memory
        }
        # All GPUs
        self.overall = {
            k: StatStream(drop_first_obs=2) for k in metrics[1:]
        }
        # Per GPUs
        self.streams = {
            k: [StatStream(drop_first_obs=2)] * self.n_gpu for k in metrics[1:]
        }

        self.ts = {}
        for k in metrics[1:]:
            for i in range(self.n_gpu):
                self.ts[f'{k}{i}'] = []

    def to_json(self, overall=True, extended=False):
        to_json = lambda x: x.to_json()
        if not extended:
            to_json = lambda x: x.avg

        return {
            k: to_json(item) for k, item in self.overall.items()
        }

    def metrics(self):
        return self.streams

    def run(self):
        try:
            with subprocess.Popen([nvidia_smi, query] + self.options, stdout=subprocess.PIPE, bufsize=1) as proc:
                self.process = proc
                count = 0
                while self.running:
                    line = proc.stdout.readline()
                    if count > 0:
                        self.parse(line.decode('UTF-8').strip())
                    count += 1
                proc.kill()
        except:
            pass

    def parse(self, line):
        if line == '':
            return

        elems = line.split(',')

        if len(elems) != self.n:
            print('Error line mismatch {} != {} with \n -> `{}`'.format(len(elems), self.n, line))
            return

        gpu_index = int(elems[0])
        gpu_data = elems[1:]

        for index, value in enumerate(gpu_data):
            metric_name = metrics[index + 1]
            self.dispatcher[metric_name](metric_name, gpu_index, value)

    def process_percentage(self, metric_name, gpu_index, value):
        try:
            value, _ = value.strip().split(' ')
            self.process_value(metric_name, gpu_index, value)
        except Exception as e:
            print('Expected value format: `66 %` got `{}`'.format(value))
            print(e)

    def process_value(self, metric_name, gpu_index, value):
        self.streams[metric_name][gpu_index] += float(value)
        self.overall[metric_name] += float(value)
        self.ts[f'{metric_name}{gpu_index}'].append(float(value))

    def process_ignore(self, metric_name, gpu_index, value):
        pass

    def process_memory(self, metric_name, gpu_index, value):
        try:
            value, _ = value.strip().split(' ')
            self.process_value(metric_name, gpu_index, value)
        except Exception as e:
            print('Expected value format: `66 Mib` got `{}`'.format(value))
            print(e)

    def stop(self):
        if self.process is not None:
            self.running = False
            self.process.terminate()

