.PHONY: run-cpu, run-gpu, stop-cpu, stop-gpu

run-gpu:
	docker compose -f compose-gpu.yml up --build -d

run-cpu:
	docker compose -f compose-cpu.yml up --build -d

stop-gpu:
	docker compose -f compose-gpu.yml down 

stop-cpu:
	docker compose -f compose-cpu.yml down 
