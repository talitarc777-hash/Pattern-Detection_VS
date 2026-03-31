import traceback
import sys
import threading
import time
import test_color_smoke

def handler():
    time.sleep(5)
    print("\n\n--- HANG TRACE ---")
    frame = sys._current_frames()[main_thread_id]
    traceback.print_stack(frame)
    print("------------------\n\n")
    sys.exit(1)

main_thread_id = threading.main_thread().ident
t = threading.Thread(target=handler, daemon=True)
t.start()

test_color_smoke.run_test()
