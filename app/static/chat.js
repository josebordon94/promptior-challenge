async function send() {
  const input = document.getElementById("input");
  const chat = document.getElementById("chat");
  const text = input.value.trim();

  if (!text) return;

  chat.innerHTML += `<li class="list-group-item"><b>You:</b> ${text}</li>`;
  input.value = "";

  const res = await fetch("/chat/invoke", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: text }),
  });

  const data = await res.json();

  chat.innerHTML += `<li class="list-group-item"><b>Bot:</b> ${data.output}</li>`;
}
