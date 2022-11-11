module.exports.apiRequest = async (endpoint, parameters={}) => {
  const req = await fetch('/api/' + endpoint + '?params=' + encodeURIComponent(JSON.stringify(parameters)),
                          { method: 'POST' });
  const res = await req.json();
  return res;
}
