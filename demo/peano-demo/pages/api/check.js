const axios = require('axios');

export default async (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const solution = params.solution;

  const response = await axios.get('http://localhost:8080/check?solution=' + encodeURIComponent(solution));

  console.log('Response from Python server:', response);

  res.status(200).json(response.data);
}
