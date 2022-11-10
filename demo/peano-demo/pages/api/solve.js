const axios = require('axios');

export default async (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const equation = params.equation;

  const response = await axios.get('http://localhost:8080/solve?equation=' + encodeURIComponent(equation));

  console.log('Response from Python server:', response);

  res.status(200).json(response.data);
}
