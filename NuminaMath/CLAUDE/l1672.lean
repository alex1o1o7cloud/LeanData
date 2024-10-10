import Mathlib

namespace inverse_scalar_multiple_l1672_167285

/-- Given a 2x2 matrix B and a constant l, prove that B^(-1) = l * B implies e = -3 and l = 1/19 -/
theorem inverse_scalar_multiple (B : Matrix (Fin 2) (Fin 2) ℝ) (l : ℝ) :
  B 0 0 = 3 ∧ B 0 1 = 4 ∧ B 1 0 = 7 ∧ B 1 1 = B.det / (3 * B 1 1 - 28) →
  B⁻¹ = l • B →
  B 1 1 = -3 ∧ l = 1 / 19 := by
sorry

end inverse_scalar_multiple_l1672_167285


namespace quadratic_root_value_l1672_167297

/-- Given a quadratic equation 6x^2 + 5x + q with roots (-5 ± i√323) / 12, q equals 14.5 -/
theorem quadratic_root_value (q : ℝ) : 
  (∀ x : ℂ, 6 * x^2 + 5 * x + q = 0 ↔ x = (-5 + Complex.I * Real.sqrt 323) / 12 ∨ x = (-5 - Complex.I * Real.sqrt 323) / 12) →
  q = 14.5 := by sorry

end quadratic_root_value_l1672_167297


namespace dean_5000th_number_l1672_167266

/-- Represents the number of numbers spoken by a player in a given round -/
def numbers_spoken (player : Nat) (round : Nat) : Nat :=
  player + round - 1

/-- Calculates the sum of numbers spoken by all players up to a given round -/
def total_numbers_spoken (round : Nat) : Nat :=
  (1 + 2 + 3 + 4) * round + (0 + 1 + 2 + 3) * (round * (round - 1) / 2)

/-- Calculates the starting number for a player in a given round -/
def start_number (player : Nat) (round : Nat) : Nat :=
  total_numbers_spoken (round - 1) + 
  (if player > 1 then (numbers_spoken 1 round + numbers_spoken 2 round + numbers_spoken 3 round) else 0) + 1

/-- The main theorem to be proved -/
theorem dean_5000th_number : 
  ∃ (round : Nat), start_number 4 round ≤ 5000 ∧ 5000 ≤ start_number 4 round + numbers_spoken 4 round - 1 :=
by sorry

end dean_5000th_number_l1672_167266


namespace sum_of_first_four_powers_of_i_is_zero_l1672_167293

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The property that i^2 = -1 -/
axiom i_squared : i^2 = -1

/-- Theorem: The sum of the first four powers of i equals 0 -/
theorem sum_of_first_four_powers_of_i_is_zero : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end sum_of_first_four_powers_of_i_is_zero_l1672_167293


namespace geometric_figure_area_l1672_167295

theorem geometric_figure_area (x : ℝ) : 
  x > 0 →
  (3*x)^2 + (4*x)^2 + (1/2) * (3*x) * (4*x) = 1200 →
  x = Real.sqrt (1200/31) := by
sorry

end geometric_figure_area_l1672_167295


namespace function_properties_l1672_167291

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * a * x)
def g (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem function_properties (a k b : ℝ) (h_k : k ≠ 0) :
  -- Part 1
  (f a 1 = Real.exp 1 ∧ ∀ x, g k b x = -g k b (-x)) →
  a = 1/2 ∧ b = 0 ∧
  -- Part 2
  (∀ x > 0, f (1/2) x > g k 0 x) →
  k < Real.exp 1 ∧
  -- Part 3
  (∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ f (1/2) x₁ = g k 0 x₁ ∧ f (1/2) x₂ = g k 0 x₂) →
  x₁ * x₂ < 1 :=
by sorry

end

end function_properties_l1672_167291


namespace train_crossing_time_l1672_167282

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 280 →
  train_speed_kmh = 50.4 →
  crossing_time = train_length / (train_speed_kmh * (1000 / 3600)) →
  crossing_time = 20 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1672_167282


namespace exactly_one_positive_l1672_167223

theorem exactly_one_positive (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (product_one : a * b * c = 1) :
  (a > 0 ∧ b ≤ 0 ∧ c ≤ 0) ∨
  (a ≤ 0 ∧ b > 0 ∧ c ≤ 0) ∨
  (a ≤ 0 ∧ b ≤ 0 ∧ c > 0) :=
sorry

end exactly_one_positive_l1672_167223


namespace julia_short_amount_l1672_167269

/-- Represents the cost and quantity of CDs Julia wants to buy -/
structure CDPurchase where
  rock_price : ℕ
  pop_price : ℕ
  dance_price : ℕ
  country_price : ℕ
  quantity : ℕ

/-- Calculates the amount Julia is short given her CD purchase and available money -/
def amount_short (purchase : CDPurchase) (available_money : ℕ) : ℕ :=
  let total_cost := purchase.quantity * (purchase.rock_price + purchase.pop_price + purchase.dance_price + purchase.country_price)
  if total_cost > available_money then
    total_cost - available_money
  else
    0

/-- Theorem stating that Julia is short $25 given the specific CD prices, quantities, and available money -/
theorem julia_short_amount : amount_short ⟨5, 10, 3, 7, 4⟩ 75 = 25 := by
  sorry

end julia_short_amount_l1672_167269


namespace binomial_22_10_l1672_167211

theorem binomial_22_10 (h1 : Nat.choose 20 8 = 125970)
                       (h2 : Nat.choose 20 9 = 167960)
                       (h3 : Nat.choose 20 10 = 184756) :
  Nat.choose 22 10 = 646646 := by
  sorry

end binomial_22_10_l1672_167211


namespace cosh_leq_exp_squared_l1672_167239

theorem cosh_leq_exp_squared (k : ℝ) :
  (∀ x : ℝ, Real.cosh x ≤ Real.exp (k * x^2)) ↔ k ≥ (1/2 : ℝ) :=
by sorry

end cosh_leq_exp_squared_l1672_167239


namespace smallest_w_l1672_167252

theorem smallest_w (w : ℕ+) (h1 : (2^5 : ℕ) ∣ (936 * w))
                            (h2 : (3^3 : ℕ) ∣ (936 * w))
                            (h3 : (12^2 : ℕ) ∣ (936 * w)) :
  w ≥ 36 ∧ (∃ (v : ℕ+), v ≥ 36 → 
    (2^5 : ℕ) ∣ (936 * v) ∧ 
    (3^3 : ℕ) ∣ (936 * v) ∧ 
    (12^2 : ℕ) ∣ (936 * v)) :=
sorry

end smallest_w_l1672_167252


namespace solid_circles_count_l1672_167229

def circleSequence (n : ℕ) : ℕ := n * (n + 3) / 2 + 1

theorem solid_circles_count (total : ℕ) (h : total = 2019) :
  ∃ n : ℕ, circleSequence n ≤ total ∧ circleSequence (n + 1) > total ∧ n = 62 :=
by sorry

end solid_circles_count_l1672_167229


namespace range_of_z_plus_4_minus_3i_l1672_167240

/-- The range of |z+4-3i| when |z| = 2 -/
theorem range_of_z_plus_4_minus_3i (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (w : ℂ), Complex.abs w = 2 ∧ Complex.abs (w + 4 - 3*Complex.I) = 3 ∧
  ∃ (v : ℂ), Complex.abs v = 2 ∧ Complex.abs (v + 4 - 3*Complex.I) = 7 ∧
  ∀ (u : ℂ), Complex.abs u = 2 → 3 ≤ Complex.abs (u + 4 - 3*Complex.I) ∧ Complex.abs (u + 4 - 3*Complex.I) ≤ 7 :=
by sorry

end range_of_z_plus_4_minus_3i_l1672_167240


namespace g_nested_3_l1672_167260

def g (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else x^2 + 2

theorem g_nested_3 : g (g (g (g 3))) = 3 := by
  sorry

end g_nested_3_l1672_167260


namespace race_distance_proof_l1672_167254

/-- The distance of the race where B beats C by 100 m, given the conditions from the problem. -/
def race_distance : ℝ := 700

theorem race_distance_proof (Va Vb Vc : ℝ) 
  (h1 : Va / Vb = 1000 / 900)
  (h2 : Va / Vc = 600 / 472.5)
  (h3 : Vb / Vc = (race_distance - 100) / race_distance) : 
  race_distance = 700 := by sorry

end race_distance_proof_l1672_167254


namespace solve_income_problem_l1672_167255

def income_problem (day2 day3 day4 day5 average : ℚ) : Prop :=
  let known_days := [day2, day3, day4, day5]
  let total := 5 * average
  let sum_known := day2 + day3 + day4 + day5
  let day1 := total - sum_known
  (day2 = 150) ∧ (day3 = 750) ∧ (day4 = 400) ∧ (day5 = 500) ∧ (average = 420) →
  day1 = 300

theorem solve_income_problem :
  ∀ day2 day3 day4 day5 average,
  income_problem day2 day3 day4 day5 average :=
by
  sorry

end solve_income_problem_l1672_167255


namespace intersection_point_of_lines_l1672_167220

theorem intersection_point_of_lines (x y : ℚ) : 
  (3 * y = -2 * x + 6 ∧ -2 * y = 7 * x + 4) ↔ (x = -24/17 ∧ y = 50/17) :=
by sorry

end intersection_point_of_lines_l1672_167220


namespace fourth_power_difference_l1672_167281

theorem fourth_power_difference (a b : ℝ) : 
  (a - b)^4 = a^4 - 4*a^3*b + 6*a^2*b^2 - 4*a*b^3 + b^4 :=
by
  sorry

-- Given condition
axiom fourth_power_sum (a b : ℝ) : 
  (a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4

end fourth_power_difference_l1672_167281


namespace subtraction_result_l1672_167231

theorem subtraction_result : 3.56 - 2.15 = 1.41 := by
  sorry

end subtraction_result_l1672_167231


namespace factory_sampling_is_systematic_l1672_167232

/-- Represents a sampling method -/
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

/-- Represents the characteristics of a sampling process -/
structure SamplingProcess where
  orderedArrangement : Bool
  fixedInterval : Bool

/-- Determines the sampling method based on the sampling process characteristics -/
def determineSamplingMethod (process : SamplingProcess) : SamplingMethod :=
  if process.orderedArrangement && process.fixedInterval then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom -- Default case, not actually used in this problem

/-- Theorem stating that the given sampling process is systematic sampling -/
theorem factory_sampling_is_systematic 
  (process : SamplingProcess)
  (h1 : process.orderedArrangement = true)
  (h2 : process.fixedInterval = true) :
  determineSamplingMethod process = SamplingMethod.Systematic := by
  sorry


end factory_sampling_is_systematic_l1672_167232


namespace pizza_area_increase_l1672_167238

theorem pizza_area_increase (r : ℝ) (hr : r > 0) :
  let medium_area := π * r^2
  let large_radius := 1.1 * r
  let large_area := π * large_radius^2
  (large_area - medium_area) / medium_area = 0.21 := by sorry

end pizza_area_increase_l1672_167238


namespace quadratic_square_completion_l1672_167216

theorem quadratic_square_completion (d e : ℤ) : 
  (∀ x, x^2 - 10*x + 13 = 0 ↔ (x + d)^2 = e) → d + e = 7 := by
  sorry

end quadratic_square_completion_l1672_167216


namespace transformation_result_l1672_167202

def initial_point : ℝ × ℝ × ℝ := (1, 1, 1)

def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def transformation_sequence (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_y_180 |> reflect_yz |> reflect_xz |> rotate_y_180 |> reflect_xz

theorem transformation_result :
  transformation_sequence initial_point = (-1, 1, 1) := by
  sorry

#eval transformation_sequence initial_point

end transformation_result_l1672_167202


namespace no_three_naturals_with_prime_sums_l1672_167218

theorem no_three_naturals_with_prime_sums :
  ¬ ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.Prime (a + b) ∧ 
    Nat.Prime (a + c) ∧ 
    Nat.Prime (b + c) :=
sorry

end no_three_naturals_with_prime_sums_l1672_167218


namespace ancient_chinese_pi_l1672_167221

/-- Proves that for a cylinder with given dimensions, the implied value of π is 3 -/
theorem ancient_chinese_pi (c h v : ℝ) (hc : c = 48) (hh : h = 11) (hv : v = 2112) :
  let r := c / (2 * π)
  v = π * r^2 * h → π = 3 := by
  sorry

end ancient_chinese_pi_l1672_167221


namespace quadratic_solution_property_l1672_167244

theorem quadratic_solution_property (p q : ℝ) : 
  (3 * p^2 + 4 * p - 8 = 0) → 
  (3 * q^2 + 4 * q - 8 = 0) → 
  (p - 2) * (q - 2) = 4 := by
sorry

end quadratic_solution_property_l1672_167244


namespace quadratic_function_range_l1672_167287

/-- A quadratic function with a positive coefficient for the squared term -/
structure PositiveQuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  positive_coeff : ∃ a b c : ℝ, (∀ x, f x = a * x^2 + b * x + c) ∧ a > 0

/-- The main theorem -/
theorem quadratic_function_range
  (f : PositiveQuadraticFunction)
  (h_symmetry : ∀ x : ℝ, f.f (2 + x) = f.f (2 - x))
  (h_inequality : ∀ x : ℝ, f.f (1 - 2*x^2) < f.f (1 + 2*x - x^2)) :
  {x : ℝ | -2 < x ∧ x < 0} = {x : ℝ | f.f (1 - 2*x^2) < f.f (1 + 2*x - x^2)} := by
  sorry

end quadratic_function_range_l1672_167287


namespace max_sum_of_seventh_powers_l1672_167263

theorem max_sum_of_seventh_powers (a b c d : ℝ) (h : a^6 + b^6 + c^6 + d^6 = 64) :
  ∃ (m : ℝ), m = 128 ∧ a^7 + b^7 + c^7 + d^7 ≤ m ∧ 
  ∃ (a' b' c' d' : ℝ), a'^6 + b'^6 + c'^6 + d'^6 = 64 ∧ a'^7 + b'^7 + c'^7 + d'^7 = m :=
by sorry

end max_sum_of_seventh_powers_l1672_167263


namespace adjacent_units_conversion_rate_l1672_167210

-- Define the units of length
inductive LengthUnit
  | Kilometer
  | Meter
  | Decimeter
  | Centimeter
  | Millimeter

-- Define the concept of adjacent units
def adjacent (u v : LengthUnit) : Prop :=
  (u = LengthUnit.Kilometer ∧ v = LengthUnit.Meter) ∨
  (u = LengthUnit.Meter ∧ v = LengthUnit.Decimeter) ∨
  (u = LengthUnit.Decimeter ∧ v = LengthUnit.Centimeter) ∨
  (u = LengthUnit.Centimeter ∧ v = LengthUnit.Millimeter)

-- Define the conversion rate function
def conversionRate (u v : LengthUnit) : ℕ := 10

-- State the theorem
theorem adjacent_units_conversion_rate (u v : LengthUnit) :
  adjacent u v → conversionRate u v = 10 := by
  sorry

end adjacent_units_conversion_rate_l1672_167210


namespace simplify_power_of_product_l1672_167286

theorem simplify_power_of_product (y : ℝ) : (3 * y^4)^2 = 9 * y^8 := by
  sorry

end simplify_power_of_product_l1672_167286


namespace crude_oil_mixture_l1672_167283

/-- Proves that 30 gallons from the second source is needed to obtain 50 gallons of 55% hydrocarbon crude oil -/
theorem crude_oil_mixture (x y : ℝ) : 
  x + y = 50 →                   -- Total amount is 50 gallons
  0.25 * x + 0.75 * y = 0.55 * 50 →  -- Hydrocarbon balance equation
  y = 30 := by sorry

end crude_oil_mixture_l1672_167283


namespace black_white_difference_l1672_167275

/-- Represents the number of pieces in a box -/
structure PieceCount where
  black : ℕ
  white : ℕ

/-- The condition of the problem -/
def satisfiesCondition (p : PieceCount) : Prop :=
  (p.black - 1) / p.white = 9 / 7 ∧
  p.black / (p.white - 1) = 7 / 5

/-- The theorem to be proved -/
theorem black_white_difference (p : PieceCount) :
  satisfiesCondition p → p.black - p.white = 7 := by
  sorry

end black_white_difference_l1672_167275


namespace seven_people_six_seats_l1672_167262

/-- The number of ways to seat 6 people from a group of 7 at a circular table -/
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  n * Nat.factorial (k - 1)

/-- Theorem stating the number of seating arrangements for 7 people at a circular table with 6 seats -/
theorem seven_people_six_seats :
  seating_arrangements 7 6 = 840 := by
  sorry

end seven_people_six_seats_l1672_167262


namespace prism_triangle_areas_sum_l1672_167265

/-- Represents a rectangular prism with dimensions a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the sum of areas of all triangles with vertices at corners of the prism -/
def sumTriangleAreas (prism : RectangularPrism) : ℝ :=
  sorry

/-- Represents the result of sumTriangleAreas as m + √n + √p -/
structure AreaSum where
  m : ℤ
  n : ℤ
  p : ℤ

/-- Converts the sum of triangle areas to the AreaSum form -/
def toAreaSum (sum : ℝ) : AreaSum :=
  sorry

theorem prism_triangle_areas_sum (prism : RectangularPrism) 
  (h1 : prism.a = 1) (h2 : prism.b = 1) (h3 : prism.c = 2) : 
  let sum := sumTriangleAreas prism
  let result := toAreaSum sum
  result.m + result.n + result.p = 41 :=
sorry

end prism_triangle_areas_sum_l1672_167265


namespace garden_breadth_calculation_l1672_167273

/-- Represents a rectangular garden with length and breadth -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.breadth)

theorem garden_breadth_calculation (garden : RectangularGarden) 
  (h1 : perimeter garden = 1800)
  (h2 : garden.length = 500) :
  garden.breadth = 400 := by
  sorry

end garden_breadth_calculation_l1672_167273


namespace tv_horizontal_length_l1672_167234

/-- Represents a rectangular TV screen -/
structure TVScreen where
  horizontal : ℝ
  vertical : ℝ
  diagonal : ℝ

/-- The TV screen satisfies the 16:9 aspect ratio and has a 36-inch diagonal -/
def is_valid_tv_screen (tv : TVScreen) : Prop :=
  tv.horizontal / tv.vertical = 16 / 9 ∧ 
  tv.diagonal = 36 ∧
  tv.diagonal^2 = tv.horizontal^2 + tv.vertical^2

/-- The theorem stating the horizontal length of the TV screen -/
theorem tv_horizontal_length (tv : TVScreen) 
  (h : is_valid_tv_screen tv) : 
  tv.horizontal = (16 * 36) / Real.sqrt 337 := by
  sorry

end tv_horizontal_length_l1672_167234


namespace julia_spent_114_on_animal_food_l1672_167289

/-- The total amount spent on animal food --/
def total_spent (weekly_total : ℕ) (rabbit_food_cost : ℕ) (rabbit_weeks : ℕ) (parrot_weeks : ℕ) : ℕ :=
  (weekly_total - rabbit_food_cost) * parrot_weeks + rabbit_food_cost * rabbit_weeks

/-- Proof that Julia spent $114 on animal food --/
theorem julia_spent_114_on_animal_food :
  total_spent 30 12 5 3 = 114 := by
  sorry

end julia_spent_114_on_animal_food_l1672_167289


namespace quadratic_discriminant_problem_l1672_167225

theorem quadratic_discriminant_problem (m : ℝ) : 
  ((-3)^2 - 4*1*(-m) = 13) → m = 1 := by
  sorry

end quadratic_discriminant_problem_l1672_167225


namespace orange_apple_pear_weight_equivalence_l1672_167236

/-- Represents the weight of a fruit -/
structure FruitWeight where
  weight : ℝ

/-- Represents the count of fruits -/
structure FruitCount where
  count : ℕ

/-- Given 9 oranges weigh the same as 6 apples and 1 pear, 
    prove that 36 oranges weigh the same as 24 apples and 4 pears -/
theorem orange_apple_pear_weight_equivalence 
  (orange : FruitWeight) 
  (apple : FruitWeight) 
  (pear : FruitWeight) 
  (h : 9 * orange.weight = 6 * apple.weight + pear.weight) : 
  36 * orange.weight = 24 * apple.weight + 4 * pear.weight := by
  sorry

end orange_apple_pear_weight_equivalence_l1672_167236


namespace initial_chicken_wings_l1672_167227

theorem initial_chicken_wings 
  (num_friends : ℕ) 
  (additional_wings : ℕ) 
  (wings_per_friend : ℕ) 
  (h1 : num_friends = 4) 
  (h2 : additional_wings = 7) 
  (h3 : wings_per_friend = 4) : 
  num_friends * wings_per_friend - additional_wings = 9 := by
sorry

end initial_chicken_wings_l1672_167227


namespace candidate_votes_l1672_167213

theorem candidate_votes (total_votes : ℕ) (invalid_percentage : ℚ) (candidate_percentage : ℚ) : 
  total_votes = 560000 →
  invalid_percentage = 15 / 100 →
  candidate_percentage = 75 / 100 →
  ⌊(total_votes : ℚ) * (1 - invalid_percentage) * candidate_percentage⌋ = 357000 := by
sorry

end candidate_votes_l1672_167213


namespace jacket_cost_calculation_l1672_167209

/-- The amount Mary spent on clothing -/
def total_spent : ℚ := 25.31

/-- The amount Mary spent on the shirt -/
def shirt_cost : ℚ := 13.04

/-- The number of shops Mary visited -/
def shops_visited : ℕ := 2

/-- The amount Mary spent on the jacket -/
def jacket_cost : ℚ := total_spent - shirt_cost

theorem jacket_cost_calculation : jacket_cost = 12.27 := by
  sorry

end jacket_cost_calculation_l1672_167209


namespace only_three_divides_2002_power_l1672_167205

theorem only_three_divides_2002_power : 
  ∀ p : ℕ, Prime p → p < 17 → (p ∣ 2002^2002 - 1) ↔ p = 3 := by
sorry

end only_three_divides_2002_power_l1672_167205


namespace fraction_transformation_l1672_167233

theorem fraction_transformation (a b : ℕ) (h : a < b) :
  (∃ x : ℕ, (a + x : ℚ) / (b + x) = 1 / 2) ∧
  (¬ ∃ y z : ℕ, ((a + y : ℚ) * z) / ((b + y) * z) = 1) := by
  sorry

end fraction_transformation_l1672_167233


namespace f_is_even_and_increasing_l1672_167298

def f (x : ℝ) := 10 * abs x

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
sorry

end f_is_even_and_increasing_l1672_167298


namespace pear_juice_percentage_l1672_167245

-- Define the juice production rates
def pear_juice_rate : ℚ := 10 / 4
def orange_juice_rate : ℚ := 12 / 3

-- Define the number of fruits used in the blend
def pears_in_blend : ℕ := 8
def oranges_in_blend : ℕ := 6

-- Define the total amount of juice in the blend
def total_juice : ℚ := pear_juice_rate * pears_in_blend + orange_juice_rate * oranges_in_blend

-- Define the amount of pear juice in the blend
def pear_juice_in_blend : ℚ := pear_juice_rate * pears_in_blend

-- Theorem statement
theorem pear_juice_percentage :
  (pear_juice_in_blend / total_juice) * 100 = 45 := by
  sorry

end pear_juice_percentage_l1672_167245


namespace remainder_divisibility_l1672_167228

theorem remainder_divisibility (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 := by
  sorry

end remainder_divisibility_l1672_167228


namespace square_point_selection_probability_square_point_selection_probability_is_three_fifths_l1672_167256

/-- The probability of selecting two points from the vertices and center of a square,
    such that their distance is not less than the side length of the square. -/
theorem square_point_selection_probability : ℚ :=
  let total_selections := (5 : ℕ).choose 2
  let favorable_selections := (4 : ℕ).choose 2
  (favorable_selections : ℚ) / total_selections

/-- The probability of selecting two points from the vertices and center of a square,
    such that their distance is not less than the side length of the square, is 3/5. -/
theorem square_point_selection_probability_is_three_fifths :
  square_point_selection_probability = 3 / 5 := by
  sorry

end square_point_selection_probability_square_point_selection_probability_is_three_fifths_l1672_167256


namespace richard_david_age_diff_l1672_167206

-- Define the ages of the three sons
def david_age : ℕ := 14
def scott_age : ℕ := 6
def richard_age : ℕ := 20

-- Define the conditions
axiom david_scott_age_diff : david_age = scott_age + 8
axiom david_past_age : david_age = 11 + 3
axiom richard_future_age : richard_age + 8 = 2 * (scott_age + 8)

-- Define the theorem to prove
theorem richard_david_age_diff : richard_age = david_age + 6 := by
  sorry

end richard_david_age_diff_l1672_167206


namespace mod_product_equality_l1672_167268

theorem mod_product_equality (m : ℕ) : 
  (256 * 738 ≡ m [ZMOD 75]) → 
  (0 ≤ m ∧ m < 75) → 
  m = 53 := by
sorry

end mod_product_equality_l1672_167268


namespace barbed_wire_rate_l1672_167250

/-- Given a square field with area 3136 sq m and a total cost of 865.80 for barbed wire
    (excluding two 1 m wide gates), the rate per meter of barbed wire is 3.90. -/
theorem barbed_wire_rate (area : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  area = 3136 →
  total_cost = 865.80 →
  gate_width = 1 →
  num_gates = 2 →
  (total_cost / (4 * Real.sqrt area - gate_width * num_gates)) = 3.90 := by
  sorry

end barbed_wire_rate_l1672_167250


namespace angle_abc_measure_l1672_167226

theorem angle_abc_measure (θ : ℝ) : 
  θ > 0 ∧ θ < 180 → -- Angle measure is positive and less than 180°
  (θ / 2) = (1 / 3) * (180 - θ) → -- Condition about angle bisector
  θ = 72 := by
sorry

end angle_abc_measure_l1672_167226


namespace smallest_factor_for_perfect_square_l1672_167257

def x : ℕ := 2^2 * 3^3 * 4^4 * 5^5 * 6^6 * 7^7 * 8^8 * 9^9

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

theorem smallest_factor_for_perfect_square :
  (∀ k : ℕ, k < 105 → ¬is_perfect_square (k * x)) ∧
  is_perfect_square (105 * x) :=
sorry

end smallest_factor_for_perfect_square_l1672_167257


namespace parallel_vectors_m_value_l1672_167200

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b in ℝ², if a is parallel to b and a = (m, 4) and b = (3, -2), then m = -6 -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  parallel a b → m = -6 := by
sorry

end parallel_vectors_m_value_l1672_167200


namespace greatest_power_less_than_500_l1672_167274

theorem greatest_power_less_than_500 (c d : ℕ+) (h1 : d > 1) 
  (h2 : c^(d:ℕ) < 500) 
  (h3 : ∀ (x y : ℕ+), y > 1 → x^(y:ℕ) < 500 → x^(y:ℕ) ≤ c^(d:ℕ)) : 
  c + d = 24 := by sorry

end greatest_power_less_than_500_l1672_167274


namespace star_difference_l1672_167264

def star (x y : ℝ) : ℝ := x * y - 3 * x + y

theorem star_difference : (star 6 5) - (star 5 6) = -4 := by
  sorry

end star_difference_l1672_167264


namespace polynomial_division_theorem_l1672_167271

-- Define the polynomial
def P (x a b : ℝ) : ℝ := 2*x^4 - 3*x^3 + a*x^2 + 7*x + b

-- Define the divisor
def D (x : ℝ) : ℝ := x^2 + x - 2

-- Theorem statement
theorem polynomial_division_theorem (a b : ℝ) :
  (∀ x, ∃ q, P x a b = D x * q) →
  a / b = -2 := by
  sorry

end polynomial_division_theorem_l1672_167271


namespace complex_subtraction_l1672_167299

theorem complex_subtraction (a b : ℂ) (ha : a = 6 - 3*I) (hb : b = 2 + 3*I) :
  a - 3*b = -12*I := by sorry

end complex_subtraction_l1672_167299


namespace cone_lateral_surface_area_l1672_167279

theorem cone_lateral_surface_area 
  (r h : ℝ) 
  (hr : r = 4) 
  (hh : h = 3) : 
  r * (Real.sqrt (r^2 + h^2)) * π = 20 * π := by
  sorry

end cone_lateral_surface_area_l1672_167279


namespace mathematicians_set_l1672_167258

-- Define the set of famous people
inductive FamousPerson
| BillGates
| Gauss
| YuanLongping
| Nobel
| ChenJingrun
| HuaLuogeng
| Gorky
| Einstein

-- Define a function to determine if a person is a mathematician
def isMathematician : FamousPerson → Prop
| FamousPerson.Gauss => True
| FamousPerson.ChenJingrun => True
| FamousPerson.HuaLuogeng => True
| _ => False

-- Define the set of all famous people
def allFamousPeople : Set FamousPerson :=
  {FamousPerson.BillGates, FamousPerson.Gauss, FamousPerson.YuanLongping,
   FamousPerson.Nobel, FamousPerson.ChenJingrun, FamousPerson.HuaLuogeng,
   FamousPerson.Gorky, FamousPerson.Einstein}

-- Theorem: The set of mathematicians is equal to {Gauss, Chen Jingrun, Hua Luogeng}
theorem mathematicians_set :
  {p ∈ allFamousPeople | isMathematician p} =
  {FamousPerson.Gauss, FamousPerson.ChenJingrun, FamousPerson.HuaLuogeng} :=
by sorry

end mathematicians_set_l1672_167258


namespace technician_permanent_percentage_l1672_167204

/-- Represents the composition of workers in a factory -/
structure Factory where
  total_workers : ℕ
  technicians : ℕ
  non_technicians : ℕ
  permanent_non_technicians : ℕ
  temporary_workers : ℕ

/-- The conditions of the factory -/
def factory_conditions (f : Factory) : Prop :=
  f.technicians = f.total_workers / 2 ∧
  f.non_technicians = f.total_workers / 2 ∧
  f.permanent_non_technicians = f.non_technicians / 2 ∧
  f.temporary_workers = f.total_workers / 2

/-- The theorem to be proved -/
theorem technician_permanent_percentage (f : Factory) 
  (h : factory_conditions f) : 
  (f.technicians - (f.temporary_workers - f.permanent_non_technicians)) * 2 = f.technicians := by
  sorry

#check technician_permanent_percentage

end technician_permanent_percentage_l1672_167204


namespace james_works_six_hours_l1672_167212

/-- Calculates the time James spends working on chores given the following conditions:
  * There are 3 bedrooms, 1 living room, and 2 bathrooms to clean
  * Bedrooms each take 20 minutes to clean
  * Living room takes as long as the 3 bedrooms combined
  * Bathroom takes twice as long as the living room
  * Outside cleaning takes twice as long as cleaning the house
  * Chores are split with 2 siblings who are just as fast -/
def james_working_time : ℕ :=
  let num_bedrooms : ℕ := 3
  let num_livingrooms : ℕ := 1
  let num_bathrooms : ℕ := 2
  let bedroom_cleaning_time : ℕ := 20
  let livingroom_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time
  let bathroom_cleaning_time : ℕ := 2 * livingroom_cleaning_time
  let inside_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time +
                                  num_livingrooms * livingroom_cleaning_time +
                                  num_bathrooms * bathroom_cleaning_time
  let outside_cleaning_time : ℕ := 2 * inside_cleaning_time
  let total_cleaning_time : ℕ := inside_cleaning_time + outside_cleaning_time
  let num_siblings : ℕ := 2
  let james_time_minutes : ℕ := total_cleaning_time / (num_siblings + 1)
  james_time_minutes / 60

theorem james_works_six_hours : james_working_time = 6 := by
  sorry

end james_works_six_hours_l1672_167212


namespace inequality_solution_set_l1672_167290

theorem inequality_solution_set (x : ℝ) : 
  (1 / (x^2 + 1) > 4 / x + 25 / 10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end inequality_solution_set_l1672_167290


namespace last_digit_of_sum_l1672_167272

theorem last_digit_of_sum (n : ℕ) : 
  (5^555 + 6^666 + 7^777) % 10 = 8 := by
  sorry

end last_digit_of_sum_l1672_167272


namespace initial_temperature_l1672_167237

theorem initial_temperature (T : ℝ) : (2 * T - 30) * 0.70 + 24 = 59 ↔ T = 40 := by
  sorry

end initial_temperature_l1672_167237


namespace cab_driver_average_income_l1672_167217

def daily_incomes : List ℝ := [45, 50, 60, 65, 70]
def num_days : ℕ := 5

theorem cab_driver_average_income : 
  (daily_incomes.sum / num_days : ℝ) = 58 := by
  sorry

end cab_driver_average_income_l1672_167217


namespace inequality_solution_set_l1672_167253

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > 4 / x + 21 / 10) ↔ (-2 < x ∧ x < 0) :=
sorry

end inequality_solution_set_l1672_167253


namespace common_internal_tangent_length_l1672_167284

theorem common_internal_tangent_length 
  (distance_between_centers : ℝ) 
  (radius1 : ℝ) 
  (radius2 : ℝ) 
  (h1 : distance_between_centers = 50) 
  (h2 : radius1 = 7) 
  (h3 : radius2 = 10) : 
  ∃ (tangent_length : ℝ), tangent_length = Real.sqrt 2211 := by
  sorry

end common_internal_tangent_length_l1672_167284


namespace negation_of_existence_sqrt_gt_three_l1672_167294

theorem negation_of_existence_sqrt_gt_three : 
  (¬ ∃ x : ℝ, Real.sqrt x > 3) ↔ (∀ x : ℝ, Real.sqrt x ≤ 3 ∨ x < 0) := by
  sorry

end negation_of_existence_sqrt_gt_three_l1672_167294


namespace sum_of_numbers_with_lcm_and_ratio_l1672_167207

/-- Given two positive integers with LCM 36 and ratio 2:3, prove their sum is 30 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 36)
  (h_ratio : a * 3 = b * 2) : 
  a + b = 30 := by
  sorry

end sum_of_numbers_with_lcm_and_ratio_l1672_167207


namespace unique_triplet_solution_l1672_167288

theorem unique_triplet_solution : 
  ∃! (m n k : ℕ), 3^n + 4^m = 5^k :=
by
  sorry

end unique_triplet_solution_l1672_167288


namespace coefficient_x3y5_in_binomial_expansion_l1672_167259

theorem coefficient_x3y5_in_binomial_expansion :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k : ℕ) * (1 : ℕ)^(8 - k) * (1 : ℕ)^k) = 256 ∧
  (Nat.choose 8 5 : ℕ) = 56 :=
sorry

end coefficient_x3y5_in_binomial_expansion_l1672_167259


namespace money_left_calculation_l1672_167201

/-- The amount of money John has left after purchasing pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_cost := 4 * drink_cost + small_pizza_cost + 2 * large_pizza_cost
  50 - total_cost

/-- Theorem stating that the money left is equal to 50 - 13q -/
theorem money_left_calculation (q : ℝ) : money_left q = 50 - 13 * q := by
  sorry

end money_left_calculation_l1672_167201


namespace planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l1672_167222

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_plane_to_plane : Plane → Plane → Prop)
variable (perpendicular_plane_to_line : Plane → Line → Prop)

-- State the theorems
theorem planes_parallel_to_same_plane_are_parallel
  (p1 p2 p3 : Plane)
  (h1 : parallel_plane_to_plane p1 p3)
  (h2 : parallel_plane_to_plane p2 p3) :
  parallel_planes p1 p2 :=
sorry

theorem planes_perpendicular_to_same_line_are_parallel
  (p1 p2 : Plane) (l : Line)
  (h1 : perpendicular_plane_to_line p1 l)
  (h2 : perpendicular_plane_to_line p2 l) :
  parallel_planes p1 p2 :=
sorry

end planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l1672_167222


namespace solution_satisfies_system_l1672_167241

theorem solution_satisfies_system :
  ∃ (x y z : ℝ), 
    (3 * x + 2 * y - z = 1) ∧
    (4 * x - 5 * y + 3 * z = 11) ∧
    (x = 1 ∧ y = 1 ∧ z = 4) :=
by
  sorry

end solution_satisfies_system_l1672_167241


namespace a_faster_than_b_l1672_167235

/-- Represents a person sawing wood --/
structure Sawyer where
  name : String
  segments_per_piece : ℕ
  total_segments : ℕ

/-- Calculates the number of pieces sawed by a sawyer --/
def pieces_sawed (s : Sawyer) : ℕ := s.total_segments / s.segments_per_piece

/-- Calculates the number of cuts made by a sawyer --/
def cuts_made (s : Sawyer) : ℕ := (s.segments_per_piece - 1) * (pieces_sawed s)

/-- Theorem stating that A takes less time to saw one piece of wood --/
theorem a_faster_than_b (a b : Sawyer)
  (ha : a.name = "A" ∧ a.segments_per_piece = 3 ∧ a.total_segments = 24)
  (hb : b.name = "B" ∧ b.segments_per_piece = 2 ∧ b.total_segments = 28) :
  cuts_made a > cuts_made b := by sorry

end a_faster_than_b_l1672_167235


namespace part_I_part_II_l1672_167203

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → m^2 - 3*m + x - 1 ≤ 0

def q (m a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ m - a*x ≤ 0

-- Part I
theorem part_I : 
  ∃ S : Set ℝ, S = {m : ℝ | (m < 1 ∨ (1 < m ∧ m ≤ 2)) ∧ 
  ((p m ∧ ¬q m 1) ∨ (¬p m ∧ q m 1))} := by sorry

-- Part II
theorem part_II : 
  ∃ S : Set ℝ, S = {a : ℝ | a ≥ 2 ∨ a ≤ -2} ∧
  ∀ m : ℝ, (p m → q m a) ∧ ¬(q m a → p m) := by sorry

end part_I_part_II_l1672_167203


namespace arithmetic_geometric_sequence_l1672_167277

-- Define the arithmetic sequence
def a (n : ℕ) : ℚ := n + 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a (n + 1) / a n + a n / a (n + 1) - 2

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := n / (n + 1)

theorem arithmetic_geometric_sequence :
  (a 4 - a 2 = 2) ∧ 
  (a 3 * a 3 = a 1 * a 7) →
  (∀ n : ℕ, a n = n + 1) ∧
  (∀ n : ℕ, S n = n / (n + 1)) :=
by sorry

end arithmetic_geometric_sequence_l1672_167277


namespace toaster_msrp_l1672_167247

/-- The MSRP of a toaster given specific conditions -/
theorem toaster_msrp (x : ℝ) : 
  x + 0.2 * x + 0.5 * (x + 0.2 * x) = 54 → x = 30 := by
  sorry

end toaster_msrp_l1672_167247


namespace no_nonzero_integer_solution_l1672_167246

theorem no_nonzero_integer_solution (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end no_nonzero_integer_solution_l1672_167246


namespace abc_def_ratio_l1672_167278

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6) :
  a * b * c / (d * e * f) = 1 / 36 := by
  sorry

end abc_def_ratio_l1672_167278


namespace increasing_interval_transformed_l1672_167267

-- Define an even function f
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define an increasing function on an interval
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

-- Main theorem
theorem increasing_interval_transformed (f : ℝ → ℝ) :
  even_function f →
  increasing_on f 2 6 →
  increasing_on (fun x ↦ f (2 - x)) 4 8 :=
sorry

end increasing_interval_transformed_l1672_167267


namespace parabola_m_range_l1672_167251

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadratic function of the form y = ax² + 4ax + c -/
structure QuadraticFunction where
  a : ℝ
  c : ℝ
  h : a ≠ 0

theorem parabola_m_range 
  (f : QuadraticFunction)
  (A B C : Point)
  (h1 : A.x = m)
  (h2 : B.x = m + 2)
  (h3 : C.x = -2)  -- vertex x-coordinate for y = ax² + 4ax + c is always -2
  (h4 : A.y = f.a * A.x^2 + 4 * f.a * A.x + f.c)
  (h5 : B.y = f.a * B.x^2 + 4 * f.a * B.x + f.c)
  (h6 : C.y = f.a * C.x^2 + 4 * f.a * C.x + f.c)
  (h7 : C.y ≥ B.y)
  (h8 : B.y > A.y)
  : m < -3 := by sorry

end parabola_m_range_l1672_167251


namespace nursery_school_fraction_l1672_167243

/-- Given a nursery school with the following conditions:
  1. 20 students are under 3 years old
  2. 50 students are not between 3 and 4 years old
  3. There are 300 children in total
  Prove that the fraction of students who are 4 years old or older is 1/10 -/
theorem nursery_school_fraction (under_three : ℕ) (not_between_three_and_four : ℕ) (total : ℕ)
  (h1 : under_three = 20)
  (h2 : not_between_three_and_four = 50)
  (h3 : total = 300) :
  (not_between_three_and_four - under_three) / total = 1 / 10 := by
  sorry

end nursery_school_fraction_l1672_167243


namespace scientific_notation_439000_l1672_167208

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_439000 :
  toScientificNotation 439000 = ScientificNotation.mk 4.39 5 (by norm_num) :=
sorry

end scientific_notation_439000_l1672_167208


namespace problem1_solution_problem2_solution_l1672_167249

-- Problem 1
def problem1 (a : ℚ) : ℚ := a * (a - 4) - (a + 6) * (a - 2)

theorem problem1_solution :
  problem1 (-1/2) = 16 := by sorry

-- Problem 2
def problem2 (x y : ℤ) : ℤ := (x + 2*y) * (x - 2*y) - (2*x - y) * (-2*x - y)

theorem problem2_solution :
  problem2 8 (-8) = 0 := by sorry

end problem1_solution_problem2_solution_l1672_167249


namespace sum_of_a_and_b_l1672_167248

theorem sum_of_a_and_b (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a + b = -1 := by
  sorry

end sum_of_a_and_b_l1672_167248


namespace max_product_sum_l1672_167224

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∃ (q : ℕ), q = A * M * C + A * M + M * C + C * A ∧
   ∀ (q' : ℕ), q' = A * M * C + A * M + M * C + C * A → q' ≤ q) ∧
  (A * M * C + A * M + M * C + C * A ≤ 200) :=
by sorry

end max_product_sum_l1672_167224


namespace young_li_age_is_20_l1672_167296

/-- Young Li's age this year -/
def young_li_age : ℕ := 20

/-- Old Li's age this year -/
def old_li_age : ℕ := young_li_age * 5 / 2

theorem young_li_age_is_20 :
  (old_li_age = young_li_age * 5 / 2) ∧
  (old_li_age + 10 = (young_li_age + 10) * 2) →
  young_li_age = 20 :=
by sorry

end young_li_age_is_20_l1672_167296


namespace train_length_l1672_167261

/-- Given a train with a speed of 125.99999999999999 km/h that can cross an electric pole in 20 seconds,
    prove that the length of the train is 700 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 125.99999999999999 → 
  time = 20 → 
  length = speed * (1000 / 3600) * time → 
  length = 700 := by sorry

end train_length_l1672_167261


namespace x_plus_z_equals_15_l1672_167230

theorem x_plus_z_equals_15 (x y z : ℝ) 
  (h1 : |x| + x + z = 15) 
  (h2 : x + |y| - y = 8) : 
  x + z = 15 := by
sorry

end x_plus_z_equals_15_l1672_167230


namespace circle_center_transformation_l1672_167270

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (3, -4)
  let reflected_x := reflect_x initial_center
  let reflected_y := reflect_y reflected_x
  let final_center := translate reflected_y 5 3
  final_center = (2, 7) := by sorry

end circle_center_transformation_l1672_167270


namespace bathing_suit_sets_proof_l1672_167215

-- Define the constants from the problem
def total_time : ℕ := 60
def runway_time : ℕ := 2
def num_models : ℕ := 6
def evening_wear_sets : ℕ := 3

-- Define the function to calculate bathing suit sets per model
def bathing_suit_sets_per_model : ℕ :=
  let total_evening_wear_time := num_models * evening_wear_sets * runway_time
  let remaining_time := total_time - total_evening_wear_time
  let total_bathing_suit_trips := remaining_time / runway_time
  total_bathing_suit_trips / num_models

-- Theorem statement
theorem bathing_suit_sets_proof :
  bathing_suit_sets_per_model = 2 :=
by sorry

end bathing_suit_sets_proof_l1672_167215


namespace set_union_problem_l1672_167214

theorem set_union_problem (a b : ℝ) :
  let A : Set ℝ := {-1, a}
  let B : Set ℝ := {2^a, b}
  A ∩ B = {1} → A ∪ B = {-1, 1, 2} := by
  sorry

end set_union_problem_l1672_167214


namespace quadrilateral_angles_l1672_167280

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties of the quadrilateral
def is_convex_quadrilateral (q : Quadrilateral) : Prop := sorry

def angle_ABD (q : Quadrilateral) : ℝ := sorry
def angle_CBD (q : Quadrilateral) : ℝ := sorry
def angle_ADC (q : Quadrilateral) : ℝ := sorry

def AB_equals_BC (q : Quadrilateral) : Prop := sorry

def angle_A (q : Quadrilateral) : ℝ := sorry
def angle_C (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_angles 
  (q : Quadrilateral) 
  (h_convex : is_convex_quadrilateral q)
  (h_ABD : angle_ABD q = 65)
  (h_CBD : angle_CBD q = 35)
  (h_ADC : angle_ADC q = 130)
  (h_AB_BC : AB_equals_BC q) :
  angle_A q = 57.5 ∧ angle_C q = 72.5 :=
sorry

end quadrilateral_angles_l1672_167280


namespace inequality_solution_set_l1672_167276

theorem inequality_solution_set :
  let S := {x : ℝ | (x + 1/2) * (3/2 - x) ≥ 0}
  S = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/2} := by sorry

end inequality_solution_set_l1672_167276


namespace inequality_solution_set_l1672_167242

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_inc : ∀ x y, x < y → x ∈ [-1, 1] → y ∈ [-1, 1] → f x < f y)
  (h_dom : ∀ x, f x ≠ 0 → x ∈ [-1, 1]) :
  {x : ℝ | f (x - 1/2) + f (1/4 - x) < 0} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 1} := by
  sorry

end inequality_solution_set_l1672_167242


namespace sphere_volume_ratio_l1672_167292

theorem sphere_volume_ratio (r : ℝ) (hr : r > 0) : 
  (4 / 3 * Real.pi * (3 * r)^3) = 3 * ((4 / 3 * Real.pi * r^3) + (4 / 3 * Real.pi * (2 * r)^3)) := by
  sorry

end sphere_volume_ratio_l1672_167292


namespace art_book_cost_l1672_167219

theorem art_book_cost (total_cost : ℕ) (num_math num_art num_science : ℕ) (cost_math cost_science : ℕ) :
  total_cost = 30 ∧
  num_math = 2 ∧
  num_art = 3 ∧
  num_science = 6 ∧
  cost_math = 3 ∧
  cost_science = 3 →
  (total_cost - num_math * cost_math - num_science * cost_science) / num_art = 2 :=
by sorry

end art_book_cost_l1672_167219
