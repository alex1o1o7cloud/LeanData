import Mathlib

namespace vector_perpendicular_value_l3542_354260

-- Define the vectors
def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the perpendicularity condition
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Theorem statement
theorem vector_perpendicular_value (x : ℝ) :
  perpendicular a (a.1 - (b x).1, a.2 - (b x).2) → x = 1/2 := by
  sorry

end vector_perpendicular_value_l3542_354260


namespace certain_number_proof_l3542_354233

/-- Given that g is the smallest positive integer such that n * g is a perfect square, 
    and g = 14, prove that n = 14 -/
theorem certain_number_proof (n : ℕ) (g : ℕ) (h1 : g = 14) 
  (h2 : ∃ m : ℕ, n * g = m^2)
  (h3 : ∀ k < g, ¬∃ m : ℕ, n * k = m^2) : n = 14 := by
  sorry

end certain_number_proof_l3542_354233


namespace tax_free_amount_correct_l3542_354261

/-- The tax-free amount for goods purchased in country B -/
def tax_free_amount : ℝ := 600

/-- The total value of goods purchased -/
def total_value : ℝ := 1720

/-- The tax rate applied to the portion exceeding the tax-free amount -/
def tax_rate : ℝ := 0.08

/-- The amount of tax paid -/
def tax_paid : ℝ := 89.6

/-- Theorem stating that the tax-free amount satisfies the given conditions -/
theorem tax_free_amount_correct : 
  tax_rate * (total_value - tax_free_amount) = tax_paid := by sorry

end tax_free_amount_correct_l3542_354261


namespace sheet_width_calculation_l3542_354259

/-- Proves that a sheet with given dimensions and margins has a width of 20 cm when 64% is used for typing -/
theorem sheet_width_calculation (w : ℝ) : 
  w > 0 ∧ 
  (w - 4) * 24 = 0.64 * w * 30 → 
  w = 20 := by
  sorry

end sheet_width_calculation_l3542_354259


namespace odd_numbers_perfect_square_l3542_354275

/-- Sum of first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- The n-th odd number -/
def nthOddNumber (n : ℕ) : ℕ := 2*n - 1

theorem odd_numbers_perfect_square (K : ℕ) :
  K % 2 = 1 →  -- K is odd
  (∃ (N : ℕ), N < 50 ∧ sumOddNumbers N = N^2 ∧ nthOddNumber N = K) →
  1 ≤ K ∧ K ≤ 97 :=
by sorry

end odd_numbers_perfect_square_l3542_354275


namespace average_of_xyz_l3542_354289

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) :
  (x + y + z) / 3 = 16 / 3 := by
  sorry

end average_of_xyz_l3542_354289


namespace inequality_proof_l3542_354270

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) := by
  sorry

end inequality_proof_l3542_354270


namespace complex_arithmetic_expression_l3542_354200

theorem complex_arithmetic_expression : -1^2009 * (-3) + 1 - 2^2 * 3 + (1 - 2^2) / 3 + (1 - 2 * 3)^2 = 16 := by
  sorry

end complex_arithmetic_expression_l3542_354200


namespace calculation_product_l3542_354255

theorem calculation_product (x : ℤ) (h : x - 9 - 12 = 24) : (x + 8 - 11) * 24 = 1008 := by
  sorry

end calculation_product_l3542_354255


namespace central_region_perimeter_l3542_354235

/-- The perimeter of the central region formed by four identical circles in a square formation --/
theorem central_region_perimeter (c : ℝ) (h : c = 48) : 
  let r := c / (2 * Real.pi)
  4 * (Real.pi * r / 2) = c :=
by sorry

end central_region_perimeter_l3542_354235


namespace expression_evaluation_l3542_354285

theorem expression_evaluation (x : ℝ) (h : x = 1) : 
  (x - 1)^2 + (x + 1)*(x - 1) - 2*x^2 = -2 := by
  sorry

end expression_evaluation_l3542_354285


namespace inequality_proof_l3542_354267

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  3 + a / b + b / c + c / a ≥ a + b + c + 1 / a + 1 / b + 1 / c := by
  sorry

end inequality_proof_l3542_354267


namespace die_roll_probability_l3542_354269

def standard_die_roll := Fin 6

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_five (roll : standard_die_roll) : Prop := roll.val + 1 = 5

def probability_odd_roll : ℚ := 1/2

def probability_not_five : ℚ := 5/6

def num_rolls : ℕ := 8

theorem die_roll_probability :
  (probability_odd_roll ^ num_rolls) * (1 - probability_not_five ^ num_rolls) = 1288991/429981696 := by
  sorry

end die_roll_probability_l3542_354269


namespace cubic_roots_sum_l3542_354254

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 3*p - 2 = 0) → 
  (q^3 - 3*q - 2 = 0) → 
  (r^3 - 3*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 0 := by
sorry

end cubic_roots_sum_l3542_354254


namespace eggs_removed_l3542_354218

theorem eggs_removed (original : ℕ) (remaining : ℕ) (removed : ℕ) : 
  original = 27 → remaining = 20 → removed = original - remaining → removed = 7 := by
sorry

end eggs_removed_l3542_354218


namespace tangent_line_coefficients_l3542_354224

/-- Given a curve y = x^2 + ax + b with a tangent line at (1, b) with equation x - y + 1 = 0,
    prove that a = -1 and b = 2 -/
theorem tangent_line_coefficients (a b : ℝ) : 
  (∀ x y : ℝ, y = x^2 + a*x + b) →
  (∃ y : ℝ, y = 1^2 + a*1 + b) →
  (∀ x y : ℝ, y = 1^2 + a*1 + b → x - y + 1 = 0 → x = 1) →
  a = -1 ∧ b = 2 := by
  sorry

end tangent_line_coefficients_l3542_354224


namespace annual_savings_20_over_30_l3542_354295

/-- Represents the internet plans and their costs -/
structure InternetPlan where
  speed : ℕ  -- Speed in Mbps
  monthlyCost : ℕ  -- Monthly cost in dollars

/-- Calculates the annual cost of an internet plan -/
def annualCost (plan : InternetPlan) : ℕ :=
  plan.monthlyCost * 12

/-- Represents Marites' internet plans -/
def marites : {currentPlan : InternetPlan // currentPlan.speed = 10 ∧ currentPlan.monthlyCost = 20} :=
  ⟨⟨10, 20⟩, by simp⟩

/-- The 30 Mbps plan -/
def plan30 : InternetPlan :=
  ⟨30, 2 * marites.val.monthlyCost⟩

/-- The 20 Mbps plan -/
def plan20 : InternetPlan :=
  ⟨20, marites.val.monthlyCost + 10⟩

/-- Theorem: Annual savings when choosing 20 Mbps over 30 Mbps is $120 -/
theorem annual_savings_20_over_30 :
  annualCost plan30 - annualCost plan20 = 120 := by
  sorry

end annual_savings_20_over_30_l3542_354295


namespace problem_statement_l3542_354294

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + x^2 / y + y^2 / x + y = 95 / 3 := by
  sorry

end problem_statement_l3542_354294


namespace scientific_notation_120000_l3542_354277

theorem scientific_notation_120000 :
  (120000 : ℝ) = 1.2 * (10 ^ 5) :=
sorry

end scientific_notation_120000_l3542_354277


namespace slope_of_line_with_30_degree_inclination_l3542_354239

theorem slope_of_line_with_30_degree_inclination :
  let angle_of_inclination : ℝ := 30 * π / 180
  let slope : ℝ := Real.tan angle_of_inclination
  slope = Real.sqrt 3 / 3 := by sorry

end slope_of_line_with_30_degree_inclination_l3542_354239


namespace line_quadrants_l3542_354265

theorem line_quadrants (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c > 0) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (a * x₁ + b * y₁ - c = 0 ∧ x₁ > 0 ∧ y₁ > 0) ∧  -- First quadrant
    (a * x₂ + b * y₂ - c = 0 ∧ x₂ < 0 ∧ y₂ < 0) ∧  -- Third quadrant
    (a * x₃ + b * y₃ - c = 0 ∧ x₃ > 0 ∧ y₃ < 0) :=  -- Fourth quadrant
by
  sorry

end line_quadrants_l3542_354265


namespace mark_fruit_count_l3542_354231

/-- The number of apples Mark has chosen -/
def num_apples : ℕ := 3

/-- The number of bananas in the bunch Mark has selected -/
def num_bananas : ℕ := 4

/-- The number of oranges Mark needs to pick out -/
def num_oranges : ℕ := 5

/-- The total number of pieces of fruit Mark is looking to buy -/
def total_fruit : ℕ := num_apples + num_bananas + num_oranges

theorem mark_fruit_count : total_fruit = 12 := by
  sorry

end mark_fruit_count_l3542_354231


namespace sons_age_l3542_354206

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end sons_age_l3542_354206


namespace x_remaining_time_l3542_354297

-- Define the work rates and time worked
def x_rate : ℚ := 1 / 20
def y_rate : ℚ := 1 / 15
def y_time_worked : ℚ := 9

-- Define the total work as 1 (representing 100%)
def total_work : ℚ := 1

-- Theorem statement
theorem x_remaining_time (x_rate : ℚ) (y_rate : ℚ) (y_time_worked : ℚ) (total_work : ℚ) :
  x_rate = 1 / 20 →
  y_rate = 1 / 15 →
  y_time_worked = 9 →
  total_work = 1 →
  (total_work - y_rate * y_time_worked) / x_rate = 8 :=
by
  sorry


end x_remaining_time_l3542_354297


namespace pentagon_section_probability_l3542_354204

/-- The probability of an arrow stopping in a specific section of a pentagon divided into 5 equal sections is 1/5. -/
theorem pentagon_section_probability :
  ∀ (n : ℕ) (sections : ℕ),
    sections = 5 →
    n ≤ sections →
    (n : ℚ) / (sections : ℚ) = 1 / 5 :=
by sorry

end pentagon_section_probability_l3542_354204


namespace truncated_tetrahedron_edge_count_l3542_354203

/-- A tetrahedron with truncated vertices -/
structure TruncatedTetrahedron where
  /-- The number of truncated vertices -/
  truncatedVertices : ℕ
  /-- Assertion that all vertices are truncated -/
  all_truncated : truncatedVertices = 4
  /-- Assertion that truncations are distinct and non-intersecting -/
  distinct_truncations : True

/-- The number of edges in a truncated tetrahedron -/
def edgeCount (t : TruncatedTetrahedron) : ℕ := sorry

/-- Theorem stating that a truncated tetrahedron has 18 edges -/
theorem truncated_tetrahedron_edge_count (t : TruncatedTetrahedron) : 
  edgeCount t = 18 := by sorry

end truncated_tetrahedron_edge_count_l3542_354203


namespace train_average_speed_l3542_354215

theorem train_average_speed :
  let distance1 : ℝ := 290
  let time1 : ℝ := 4.5
  let distance2 : ℝ := 400
  let time2 : ℝ := 5.5
  let total_distance : ℝ := distance1 + distance2
  let total_time : ℝ := time1 + time2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 69 := by sorry

end train_average_speed_l3542_354215


namespace smaller_number_proof_l3542_354273

theorem smaller_number_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x - y = 1650) (h4 : 0.075 * x = 0.125 * y) : y = 2475 := by
  sorry

end smaller_number_proof_l3542_354273


namespace min_value_theorem_l3542_354232

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_condition : a + b + c = 3) :
  ∃ (min_value : ℝ), min_value = 9 ∧ 
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
  x^2 + y^2 + (x + y)^2 + z^2 ≥ min_value :=
sorry

end min_value_theorem_l3542_354232


namespace min_value_of_sum_absolute_differences_l3542_354202

theorem min_value_of_sum_absolute_differences (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, |x - 2| + |x - 3| + |x - 4| < a) → a > 3 :=
by sorry

end min_value_of_sum_absolute_differences_l3542_354202


namespace monomial_degree_implies_a_value_l3542_354227

/-- Given that (a-2)x^2y^(|a|+1) is a monomial of degree 5 in x and y, prove that a = -2 -/
theorem monomial_degree_implies_a_value (a : ℤ) : 
  (∃ (x y : ℚ), (a - 2) * x^2 * y^(|a| + 1) ≠ 0) ∧ 
  (2 + (|a| + 1) = 5) → 
  a = -2 := by
sorry

end monomial_degree_implies_a_value_l3542_354227


namespace max_sum_xyz_l3542_354222

theorem max_sum_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 :=
sorry

end max_sum_xyz_l3542_354222


namespace trip_distance_is_150_l3542_354216

/-- Represents the problem of determining the trip distance --/
def TripDistance (D : ℝ) : Prop :=
  let rental_cost_1 : ℝ := 50
  let rental_cost_2 : ℝ := 90
  let gas_efficiency : ℝ := 15  -- km per liter
  let gas_cost : ℝ := 0.9  -- dollars per liter
  let cost_difference : ℝ := 22
  rental_cost_1 + (2 * D / gas_efficiency * gas_cost) = rental_cost_2 - cost_difference

/-- The theorem stating that the trip distance each way is 150 km --/
theorem trip_distance_is_150 : TripDistance 150 := by
  sorry

end trip_distance_is_150_l3542_354216


namespace restaurant_spend_l3542_354293

/-- The total amount spent by a group at a restaurant -/
def total_spent (n : ℕ) (individual_spends : Fin n → ℚ) : ℚ :=
  (Finset.univ.sum fun i => individual_spends i)

/-- The average expenditure of a group -/
def average_spend (n : ℕ) (individual_spends : Fin n → ℚ) : ℚ :=
  (total_spent n individual_spends) / n

theorem restaurant_spend :
  ∀ (individual_spends : Fin 8 → ℚ),
  (∀ i : Fin 7, individual_spends i = 10) →
  (individual_spends 7 = average_spend 8 individual_spends + 7) →
  total_spent 8 individual_spends = 88 := by
sorry

end restaurant_spend_l3542_354293


namespace candy_mixture_price_l3542_354262

theorem candy_mixture_price (total_mixture : ℝ) (mixture_price : ℝ) 
  (first_candy_amount : ℝ) (second_candy_amount : ℝ) (first_candy_price : ℝ) :
  total_mixture = 30 ∧
  mixture_price = 3 ∧
  first_candy_amount = 20 ∧
  second_candy_amount = 10 ∧
  first_candy_price = 2.95 →
  ∃ (second_candy_price : ℝ),
    second_candy_price = 3.10 ∧
    total_mixture * mixture_price = 
      first_candy_amount * first_candy_price + second_candy_amount * second_candy_price :=
by
  sorry

end candy_mixture_price_l3542_354262


namespace sqrt_difference_simplification_l3542_354245

theorem sqrt_difference_simplification (x : ℝ) (h : -1 < x ∧ x < 0) :
  Real.sqrt (x^2) - Real.sqrt ((x+1)^2) = -2*x - 1 := by
  sorry

end sqrt_difference_simplification_l3542_354245


namespace largest_n_satisfying_inequality_l3542_354220

theorem largest_n_satisfying_inequality :
  ∃ (n : ℕ), n^300 < 3^500 ∧ ∀ (m : ℕ), m^300 < 3^500 → m ≤ n :=
by
  -- The proof goes here
  sorry

end largest_n_satisfying_inequality_l3542_354220


namespace eight_jaguars_arrangement_l3542_354278

/-- The number of ways to arrange n different objects in a line -/
def linearArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n different objects in a line with the largest and smallest at the ends -/
def arrangementsWithExtremes (n : ℕ) : ℕ :=
  2 * linearArrangements (n - 2)

/-- Theorem: There are 1440 ways to arrange 8 different objects in a line with the largest and smallest at the ends -/
theorem eight_jaguars_arrangement :
  arrangementsWithExtremes 8 = 1440 := by
  sorry

end eight_jaguars_arrangement_l3542_354278


namespace inequality_implies_a_bounds_l3542_354201

-- Define the operation ⊕
def circleplus (x y : ℝ) : ℝ := (x + 3) * (y - 1)

-- State the theorem
theorem inequality_implies_a_bounds :
  (∀ x : ℝ, circleplus (x - a) (x + a) > -16) → -2 < a ∧ a < 6 :=
by sorry

end inequality_implies_a_bounds_l3542_354201


namespace paint_problem_solution_l3542_354208

def paint_problem (total_paint : ℚ) (second_week_fraction : ℚ) (total_used : ℚ) (first_week_fraction : ℚ) : Prop :=
  total_paint > 0 ∧
  second_week_fraction > 0 ∧ second_week_fraction < 1 ∧
  total_used > 0 ∧ total_used < total_paint ∧
  first_week_fraction > 0 ∧ first_week_fraction < 1 ∧
  first_week_fraction * total_paint + second_week_fraction * (total_paint - first_week_fraction * total_paint) = total_used

theorem paint_problem_solution :
  paint_problem 360 (1/3) 180 (1/4) := by
  sorry

end paint_problem_solution_l3542_354208


namespace problem_solution_l3542_354243

def f (x : ℝ) : ℝ := 2 * x^2 + 7

def g (x : ℝ) : ℝ := x^3 - 4

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 23) : 
  a = (2 * Real.sqrt 2 + 4) ^ (1/3) := by
  sorry

end problem_solution_l3542_354243


namespace largest_value_when_x_is_quarter_l3542_354274

theorem largest_value_when_x_is_quarter (x : ℝ) (h : x = 1/4) : 
  (1/x > x) ∧ (1/x > x^2) ∧ (1/x > (1/2)*x) ∧ (1/x > Real.sqrt x) := by
  sorry

end largest_value_when_x_is_quarter_l3542_354274


namespace tan_alpha_two_l3542_354256

theorem tan_alpha_two (α : Real) (h : Real.tan α = 2) : 
  Real.tan (2 * α + π / 4) = 9 ∧ (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13/4 := by
  sorry

end tan_alpha_two_l3542_354256


namespace sqrt_two_expression_l3542_354205

theorem sqrt_two_expression : Real.sqrt 2 * (Real.sqrt 2 + 2) - |Real.sqrt 2 - 2| = 3 * Real.sqrt 2 := by
  sorry

end sqrt_two_expression_l3542_354205


namespace new_average_production_l3542_354266

theorem new_average_production (n : ℕ) (past_avg : ℝ) (today_prod : ℝ) :
  n = 9 →
  past_avg = 50 →
  today_prod = 100 →
  (n * past_avg + today_prod) / (n + 1) = 55 :=
by sorry

end new_average_production_l3542_354266


namespace no_integer_solution_l3542_354264

theorem no_integer_solution : ¬∃ (x : ℝ), 
  (∃ (a b c : ℤ), (x - 1/x = a) ∧ (1/x - 1/(x^2 + 1) = b) ∧ (1/(x^2 + 1) - 2*x = c)) :=
by sorry

end no_integer_solution_l3542_354264


namespace opposite_of_negative_2023_l3542_354241

theorem opposite_of_negative_2023 : 
  (-(- 2023 : ℤ)) = (2023 : ℤ) := by sorry

end opposite_of_negative_2023_l3542_354241


namespace ball_diameter_l3542_354211

theorem ball_diameter (h : Real) (d : Real) (r : Real) : 
  h = 2 → d = 8 → r^2 = (d/2)^2 + (r - h)^2 → 2*r = 10 :=
by
  sorry

end ball_diameter_l3542_354211


namespace compound_interest_calculation_l3542_354257

/-- Given a principal amount where the simple interest for 2 years at 5% per annum is 52,
    prove that the compound interest at 5% per annum for 2 years is 53.30 -/
theorem compound_interest_calculation (P : ℝ) : 
  (P * 5 * 2) / 100 = 52 →
  P * ((1 + 5/100)^2 - 1) = 53.30 := by
sorry

end compound_interest_calculation_l3542_354257


namespace point_opposite_sides_range_l3542_354281

/-- Determines if two points are on opposite sides of a line -/
def oppositeSides (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  (a * x1 + b * y1 + c) * (a * x2 + b * y2 + c) < 0

theorem point_opposite_sides_range (a : ℝ) :
  oppositeSides 3 1 (-4) 6 3 (-2) a ↔ -7 < a ∧ a < 24 := by
  sorry

end point_opposite_sides_range_l3542_354281


namespace remainder_double_number_l3542_354292

theorem remainder_double_number (N : ℤ) : 
  N % 398 = 255 → (2 * N) % 398 = 112 := by
sorry

end remainder_double_number_l3542_354292


namespace initial_bacteria_count_l3542_354225

/-- The number of bacteria after a given number of doubling periods -/
def bacteria_count (initial_count : ℕ) (periods : ℕ) : ℕ :=
  initial_count * 2^periods

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ),
    bacteria_count initial_count 8 = 262144 ∧
    initial_count = 1024 :=
by
  sorry

end initial_bacteria_count_l3542_354225


namespace factorization_x4_minus_16y4_l3542_354217

theorem factorization_x4_minus_16y4 (x y : ℚ) : 
  x^4 - 16*y^4 = (x^2 + 4*y^2)*(x + 2*y)*(x - 2*y) := by
  sorry

end factorization_x4_minus_16y4_l3542_354217


namespace intersection_A_B_union_complement_A_B_l3542_354276

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | -1 < x ∧ x < 1 ∨ 2 < x ∧ x < 3} := by sorry

-- Theorem for (C_U A) ∪ B
theorem union_complement_A_B : (Set.compl A) ∪ B = {x | x < 1 ∨ x > 2} := by sorry

end intersection_A_B_union_complement_A_B_l3542_354276


namespace expected_profit_is_37_l3542_354219

/-- Represents the possible product grades produced by the machine -/
inductive ProductGrade
  | GradeA
  | GradeB
  | Defective

/-- Returns the profit for a given product grade -/
def profit (grade : ProductGrade) : ℝ :=
  match grade with
  | ProductGrade.GradeA => 50
  | ProductGrade.GradeB => 30
  | ProductGrade.Defective => -20

/-- Returns the probability of producing a given product grade -/
def probability (grade : ProductGrade) : ℝ :=
  match grade with
  | ProductGrade.GradeA => 0.6
  | ProductGrade.GradeB => 0.3
  | ProductGrade.Defective => 0.1

/-- Calculates the expected profit -/
def expectedProfit : ℝ :=
  (profit ProductGrade.GradeA * probability ProductGrade.GradeA) +
  (profit ProductGrade.GradeB * probability ProductGrade.GradeB) +
  (profit ProductGrade.Defective * probability ProductGrade.Defective)

theorem expected_profit_is_37 : expectedProfit = 37 := by
  sorry

end expected_profit_is_37_l3542_354219


namespace fraction_product_l3542_354286

theorem fraction_product : (2 : ℚ) / 9 * (4 : ℚ) / 5 = (8 : ℚ) / 45 := by
  sorry

end fraction_product_l3542_354286


namespace homework_time_difference_prove_homework_time_difference_l3542_354242

/-- The difference in time taken to finish homework between Sarah and Samuel is 48 minutes -/
theorem homework_time_difference : ℝ → Prop :=
  fun difference =>
    let samuel_time : ℝ := 30  -- Samuel's time in minutes
    let sarah_time : ℝ := 1.3 * 60  -- Sarah's time converted to minutes
    difference = sarah_time - samuel_time ∧ difference = 48

/-- Proof of the homework time difference theorem -/
theorem prove_homework_time_difference : ∃ (difference : ℝ), homework_time_difference difference := by
  sorry

end homework_time_difference_prove_homework_time_difference_l3542_354242


namespace circle_containing_three_points_l3542_354279

theorem circle_containing_three_points 
  (points : Finset (ℝ × ℝ)) 
  (h1 : points.card = 51) 
  (h2 : ∀ p ∈ points, 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) :
  ∃ (center : ℝ × ℝ), ∃ (contained_points : Finset (ℝ × ℝ)),
    contained_points ⊆ points ∧
    contained_points.card ≥ 3 ∧
    ∀ p ∈ contained_points, Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ≤ 1/7 :=
by
  sorry

end circle_containing_three_points_l3542_354279


namespace binomial_12_6_l3542_354296

theorem binomial_12_6 : Nat.choose 12 6 = 1848 := by sorry

end binomial_12_6_l3542_354296


namespace line_vector_at_t_one_l3542_354207

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  vector : ℝ → Fin 2 → ℝ

/-- The theorem stating the properties of the given line and the vector to be proved -/
theorem line_vector_at_t_one
  (L : ParameterizedLine)
  (h1 : L.vector 4 = ![2, 5])
  (h2 : L.vector 5 = ![4, -3]) :
  L.vector 1 = ![8, -19] := by
  sorry

end line_vector_at_t_one_l3542_354207


namespace tree_planting_participants_l3542_354271

theorem tree_planting_participants : ∃ (x y : ℕ), 
  x * y = 2013 ∧ 
  (x - 5) * (y + 2) < 2013 ∧ 
  (x - 5) * (y + 3) > 2013 ∧ 
  x = 61 := by
  sorry

end tree_planting_participants_l3542_354271


namespace integral_even_odd_functions_l3542_354210

open Set
open Interval
open MeasureTheory
open Measure

/-- A function f is even on [-a,a] -/
def IsEven (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x ∈ Icc (-a) a, f (-x) = f x

/-- A function f is odd on [-a,a] -/
def IsOdd (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x ∈ Icc (-a) a, f (-x) = -f x

theorem integral_even_odd_functions (f : ℝ → ℝ) (a : ℝ) :
  (IsEven f a → ∫ x in Icc (-a) a, f x = 2 * ∫ x in Icc 0 a, f x) ∧
  (IsOdd f a → ∫ x in Icc (-a) a, f x = 0) := by
  sorry

end integral_even_odd_functions_l3542_354210


namespace vector_equation_solution_l3542_354246

theorem vector_equation_solution :
  let c₁ : ℚ := 5/6
  let c₂ : ℚ := -7/18
  let v₁ : Fin 2 → ℚ := ![1, 4]
  let v₂ : Fin 2 → ℚ := ![-3, 6]
  let result : Fin 2 → ℚ := ![2, 1]
  c₁ • v₁ + c₂ • v₂ = result :=
by
  sorry

#check vector_equation_solution

end vector_equation_solution_l3542_354246


namespace tree_planting_problem_l3542_354248

theorem tree_planting_problem (n t : ℕ) 
  (h1 : 4 * n = t + 11) 
  (h2 : 2 * n = t - 13) : 
  n = 12 ∧ t = 37 := by
  sorry

end tree_planting_problem_l3542_354248


namespace last_digit_of_expression_l3542_354212

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_expression : last_digit (287 * 287 + 269 * 269 - 2 * 287 * 269) = 4 := by
  sorry

end last_digit_of_expression_l3542_354212


namespace horner_method_v3_l3542_354229

def horner_polynomial (x : ℝ) : ℝ := 1 + 5*x + 10*x^2 + 10*x^3 + 5*x^4 + x^5

def horner_v1 (x : ℝ) : ℝ := x + 5

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 10

def horner_v3 (x : ℝ) : ℝ := horner_v2 x * x + 10

theorem horner_method_v3 :
  horner_v3 (-2) = 2 :=
sorry

end horner_method_v3_l3542_354229


namespace factorial_fraction_simplification_l3542_354240

theorem factorial_fraction_simplification (N : ℕ) (h : N > 2) :
  (Nat.factorial (N - 2) * (N - 1)^2) / Nat.factorial (N + 2) =
  (N - 1) / (N * (N + 1) * (N + 2)) :=
by sorry

end factorial_fraction_simplification_l3542_354240


namespace min_value_of_fraction_l3542_354214

theorem min_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x + 9 / y) ≥ 25 := by
  sorry

end min_value_of_fraction_l3542_354214


namespace tax_calculation_l3542_354290

/-- Calculate the tax amount given gross pay and net pay -/
def calculate_tax (gross_pay : ℝ) (net_pay : ℝ) : ℝ :=
  gross_pay - net_pay

theorem tax_calculation :
  let gross_pay : ℝ := 450
  let net_pay : ℝ := 315
  calculate_tax gross_pay net_pay = 135 := by
sorry

end tax_calculation_l3542_354290


namespace complex_equation_solution_l3542_354298

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l3542_354298


namespace f_is_2x_plus_7_range_f_is_5_to_13_l3542_354263

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being a first-degree function
def is_first_degree (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

-- Define the given condition for f
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17

-- Theorem for the first part
theorem f_is_2x_plus_7 (h1 : is_first_degree f) (h2 : satisfies_condition f) :
  ∀ x, f x = 2 * x + 7 :=
sorry

-- Define the range of f for x ∈ (-1, 3]
def range_f : Set ℝ := { y | ∃ x ∈ Set.Ioc (-1) 3, f x = y }

-- Theorem for the second part
theorem range_f_is_5_to_13 (h1 : is_first_degree f) (h2 : satisfies_condition f) :
  range_f = Set.Ioc 5 13 :=
sorry

end f_is_2x_plus_7_range_f_is_5_to_13_l3542_354263


namespace find_x_l3542_354268

theorem find_x (x y z : ℝ) 
  (h1 : x - y = 10) 
  (h2 : x * z = 2 * y) 
  (h3 : x + y = 14) : 
  x = 12 := by sorry

end find_x_l3542_354268


namespace same_terminal_side_angles_l3542_354226

/-- 
Theorem: The angles -675° and -315° are the only angles in the range [-720°, 0°) 
that have the same terminal side as 45°.
-/
theorem same_terminal_side_angles : 
  ∀ θ : ℝ, -720 ≤ θ ∧ θ < 0 → 
  (∃ k : ℤ, θ = 45 + 360 * k) ↔ (θ = -675 ∨ θ = -315) := by
  sorry


end same_terminal_side_angles_l3542_354226


namespace bathroom_tile_side_length_l3542_354238

-- Define the dimensions of the bathroom
def bathroom_length : ℝ := 6
def bathroom_width : ℝ := 10

-- Define the number of tiles
def number_of_tiles : ℕ := 240

-- Define the side length of a tile
def tile_side_length : ℝ := 0.5

-- Theorem statement
theorem bathroom_tile_side_length :
  bathroom_length * bathroom_width = (number_of_tiles : ℝ) * tile_side_length^2 :=
by sorry

end bathroom_tile_side_length_l3542_354238


namespace fractional_equation_integer_solution_l3542_354282

theorem fractional_equation_integer_solution (m : ℤ) : 
  (∃ x : ℤ, (m * x - 1) / (x - 2) + 1 / (2 - x) = 2 ∧ x ≠ 2) ↔ 
  (m = 4 ∨ m = 3 ∨ m = 0) :=
sorry

end fractional_equation_integer_solution_l3542_354282


namespace circle_k_range_l3542_354272

/-- The equation of a circle in general form --/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

/-- The condition for the equation to represent a circle --/
def is_circle (k : ℝ) : Prop :=
  ∃ (h c r : ℝ), ∀ (x y : ℝ), circle_equation x y k ↔ (x - h)^2 + (y - c)^2 = r^2 ∧ r > 0

/-- The theorem stating the range of k for which the equation represents a circle --/
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k > 4 ∨ k < -1 :=
sorry

end circle_k_range_l3542_354272


namespace negation_of_universal_statement_l3542_354244

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) ↔ ∃ x : ℝ, x^2 + 2*x + 5 = 0 :=
by sorry

end negation_of_universal_statement_l3542_354244


namespace fraction_evaluation_l3542_354250

theorem fraction_evaluation : (5 : ℝ) / (1 - 1/2) = 10 := by
  sorry

end fraction_evaluation_l3542_354250


namespace value_of_a_l3542_354291

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (2 * x + 2) - 5

-- State the theorem
theorem value_of_a : ∃ a : ℝ, f (1/2 * a - 1) = 2 * a - 5 ∧ f a = 6 → a = 7/4 := by
  sorry

end value_of_a_l3542_354291


namespace tabitha_honey_nights_l3542_354209

/-- The number of nights Tabitha can enjoy honey in her tea before bed -/
def honey_nights (servings_per_cup : ℕ) (cups_per_night : ℕ) (container_size : ℕ) (servings_per_ounce : ℕ) : ℕ :=
  (container_size * servings_per_ounce) / (servings_per_cup * cups_per_night)

/-- Theorem stating that Tabitha can enjoy honey in her tea for 48 nights before bed -/
theorem tabitha_honey_nights :
  honey_nights 1 2 16 6 = 48 := by sorry

end tabitha_honey_nights_l3542_354209


namespace problem_solution_l3542_354251

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}

theorem problem_solution :
  (∀ a : ℝ, a = 3 → M ∪ (Nᶜ a) = Set.univ) ∧
  (∀ a : ℝ, N a ⊆ M ↔ a ≤ 3) := by
  sorry

end problem_solution_l3542_354251


namespace function_passes_through_point_l3542_354249

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 2
  f 1 = 3 := by sorry

end function_passes_through_point_l3542_354249


namespace freshman_psychology_liberal_arts_percentage_l3542_354287

/-- Represents the student categories -/
inductive StudentCategory
| Freshman
| Sophomore
| Junior
| Senior

/-- Represents the schools -/
inductive School
| LiberalArts
| Science
| Business

/-- Represents the distribution of students across categories and schools -/
structure StudentDistribution where
  totalStudents : ℕ
  categoryPercentage : StudentCategory → ℚ
  schoolPercentage : StudentCategory → School → ℚ
  psychologyMajorPercentage : ℚ

/-- The given student distribution -/
def givenDistribution : StudentDistribution :=
  { totalStudents := 1000,  -- Arbitrary total, doesn't affect the percentage
    categoryPercentage := fun c => match c with
      | StudentCategory.Freshman => 2/5
      | StudentCategory.Sophomore => 3/10
      | StudentCategory.Junior => 1/5
      | StudentCategory.Senior => 1/10,
    schoolPercentage := fun c s => match c, s with
      | StudentCategory.Freshman, School.LiberalArts => 3/5
      | StudentCategory.Freshman, School.Science => 3/10
      | StudentCategory.Freshman, School.Business => 1/10
      | _, _ => 0,  -- Other percentages are not needed for this problem
    psychologyMajorPercentage := 1/2 }

theorem freshman_psychology_liberal_arts_percentage 
  (d : StudentDistribution) 
  (h1 : d.categoryPercentage StudentCategory.Freshman = 2/5)
  (h2 : d.schoolPercentage StudentCategory.Freshman School.LiberalArts = 3/5)
  (h3 : d.psychologyMajorPercentage = 1/2) :
  d.categoryPercentage StudentCategory.Freshman * 
  d.schoolPercentage StudentCategory.Freshman School.LiberalArts * 
  d.psychologyMajorPercentage = 12/100 := by
  sorry

end freshman_psychology_liberal_arts_percentage_l3542_354287


namespace sum_representation_exists_l3542_354299

/-- Regular 15-gon inscribed in a circle -/
structure RegularPolygon :=
  (n : ℕ)
  (radius : ℝ)
  (is_regular : n = 15)
  (is_inscribed : radius = 15)

/-- Sum of lengths of all sides and diagonals -/
def sum_lengths (p : RegularPolygon) : ℝ := sorry

/-- Representation of the sum in the required form -/
structure SumRepresentation :=
  (a b c d : ℕ)
  (sum : ℝ)
  (eq : sum = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5)

/-- Theorem stating the existence of the representation -/
theorem sum_representation_exists (p : RegularPolygon) :
  ∃ (rep : SumRepresentation), sum_lengths p = rep.sum :=
sorry

end sum_representation_exists_l3542_354299


namespace mac_preference_l3542_354283

theorem mac_preference (total : ℕ) (no_pref : ℕ) (windows_pref : ℕ) 
  (h_total : total = 210)
  (h_no_pref : no_pref = 90)
  (h_windows_pref : windows_pref = 40)
  : ∃ (mac_pref : ℕ), 
    mac_pref = 60 ∧ 
    (total - no_pref = mac_pref + windows_pref + mac_pref / 3) :=
by sorry

end mac_preference_l3542_354283


namespace teacher_zhao_masks_l3542_354230

theorem teacher_zhao_masks (n : ℕ) : 
  (n / 2 * 5 + n / 2 * 7 + 25 = n / 3 * 10 + 2 * n / 3 * 7 - 35) →
  (n / 2 * 5 + n / 2 * 7 + 25 = 205) := by
  sorry

end teacher_zhao_masks_l3542_354230


namespace sin_sum_to_product_l3542_354228

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end sin_sum_to_product_l3542_354228


namespace greatest_divisor_with_remainders_l3542_354213

theorem greatest_divisor_with_remainders : 
  Nat.gcd (976543 - 7) (897623 - 11) = 4 := by sorry

end greatest_divisor_with_remainders_l3542_354213


namespace solution_set_implies_m_range_l3542_354221

theorem solution_set_implies_m_range :
  (∀ x : ℝ, x^2 + m*x + 1 > 0) → m ∈ Set.Ioo (-2 : ℝ) 2 :=
by
  sorry

end solution_set_implies_m_range_l3542_354221


namespace max_fraction_sum_l3542_354237

theorem max_fraction_sum (n : ℕ) (hn : n ≥ 2) :
  ∃ (a b c d : ℕ),
    a / b + c / d < 1 ∧
    a + c ≤ n ∧
    ∀ (a' b' c' d' : ℕ),
      a' / b' + c' / d' < 1 →
      a' + c' ≤ n →
      a' / b' + c' / d' ≤ a / (a + (a * c + 1)) + c / (c + 1) :=
by sorry

end max_fraction_sum_l3542_354237


namespace ellipse_and_point_G_l3542_354253

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given an ellipse C and points on it, prove the equation and find point G -/
theorem ellipse_and_point_G (C : Ellipse) 
  (h_triangle_area : (1/2) * C.a * (C.a^2 - C.b^2).sqrt * C.a = 4 * Real.sqrt 3)
  (B : Point) (h_B_on_C : B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1)
  (h_B_nonzero : B.x * B.y ≠ 0)
  (A : Point) (h_A : A = ⟨0, 2 * Real.sqrt 3⟩)
  (D E : Point) (h_D : D.y = 0) (h_E : E.y = 0)
  (h_collinear_ABD : (A.y - D.y) / (A.x - D.x) = (B.y - D.y) / (B.x - D.x))
  (h_collinear_ABE : (A.y - E.y) / (A.x - E.x) = (B.y + E.y) / (B.x - E.x))
  (G : Point) (h_G : G.x = 0)
  (h_angle_equal : (G.y / D.x)^2 = (G.y / E.x)^2) :
  (C.a = 4 ∧ C.b = 2 * Real.sqrt 3) ∧ 
  (G.y = 4 ∨ G.y = -4) := by sorry

end ellipse_and_point_G_l3542_354253


namespace wage_payment_days_l3542_354280

theorem wage_payment_days (S : ℝ) (hX : S > 0) (hY : S > 0) : 
  (∃ (wX wY : ℝ), wX > 0 ∧ wY > 0 ∧ S = 36 * wX ∧ S = 45 * wY) →
  ∃ (d : ℝ), d = 20 ∧ S = d * (S / 36 + S / 45) :=
by sorry

end wage_payment_days_l3542_354280


namespace remaining_pages_l3542_354288

-- Define the total number of pages
def total_pages : ℕ := 120

-- Define the percentage used for the science project
def science_project_percentage : ℚ := 25 / 100

-- Define the number of pages used for math homework
def math_homework_pages : ℕ := 10

-- Theorem statement
theorem remaining_pages :
  total_pages - (total_pages * science_project_percentage).floor - math_homework_pages = 80 := by
  sorry

end remaining_pages_l3542_354288


namespace store_socks_problem_l3542_354284

theorem store_socks_problem (x y w z : ℕ) : 
  x + y + w + z = 15 →
  x + 2*y + 3*w + 4*z = 36 →
  x ≥ 1 →
  y ≥ 1 →
  w ≥ 1 →
  z ≥ 1 →
  x = 5 :=
by sorry

end store_socks_problem_l3542_354284


namespace aquarium_cost_is_63_l3542_354236

/-- The total cost of an aquarium after markdown and sales tax --/
def aquarium_total_cost (original_price : ℝ) (markdown_percent : ℝ) (tax_percent : ℝ) : ℝ :=
  let reduced_price := original_price * (1 - markdown_percent)
  let sales_tax := reduced_price * tax_percent
  reduced_price + sales_tax

/-- Theorem stating that the total cost of the aquarium is $63 --/
theorem aquarium_cost_is_63 :
  aquarium_total_cost 120 0.5 0.05 = 63 := by
  sorry

end aquarium_cost_is_63_l3542_354236


namespace katie_earnings_l3542_354258

/-- Calculates the total money earned from selling necklaces -/
def total_money_earned (bead_necklaces gem_necklaces price_per_necklace : ℕ) : ℕ :=
  (bead_necklaces + gem_necklaces) * price_per_necklace

/-- Proves that Katie earned 21 dollars from selling her necklaces -/
theorem katie_earnings : 
  let bead_necklaces : ℕ := 4
  let gem_necklaces : ℕ := 3
  let price_per_necklace : ℕ := 3
  total_money_earned bead_necklaces gem_necklaces price_per_necklace = 21 := by
sorry

end katie_earnings_l3542_354258


namespace typing_competition_equation_l3542_354223

/-- Prove that in a typing competition where A types x characters per minute and 
    B types (x-10) characters per minute, if A types 900 characters and B types 840 characters 
    in the same amount of time, then the equation 900/x = 840/(x-10) holds. -/
theorem typing_competition_equation (x : ℝ) 
    (hx : x > 10) -- Ensure x - 10 is positive
    (hA : 900 / x = 840 / (x - 10)) : -- Time taken by A equals time taken by B
  900 / x = 840 / (x - 10) := by
  sorry

end typing_competition_equation_l3542_354223


namespace sum_of_squares_of_roots_l3542_354252

-- Define the polynomial
def P (k X : ℝ) : ℝ := X^4 + 2*X^3 + (2 + 2*k)*X^2 + (1 + 2*k)*X + 2*k

-- Define the theorem
theorem sum_of_squares_of_roots (k : ℝ) :
  (∃ r₁ r₂ : ℝ, P k r₁ = 0 ∧ P k r₂ = 0 ∧ r₁ * r₂ = -2013) →
  (∃ r₁ r₂ : ℝ, P k r₁ = 0 ∧ P k r₂ = 0 ∧ r₁^2 + r₂^2 = 4027) :=
by sorry

end sum_of_squares_of_roots_l3542_354252


namespace tenth_term_of_sequence_l3542_354234

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 10th term of the specific arithmetic sequence -/
theorem tenth_term_of_sequence : 
  arithmeticSequenceTerm 3 6 10 = 57 := by
sorry

end tenth_term_of_sequence_l3542_354234


namespace rabbit_prob_reach_target_l3542_354247

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Defines the grid -/
def Grid := {p : Point | p.x ≤ 6 ∧ p.y ≤ 6}

/-- Defines the vertices of the grid -/
def Vertices : Set Point := {⟨0, 0⟩, ⟨0, 6⟩, ⟨6, 6⟩, ⟨6, 0⟩}

/-- Defines a valid jump on the grid -/
def ValidJump (p q : Point) : Prop :=
  (p.x = q.x ∧ (p.y + 1 = q.y ∨ q.y + 1 = p.y)) ∨
  (p.y = q.y ∧ (p.x + 1 = q.x ∨ q.x + 1 = p.x))

/-- Defines the probability of reaching (0,6) from a given point -/
noncomputable def ProbReachTarget (p : Point) : ℝ := sorry

/-- The main theorem to prove -/
theorem rabbit_prob_reach_target :
  ProbReachTarget ⟨1, 3⟩ = 1/4 := by sorry

end rabbit_prob_reach_target_l3542_354247
