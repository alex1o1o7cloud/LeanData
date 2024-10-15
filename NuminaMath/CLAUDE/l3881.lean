import Mathlib

namespace NUMINAMATH_CALUDE_farmer_tomato_rows_l3881_388108

/-- The number of tomato plants in each row -/
def plants_per_row : ℕ := 10

/-- The number of tomatoes yielded by each plant -/
def tomatoes_per_plant : ℕ := 20

/-- The total number of tomatoes harvested by the farmer -/
def total_tomatoes : ℕ := 6000

/-- The number of rows of tomatoes planted by the farmer -/
def rows_of_tomatoes : ℕ := total_tomatoes / (plants_per_row * tomatoes_per_plant)

theorem farmer_tomato_rows : rows_of_tomatoes = 30 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomato_rows_l3881_388108


namespace NUMINAMATH_CALUDE_power_seven_mod_nineteen_l3881_388157

theorem power_seven_mod_nineteen : 7^2023 ≡ 4 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_nineteen_l3881_388157


namespace NUMINAMATH_CALUDE_binomial_1300_2_l3881_388113

theorem binomial_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1300_2_l3881_388113


namespace NUMINAMATH_CALUDE_max_value_base_conversion_l3881_388119

theorem max_value_base_conversion (n A B C : ℕ) : 
  n > 0 →
  n = 64 * A + 8 * B + C →
  n = 81 * C + 9 * B + A →
  C % 2 = 0 →
  A ≤ 7 →
  B ≤ 7 →
  C ≤ 7 →
  n ≤ 64 :=
by sorry

end NUMINAMATH_CALUDE_max_value_base_conversion_l3881_388119


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3881_388136

theorem unique_solution_equation :
  ∃! x : ℝ, x ≠ 2 ∧ x - 6 / (x - 2) = 4 - 6 / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3881_388136


namespace NUMINAMATH_CALUDE_apples_per_box_l3881_388148

/-- Proves that the number of apples per box is 50 given the total number of apples,
    the desired amount to take home, and the price per box. -/
theorem apples_per_box
  (total_apples : ℕ)
  (take_home_amount : ℕ)
  (price_per_box : ℕ)
  (h1 : total_apples = 10000)
  (h2 : take_home_amount = 7000)
  (h3 : price_per_box = 35) :
  total_apples / (take_home_amount / price_per_box) = 50 := by
  sorry

#check apples_per_box

end NUMINAMATH_CALUDE_apples_per_box_l3881_388148


namespace NUMINAMATH_CALUDE_fibonacci_product_theorem_l3881_388184

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The property we want to prove -/
def satisfies_property (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ fib m * fib n = m * n

/-- The theorem statement -/
theorem fibonacci_product_theorem :
  ∀ m n : ℕ, satisfies_property m n ↔ (m = 1 ∧ n = 1) ∨ (m = 5 ∧ n = 5) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_product_theorem_l3881_388184


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3881_388181

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.sin (10 * π / 180) - 
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3881_388181


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3881_388192

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

-- State the theorem
theorem arithmetic_sequence_formula 
  (a : ℕ → ℚ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h1 : a 3 + a 4 = 4)
  (h2 : a 5 + a 7 = 6) :
  ∃ C : ℚ, ∀ n : ℕ, a n = (2 * n + C) / 5 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3881_388192


namespace NUMINAMATH_CALUDE_closed_path_count_l3881_388182

/-- The number of distinct closed paths on a grid with total length 2n -/
def num_closed_paths (n : ℕ) : ℕ := (Nat.choose (2 * n) n) ^ 2

/-- Theorem stating that the number of distinct closed paths on a grid
    with total length 2n is equal to (C_{2n}^n)^2 -/
theorem closed_path_count (n : ℕ) : 
  num_closed_paths n = (Nat.choose (2 * n) n) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_closed_path_count_l3881_388182


namespace NUMINAMATH_CALUDE_min_value_x_plus_sqrt_x2_y2_l3881_388158

theorem min_value_x_plus_sqrt_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  ∃ (min : ℝ), min = 8/5 ∧ ∀ (z : ℝ), z > 0 → 2 * z + (2 - 2 * z) = 2 →
    x + Real.sqrt (x^2 + y^2) ≥ min ∧ z + Real.sqrt (z^2 + (2 - 2 * z)^2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_sqrt_x2_y2_l3881_388158


namespace NUMINAMATH_CALUDE_solve_aunt_gift_problem_l3881_388115

def aunt_gift_problem (jade_initial : ℕ) (julia_initial : ℕ) (total_final : ℕ) : Prop :=
  let total_initial := jade_initial + julia_initial
  let total_gift := total_final - total_initial
  let gift_per_person := total_gift / 2
  (jade_initial = 38) ∧
  (julia_initial = jade_initial / 2) ∧
  (total_final = 97) ∧
  (gift_per_person = 20)

theorem solve_aunt_gift_problem :
  ∃ (jade_initial julia_initial total_final : ℕ),
    aunt_gift_problem jade_initial julia_initial total_final :=
by
  sorry

end NUMINAMATH_CALUDE_solve_aunt_gift_problem_l3881_388115


namespace NUMINAMATH_CALUDE_dollar_function_iteration_l3881_388100

-- Define the dollar function
def dollar (N : ℝ) : ℝ := 0.3 * N + 2

-- State the theorem
theorem dollar_function_iteration : dollar (dollar (dollar 60)) = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_dollar_function_iteration_l3881_388100


namespace NUMINAMATH_CALUDE_bank_deposit_time_calculation_l3881_388129

/-- Proves that given two equal deposits at the same interest rate, 
    if the difference in interest is known, we can determine the time for the first deposit. -/
theorem bank_deposit_time_calculation 
  (deposit : ℝ) 
  (rate : ℝ) 
  (time_second : ℝ) 
  (interest_diff : ℝ) 
  (h1 : deposit = 640)
  (h2 : rate = 0.15)
  (h3 : time_second = 5)
  (h4 : interest_diff = 144) :
  ∃ (time_first : ℝ), 
    deposit * rate * time_second - deposit * rate * time_first = interest_diff ∧ 
    time_first = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_bank_deposit_time_calculation_l3881_388129


namespace NUMINAMATH_CALUDE_ratio_from_S1_ratio_from_S1_S2_ratio_from_S2_l3881_388187

/-- Represents a trapezoid with diagonals intersecting at a point -/
structure Trapezoid where
  S : ℝ  -- Area of the trapezoid
  S1 : ℝ  -- Area of triangle OBC
  S2 : ℝ  -- Area of triangle OCD
  S3 : ℝ  -- Area of triangle ODA
  S4 : ℝ  -- Area of triangle AOB
  AD : ℝ  -- Length of side AD
  BC : ℝ  -- Length of side BC

/-- There exists a function that determines AD/BC given S1/S -/
theorem ratio_from_S1 (t : Trapezoid) : 
  ∃ f : ℝ → ℝ, t.AD / t.BC = f (t.S1 / t.S) :=
sorry

/-- There exists a function that determines AD/BC given (S1+S2)/S -/
theorem ratio_from_S1_S2 (t : Trapezoid) : 
  ∃ f : ℝ → ℝ, t.AD / t.BC = f ((t.S1 + t.S2) / t.S) :=
sorry

/-- There exists a function that determines AD/BC given S2/S -/
theorem ratio_from_S2 (t : Trapezoid) : 
  ∃ f : ℝ → ℝ, t.AD / t.BC = f (t.S2 / t.S) :=
sorry

end NUMINAMATH_CALUDE_ratio_from_S1_ratio_from_S1_S2_ratio_from_S2_l3881_388187


namespace NUMINAMATH_CALUDE_coefficient_x_term_expansion_l3881_388188

theorem coefficient_x_term_expansion (x : ℝ) : 
  (∃ a b c d e : ℝ, (1 + x) * (2 - x)^4 = a*x^4 + b*x^3 + c*x^2 + d*x + e) → 
  (∃ a b c d e : ℝ, (1 + x) * (2 - x)^4 = a*x^4 + b*x^3 + c*x^2 + (-16)*x + e) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_term_expansion_l3881_388188


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3881_388191

theorem polynomial_factorization (a b c : ℝ) : 
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) = 
  (a - b) * (b - c) * (c - a) * ((a + b) * a^2 * b^2 + (b + c) * b^2 * c^2 + (a + c) * c^2 * a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3881_388191


namespace NUMINAMATH_CALUDE_gcf_98_140_245_l3881_388144

theorem gcf_98_140_245 : Nat.gcd 98 (Nat.gcd 140 245) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_98_140_245_l3881_388144


namespace NUMINAMATH_CALUDE_m_range_l3881_388150

open Set

def A : Set ℝ := {x : ℝ | |x - 1| + |x + 1| ≤ 3}

def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - (2*m + 1)*x + m^2 + m < 0}

theorem m_range (m : ℝ) : (A ∩ B m).Nonempty → m ∈ Set.Ioo (-5/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_m_range_l3881_388150


namespace NUMINAMATH_CALUDE_interest_difference_approx_l3881_388178

-- Define the initial deposit
def initial_deposit : ℝ := 12000

-- Define the interest rates
def compound_rate : ℝ := 0.06
def simple_rate : ℝ := 0.08

-- Define the time period
def years : ℕ := 20

-- Define the compound interest function
def compound_balance (p r : ℝ) (n : ℕ) : ℝ := p * (1 + r) ^ n

-- Define the simple interest function
def simple_balance (p r : ℝ) (n : ℕ) : ℝ := p * (1 + n * r)

-- State the theorem
theorem interest_difference_approx :
  ∃ (ε : ℝ), ε < 1 ∧ 
  |round (compound_balance initial_deposit compound_rate years - 
          simple_balance initial_deposit simple_rate years) - 7286| ≤ ε :=
sorry

end NUMINAMATH_CALUDE_interest_difference_approx_l3881_388178


namespace NUMINAMATH_CALUDE_siblings_selection_probability_l3881_388164

/-- The probability of three siblings being selected simultaneously -/
theorem siblings_selection_probability (px py pz : ℚ) 
  (hx : px = 1/7) (hy : py = 2/9) (hz : pz = 3/11) : 
  px * py * pz = 1/115.5 := by
  sorry

end NUMINAMATH_CALUDE_siblings_selection_probability_l3881_388164


namespace NUMINAMATH_CALUDE_work_completion_time_l3881_388163

/-- The time it takes to complete a work given two workers with different rates and a specific work schedule. -/
theorem work_completion_time 
  (total_work : ℝ) 
  (p_rate : ℝ) 
  (q_rate : ℝ) 
  (p_solo_days : ℝ) 
  (hp : p_rate = total_work / 10) 
  (hq : q_rate = total_work / 6) 
  (hp_solo : p_solo_days = 2) : 
  p_solo_days + (total_work - p_solo_days * p_rate) / (p_rate + q_rate) = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3881_388163


namespace NUMINAMATH_CALUDE_solve_problem_l3881_388120

def problem (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧
  (x + y + 9 + 10 + 11) / 5 = 10 ∧
  ((x - 10)^2 + (y - 10)^2 + (9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2) / 5 = 2

theorem solve_problem (x y : ℝ) (h : problem x y) : |x - y| = 4 :=
by sorry

end NUMINAMATH_CALUDE_solve_problem_l3881_388120


namespace NUMINAMATH_CALUDE_tangent_sqrt_two_implications_l3881_388107

theorem tangent_sqrt_two_implications (θ : Real) (h : Real.tan θ = Real.sqrt 2) :
  ((Real.cos θ + Real.sin θ) / (Real.cos θ - Real.sin θ) = -3 - 2 * Real.sqrt 2) ∧
  (Real.sin θ ^ 2 - Real.sin θ * Real.cos θ + 2 * Real.cos θ ^ 2 = (4 - Real.sqrt 2) / 3) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sqrt_two_implications_l3881_388107


namespace NUMINAMATH_CALUDE_equal_savings_l3881_388176

/-- Represents the financial situation of Uma and Bala -/
structure FinancialSituation where
  uma_income : ℝ
  bala_income : ℝ
  uma_expenditure : ℝ
  bala_expenditure : ℝ

/-- The conditions given in the problem -/
def problem_conditions (fs : FinancialSituation) : Prop :=
  fs.uma_income / fs.bala_income = 8 / 7 ∧
  fs.uma_expenditure / fs.bala_expenditure = 7 / 6 ∧
  fs.uma_income = 16000

/-- The savings of Uma and Bala -/
def savings (fs : FinancialSituation) : ℝ × ℝ :=
  (fs.uma_income - fs.uma_expenditure, fs.bala_income - fs.bala_expenditure)

/-- The theorem to be proved -/
theorem equal_savings (fs : FinancialSituation) :
  problem_conditions fs → savings fs = (2000, 2000) := by
  sorry


end NUMINAMATH_CALUDE_equal_savings_l3881_388176


namespace NUMINAMATH_CALUDE_equation_roots_l3881_388133

def equation (x : ℝ) : ℝ := x * (x + 2)^2 * (3 - x) * (5 + x)

theorem equation_roots : 
  {x : ℝ | equation x = 0} = {0, -2, 3, -5} := by sorry

end NUMINAMATH_CALUDE_equation_roots_l3881_388133


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3881_388171

theorem linear_equation_solution (a b : ℝ) : 
  (a * (-2) - 3 * b * 3 = 5) → 
  (a * 4 - 3 * b * 1 = 5) → 
  a + b = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3881_388171


namespace NUMINAMATH_CALUDE_f_min_and_inequality_l3881_388130

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

noncomputable def g (x : ℝ) : ℝ := x / exp x - 2 / exp 1

theorem f_min_and_inequality :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x) ∧
  (∀ (x : ℝ), x > 0 → f x ≥ -1 / exp 1) ∧
  (∀ (m n : ℝ), m > 0 → n > 0 → f m ≥ g n) :=
sorry

end NUMINAMATH_CALUDE_f_min_and_inequality_l3881_388130


namespace NUMINAMATH_CALUDE_second_coin_value_l3881_388142

/-- Proves that the value of the second type of coin is 0.5 rupees -/
theorem second_coin_value (total_value : ℝ) (num_coins : ℕ) (coin1_value : ℝ) (coin3_value : ℝ) :
  total_value = 35 →
  num_coins = 20 →
  coin1_value = 1 →
  coin3_value = 0.25 →
  ∃ (coin2_value : ℝ), 
    coin2_value = 0.5 ∧
    num_coins * (coin1_value + coin2_value + coin3_value) = total_value :=
by sorry

end NUMINAMATH_CALUDE_second_coin_value_l3881_388142


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l3881_388151

theorem binomial_expansion_property (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_property_l3881_388151


namespace NUMINAMATH_CALUDE_second_order_arithmetic_sequence_property_l3881_388174

/-- Second-order arithmetic sequence -/
def SecondOrderArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ x y z : ℝ, ∀ n : ℕ, a n = x * n^2 + y * n + z

/-- First-order difference sequence -/
def FirstOrderDifference (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a (n + 1) - a n

/-- Second-order difference sequence -/
def SecondOrderDifference (b : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, c n = b (n + 1) - b n

theorem second_order_arithmetic_sequence_property
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (c : ℕ → ℝ)
  (h1 : SecondOrderArithmeticSequence a)
  (h2 : FirstOrderDifference a b)
  (h3 : SecondOrderDifference b c)
  (h4 : ∀ n : ℕ, c n = 20)
  (h5 : a 10 = 23)
  (h6 : a 20 = 23) :
  a 30 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_second_order_arithmetic_sequence_property_l3881_388174


namespace NUMINAMATH_CALUDE_production_rate_equation_l3881_388166

theorem production_rate_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_diff : x = y + 4) :
  (100 / x = 80 / y) ↔ 
  (∃ (rate_A rate_B : ℝ), 
    rate_A = x ∧ 
    rate_B = y ∧ 
    rate_A > rate_B ∧ 
    rate_A - rate_B = 4 ∧
    (100 / rate_A) = (80 / rate_B)) :=
by sorry

end NUMINAMATH_CALUDE_production_rate_equation_l3881_388166


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3881_388128

theorem arithmetic_evaluation : (7 + 5 + 3) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3881_388128


namespace NUMINAMATH_CALUDE_travis_annual_cereal_cost_l3881_388198

/-- Calculates the annual cereal cost for Travis --/
theorem travis_annual_cereal_cost :
  let box_a_cost : ℝ := 3.50
  let box_b_cost : ℝ := 4.00
  let box_c_cost : ℝ := 5.25
  let box_a_consumption : ℝ := 1
  let box_b_consumption : ℝ := 0.5
  let box_c_consumption : ℝ := 1/3
  let discount_rate : ℝ := 0.1
  let weeks_per_year : ℕ := 52

  let weekly_cost : ℝ := 
    box_a_cost * box_a_consumption + 
    box_b_cost * box_b_consumption + 
    box_c_cost * box_c_consumption

  let discounted_weekly_cost : ℝ := weekly_cost * (1 - discount_rate)

  let annual_cost : ℝ := discounted_weekly_cost * weeks_per_year

  annual_cost = 339.30 := by sorry

end NUMINAMATH_CALUDE_travis_annual_cereal_cost_l3881_388198


namespace NUMINAMATH_CALUDE_cooper_savings_l3881_388189

theorem cooper_savings (total_savings : ℕ) (days_in_year : ℕ) (daily_savings : ℕ) :
  total_savings = 12410 →
  days_in_year = 365 →
  daily_savings * days_in_year = total_savings →
  daily_savings = 34 := by
  sorry

end NUMINAMATH_CALUDE_cooper_savings_l3881_388189


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3881_388194

theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3881_388194


namespace NUMINAMATH_CALUDE_paise_to_rupees_l3881_388154

/-- 
If 0.5% of a quantity is equal to 65 paise, then the quantity is equal to 130 rupees.
-/
theorem paise_to_rupees (a : ℝ) : (0.005 * a = 65) → (a = 130 * 100) := by
  sorry

end NUMINAMATH_CALUDE_paise_to_rupees_l3881_388154


namespace NUMINAMATH_CALUDE_systematic_sampling_l3881_388167

theorem systematic_sampling (total : Nat) (sample_size : Nat) (drawn : Nat) : 
  total = 800 → 
  sample_size = 50 → 
  drawn = 7 → 
  ∃ (selected : Nat), 
    selected = drawn + 2 * (total / sample_size) ∧ 
    33 ≤ selected ∧ 
    selected ≤ 48 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l3881_388167


namespace NUMINAMATH_CALUDE_special_polyhedron_sum_l3881_388160

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex

/-- The theorem about the special polyhedron -/
theorem special_polyhedron_sum (p : SpecialPolyhedron) : 
  p.F = 30 ∧ 
  p.F = p.t + p.h ∧
  p.T = 3 ∧ 
  p.H = 2 ∧
  p.V - p.E + p.F = 2 ∧ 
  p.E = (3 * p.t + 6 * p.h) / 2 →
  100 * p.H + 10 * p.T + p.V = 262 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_sum_l3881_388160


namespace NUMINAMATH_CALUDE_backpacking_roles_l3881_388175

theorem backpacking_roles (n : ℕ) (h : n = 10) : 
  (n.choose 2) * ((n - 2).choose 1) = 360 := by
  sorry

end NUMINAMATH_CALUDE_backpacking_roles_l3881_388175


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3881_388179

theorem sum_interior_angles_regular_polygon (n : ℕ) (h : n > 2) :
  (360 / 45 : ℝ) = n →
  (180 * (n - 2) : ℝ) = 1080 :=
by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3881_388179


namespace NUMINAMATH_CALUDE_gcd_210_162_l3881_388109

theorem gcd_210_162 : Nat.gcd 210 162 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_210_162_l3881_388109


namespace NUMINAMATH_CALUDE_q_is_zero_l3881_388118

/-- A cubic polynomial with roots at -2, 0, and 2, passing through (1, -3) -/
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem q_is_zero (p q r s : ℝ) :
  (∀ x, x = -2 ∨ x = 0 ∨ x = 2 → g p q r s x = 0) →
  g p q r s 1 = -3 →
  q = 0 :=
sorry

end NUMINAMATH_CALUDE_q_is_zero_l3881_388118


namespace NUMINAMATH_CALUDE_unique_solution_iff_n_eleven_l3881_388173

/-- The equation x^2 - 3x + 5 = 0 has a unique solution in (ℤ_n, +, ·) if and only if n = 11 -/
theorem unique_solution_iff_n_eleven (n : ℕ) (hn : n ≥ 2) :
  (∃! x : ZMod n, x^2 - 3*x + 5 = 0) ↔ n = 11 := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_n_eleven_l3881_388173


namespace NUMINAMATH_CALUDE_sequence_general_term_l3881_388185

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = n^2 - 4n,
    prove that the general term a_n is equal to 2n - 5. -/
theorem sequence_general_term (a : ℕ → ℤ) (S : ℕ → ℤ)
    (h : ∀ n : ℕ, S n = n^2 - 4*n) :
  ∀ n : ℕ, a n = 2*n - 5 :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3881_388185


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3881_388145

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 9 = 1 ∨ y^2 / 25 - x^2 / 9 = 1

theorem hyperbola_standard_equation
  (center_origin : ℝ × ℝ)
  (real_axis_length : ℝ)
  (imaginary_axis_length : ℝ)
  (h1 : center_origin = (0, 0))
  (h2 : real_axis_length = 10)
  (h3 : imaginary_axis_length = 6) :
  ∀ x y : ℝ, hyperbola_equation x y := by
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3881_388145


namespace NUMINAMATH_CALUDE_max_xy_given_constraint_l3881_388138

theorem max_xy_given_constraint (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4 * y = 4) :
  ∀ x' y' : ℝ, 0 < x' → 0 < y' → x' + 4 * y' = 4 → x' * y' ≤ x * y ∧ x * y = 1 :=
sorry

end NUMINAMATH_CALUDE_max_xy_given_constraint_l3881_388138


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3881_388152

theorem complex_equation_solution (z : ℂ) : z / (1 - I) = 3 + 2*I → z = 5 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3881_388152


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l3881_388121

theorem product_mod_seventeen :
  (1234 * 1235 * 1236 * 1237 * 1238) % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l3881_388121


namespace NUMINAMATH_CALUDE_percent_of_a_l3881_388147

theorem percent_of_a (a b c : ℝ) (h1 : b = 0.5 * a) (h2 : c = 0.5 * b) :
  c = 0.25 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_l3881_388147


namespace NUMINAMATH_CALUDE_common_divisors_8400_7560_l3881_388127

theorem common_divisors_8400_7560 : Nat.card {d : ℕ | d ∣ 8400 ∧ d ∣ 7560} = 32 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_8400_7560_l3881_388127


namespace NUMINAMATH_CALUDE_min_transport_cost_l3881_388159

/-- Represents the transportation problem between two villages and two destinations -/
structure TransportProblem where
  villageA_supply : ℝ
  villageB_supply : ℝ
  destX_demand : ℝ
  destY_demand : ℝ
  costA_to_X : ℝ
  costA_to_Y : ℝ
  costB_to_X : ℝ
  costB_to_Y : ℝ

/-- Calculates the total transportation cost given the amount transported from A to X -/
def totalCost (p : TransportProblem) (x : ℝ) : ℝ :=
  p.costA_to_X * x + p.costA_to_Y * (p.villageA_supply - x) +
  p.costB_to_X * (p.destX_demand - x) + p.costB_to_Y * (x - (p.villageA_supply + p.villageB_supply - p.destX_demand - p.destY_demand))

/-- The specific problem instance -/
def vegetableProblem : TransportProblem :=
  { villageA_supply := 80
  , villageB_supply := 60
  , destX_demand := 65
  , destY_demand := 75
  , costA_to_X := 50
  , costA_to_Y := 30
  , costB_to_X := 60
  , costB_to_Y := 45 }

/-- Theorem stating that the minimum transportation cost for the vegetable problem is 6100 -/
theorem min_transport_cost :
  ∃ x, x ≥ 0 ∧ x ≤ vegetableProblem.villageA_supply ∧
       x ≤ vegetableProblem.destX_demand ∧
       x ≥ (vegetableProblem.villageA_supply + vegetableProblem.villageB_supply - vegetableProblem.destX_demand - vegetableProblem.destY_demand) ∧
       totalCost vegetableProblem x = 6100 ∧
       ∀ y, y ≥ 0 → y ≤ vegetableProblem.villageA_supply →
             y ≤ vegetableProblem.destX_demand →
             y ≥ (vegetableProblem.villageA_supply + vegetableProblem.villageB_supply - vegetableProblem.destX_demand - vegetableProblem.destY_demand) →
             totalCost vegetableProblem x ≤ totalCost vegetableProblem y :=
by sorry


end NUMINAMATH_CALUDE_min_transport_cost_l3881_388159


namespace NUMINAMATH_CALUDE_cube_edge_length_l3881_388116

theorem cube_edge_length (l w h : ℝ) (cube_edge : ℝ) : 
  l = 2 → w = 4 → h = 8 → l * w * h = cube_edge^3 → cube_edge = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3881_388116


namespace NUMINAMATH_CALUDE_percent_equality_l3881_388196

theorem percent_equality (x y : ℝ) (P : ℝ) (h1 : y = 0.25 * x) 
  (h2 : (P / 100) * (x - y) = 0.15 * (x + y)) : P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l3881_388196


namespace NUMINAMATH_CALUDE_supplementary_angle_theorem_l3881_388190

theorem supplementary_angle_theorem (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ x = 3 * (180 - x) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angle_theorem_l3881_388190


namespace NUMINAMATH_CALUDE_absolute_sum_a_b_l3881_388131

theorem absolute_sum_a_b : ∀ a b : ℝ, 
  (∀ x : ℝ, (7*x - a)^2 = 49*x^2 - b*x + 9) → 
  |a + b| = 45 := by
sorry

end NUMINAMATH_CALUDE_absolute_sum_a_b_l3881_388131


namespace NUMINAMATH_CALUDE_income_ratio_l3881_388103

def monthly_income_C : ℕ := 17000
def annual_income_A : ℕ := 571200

def monthly_income_B : ℕ := monthly_income_C + (12 * monthly_income_C) / 100
def monthly_income_A : ℕ := annual_income_A / 12

theorem income_ratio :
  (monthly_income_A : ℚ) / monthly_income_B = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_income_ratio_l3881_388103


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3881_388170

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (15 * Complex.I) / (3 + 4 * Complex.I)
  Complex.im z = 9/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3881_388170


namespace NUMINAMATH_CALUDE_f_monotonicity_l3881_388180

noncomputable def f (x : ℝ) : ℝ := -2 * x / (1 + x^2)

theorem f_monotonicity :
  (∀ x y, x < y ∧ y < -1 → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_l3881_388180


namespace NUMINAMATH_CALUDE_train_crossing_time_specific_train_crossing_time_l3881_388155

/-- The time (in seconds) it takes for a train to cross a man walking in the same direction --/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let train_speed_ms := train_speed * 1000 / 3600
  let man_speed_ms := man_speed * 1000 / 3600
  let relative_speed := train_speed_ms - man_speed_ms
  train_length / relative_speed

/-- The specific problem instance --/
theorem specific_train_crossing_time :
  train_crossing_time 900 63 3 = 54 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_specific_train_crossing_time_l3881_388155


namespace NUMINAMATH_CALUDE_logarithm_equality_l3881_388139

theorem logarithm_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 4*y^2 = 12*x*y) :
  Real.log (x + 2*y) / Real.log 10 - 2 * (Real.log 2 / Real.log 10) = 
  (1/2) * (Real.log x / Real.log 10 + Real.log y / Real.log 10) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_l3881_388139


namespace NUMINAMATH_CALUDE_statement_a_incorrect_statement_b_correct_statement_c_correct_statement_d_correct_l3881_388117

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary operations and relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect_line : Line → Line → Prop)
variable (intersect_plane : Plane → Plane → Line)

-- Statement A
theorem statement_a_incorrect 
  (a b : Line) (α : Plane) :
  ∃ a b α, subset b α ∧ parallel a b ∧ ¬(parallel_line_plane a α) := by sorry

-- Statement B
theorem statement_b_correct 
  (a b : Line) (α β : Plane) :
  parallel_line_plane a α → intersect_plane α β = b → subset a β → parallel a b := by sorry

-- Statement C
theorem statement_c_correct 
  (a b : Line) (α β : Plane) (p : Line) :
  subset a α → subset b α → intersect_line a b → 
  parallel_line_plane a β → parallel_line_plane b β → 
  parallel_plane α β := by sorry

-- Statement D
theorem statement_d_correct 
  (a b : Line) (α β γ : Plane) :
  parallel_plane α β → intersect_plane α γ = a → intersect_plane β γ = b → 
  parallel a b := by sorry

end NUMINAMATH_CALUDE_statement_a_incorrect_statement_b_correct_statement_c_correct_statement_d_correct_l3881_388117


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3881_388124

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : 
  distribute_balls 5 4 = 56 := by
  sorry

#eval distribute_balls 5 4

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3881_388124


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3881_388135

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 149) 
  (sum_of_products : a*b + b*c + a*c = 70) : 
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3881_388135


namespace NUMINAMATH_CALUDE_x_intercept_distance_l3881_388125

/-- Given two lines intersecting at (8, 20) with slopes 4 and -2,
    the distance between their x-intercepts is 15. -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∀ x, line1 x = 4 * x - 12) →
  (∀ x, line2 x = -2 * x + 36) →
  line1 8 = 20 →
  line2 8 = 20 →
  |line1⁻¹ 0 - line2⁻¹ 0| = 15 :=
sorry

end NUMINAMATH_CALUDE_x_intercept_distance_l3881_388125


namespace NUMINAMATH_CALUDE_total_bathing_suits_l3881_388122

theorem total_bathing_suits (men_suits : ℕ) (women_suits : ℕ)
  (h1 : men_suits = 14797)
  (h2 : women_suits = 4969) :
  men_suits + women_suits = 19766 := by
  sorry

end NUMINAMATH_CALUDE_total_bathing_suits_l3881_388122


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3881_388169

theorem trigonometric_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.cos (x + y) ^ 2 + 2 * Real.sin x * Real.sin y * Real.cos (x + y) = 1 + Real.cos y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3881_388169


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3881_388149

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3881_388149


namespace NUMINAMATH_CALUDE_min_sqrt_equality_l3881_388101

theorem min_sqrt_equality (x y z : ℝ) : 
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 →
  (min (Real.sqrt (x + x*y*z)) (min (Real.sqrt (y + x*y*z)) (Real.sqrt (z + x*y*z))) = 
   Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1)) ↔
  ∃ t : ℝ, t > 0 ∧ 
    x = 1 + (t / (t^2 + 1))^2 ∧ 
    y = 1 + 1 / t^2 ∧ 
    z = 1 + t^2 :=
by sorry

end NUMINAMATH_CALUDE_min_sqrt_equality_l3881_388101


namespace NUMINAMATH_CALUDE_cherry_pie_degree_is_48_l3881_388102

/-- Represents the pie preferences in a class --/
structure PiePreferences where
  total : ℕ
  chocolate : ℕ
  apple : ℕ
  blueberry : ℕ
  cherry_lemon_equal : Bool

/-- Calculates the degree for cherry pie in a pie chart --/
def cherry_pie_degree (prefs : PiePreferences) : ℕ :=
  let remaining := prefs.total - (prefs.chocolate + prefs.apple + prefs.blueberry)
  let cherry := (remaining + 1) / 2  -- Round up for cherry
  (cherry * 360) / prefs.total

/-- The main theorem stating the degree for cherry pie --/
theorem cherry_pie_degree_is_48 (prefs : PiePreferences) 
  (h1 : prefs.total = 45)
  (h2 : prefs.chocolate = 15)
  (h3 : prefs.apple = 10)
  (h4 : prefs.blueberry = 9)
  (h5 : prefs.cherry_lemon_equal = true) :
  cherry_pie_degree prefs = 48 := by
  sorry

#eval cherry_pie_degree ⟨45, 15, 10, 9, true⟩

end NUMINAMATH_CALUDE_cherry_pie_degree_is_48_l3881_388102


namespace NUMINAMATH_CALUDE_min_a_value_l3881_388146

/-- Set A defined by the quadratic inequality -/
def set_A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (1 - a) * p.1^2 + 2 * p.1 * p.2 - a * p.2^2 ≤ 0}

/-- Set B defined by the linear inequality and positivity conditions -/
def set_B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 5 * p.2 ≥ 0 ∧ p.1 > 0 ∧ p.2 > 0}

/-- Theorem stating the minimum value of a given the subset relationship -/
theorem min_a_value (h : set_B ⊆ set_A a) : a ≥ 55 / 34 := by
  sorry

#check min_a_value

end NUMINAMATH_CALUDE_min_a_value_l3881_388146


namespace NUMINAMATH_CALUDE_expression_equality_l3881_388168

theorem expression_equality : 
  (Real.log 5) ^ 0 + (9 / 4) ^ (1 / 2) + Real.sqrt ((1 - Real.sqrt 2) ^ 2) - 2 ^ (Real.log 2 / Real.log 4) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3881_388168


namespace NUMINAMATH_CALUDE_digit_product_theorem_l3881_388193

theorem digit_product_theorem (A M C : ℕ) : 
  A < 10 → M < 10 → C < 10 →
  (100 * A + 10 * M + C) * (A + M + C) = 2244 →
  A = 3 := by
sorry

end NUMINAMATH_CALUDE_digit_product_theorem_l3881_388193


namespace NUMINAMATH_CALUDE_delivery_problem_l3881_388186

theorem delivery_problem (total : ℕ) (cider : ℕ) (beer : ℕ) 
  (h_total : total = 180)
  (h_cider : cider = 40)
  (h_beer : beer = 80) :
  let mixture := total - (cider + beer)
  (cider / 2 + beer / 2 + mixture / 2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_delivery_problem_l3881_388186


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3881_388132

/-- The average speed of a round trip journey given outbound and inbound speeds -/
theorem round_trip_average_speed 
  (outbound_speed inbound_speed : ℝ) 
  (outbound_speed_pos : outbound_speed > 0)
  (inbound_speed_pos : inbound_speed > 0)
  (h_outbound : outbound_speed = 44)
  (h_inbound : inbound_speed = 36) :
  2 * outbound_speed * inbound_speed / (outbound_speed + inbound_speed) = 39.6 := by
  sorry

#check round_trip_average_speed

end NUMINAMATH_CALUDE_round_trip_average_speed_l3881_388132


namespace NUMINAMATH_CALUDE_train_speed_problem_l3881_388165

/-- Proves that given the conditions of the train problem, the speed of Train A is 43 miles per hour. -/
theorem train_speed_problem (speed_B : ℝ) (headstart : ℝ) (overtake_distance : ℝ) 
  (h1 : speed_B = 45)
  (h2 : headstart = 2)
  (h3 : overtake_distance = 180) :
  ∃ (speed_A : ℝ) (overtake_time : ℝ), 
    speed_A = 43 ∧ 
    speed_A * (headstart + overtake_time) = overtake_distance ∧
    speed_B * overtake_time = overtake_distance :=
by
  sorry


end NUMINAMATH_CALUDE_train_speed_problem_l3881_388165


namespace NUMINAMATH_CALUDE_exponent_identities_l3881_388123

theorem exponent_identities (x a : ℝ) (h : a ≠ 0) : 
  (3 * x^2 * x^4 - (-x^3)^2 = 2 * x^6) ∧ 
  (a^3 * a + (-a^2)^3 / a^2 = 0) := by sorry

end NUMINAMATH_CALUDE_exponent_identities_l3881_388123


namespace NUMINAMATH_CALUDE_complement_of_A_l3881_388112

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

theorem complement_of_A : (Aᶜ : Set ℝ) = {x | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3881_388112


namespace NUMINAMATH_CALUDE_work_completion_time_l3881_388137

/-- Represents the time it takes for A to complete the work alone -/
def time_A : ℝ := 15

/-- Represents the time it takes for B to complete the work alone -/
def time_B : ℝ := 27

/-- Represents the total amount of work -/
def total_work : ℝ := 1

/-- Represents the number of days A works before leaving -/
def days_A_worked : ℝ := 5

/-- Represents the number of days B works to complete the remaining work -/
def days_B_worked : ℝ := 18

theorem work_completion_time :
  (days_A_worked / time_A) + (days_B_worked / time_B) = total_work ∧
  time_A = total_work / ((total_work - (days_B_worked / time_B)) / days_A_worked) :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3881_388137


namespace NUMINAMATH_CALUDE_sector_max_area_l3881_388104

/-- Given a circular sector with perimeter 40 units, prove that the area is maximized
    when the central angle is 2 radians and the maximum area is 100 square units. -/
theorem sector_max_area (R : ℝ) (α : ℝ) (h : R * α + 2 * R = 40) :
  (R * α * R / 2 ≤ 100) ∧
  (R * α * R / 2 = 100 ↔ α = 2 ∧ R = 10) :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l3881_388104


namespace NUMINAMATH_CALUDE_propositions_correctness_l3881_388143

-- Define the property of being even
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define the property of being divisible by 2
def DivisibleBy2 (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define the inequality from proposition ③
def Inequality (a x : ℝ) : Prop := (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0

theorem propositions_correctness :
  -- Proposition ②
  (¬ ∀ n : ℤ, DivisibleBy2 n → IsEven n) ↔ (∃ n : ℤ, DivisibleBy2 n ∧ ¬IsEven n)
  ∧
  -- Proposition ③
  ∃ a : ℝ, (¬ (|a| ≤ 1)) ∧ (∀ x : ℝ, ¬Inequality a x) :=
by sorry

end NUMINAMATH_CALUDE_propositions_correctness_l3881_388143


namespace NUMINAMATH_CALUDE_diagonal_length_of_courtyard_l3881_388183

/-- Represents a rectangular courtyard with sides in ratio 4:3 -/
structure Courtyard where
  length : ℝ
  width : ℝ
  ratio_constraint : length = (4/3) * width

/-- The cost of paving in Rupees per square meter -/
def paving_cost_per_sqm : ℝ := 0.5

/-- The total cost of paving the courtyard in Rupees -/
def total_paving_cost : ℝ := 600

theorem diagonal_length_of_courtyard (c : Courtyard) : 
  c.length * c.width * paving_cost_per_sqm = total_paving_cost →
  Real.sqrt (c.length^2 + c.width^2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_of_courtyard_l3881_388183


namespace NUMINAMATH_CALUDE_daily_savings_amount_l3881_388153

/-- Represents the number of days Ian saves money -/
def savingDays : ℕ := 40

/-- Represents the total amount saved in dimes -/
def totalSavedDimes : ℕ := 4

/-- Represents the value of a dime in cents -/
def dimeValueInCents : ℕ := 10

/-- Theorem: If Ian saves for 40 days and accumulates 4 dimes, his daily savings is 1 cent -/
theorem daily_savings_amount : 
  (totalSavedDimes * dimeValueInCents) / savingDays = 1 := by
  sorry

end NUMINAMATH_CALUDE_daily_savings_amount_l3881_388153


namespace NUMINAMATH_CALUDE_prime_square_minus_one_div_24_l3881_388140

theorem prime_square_minus_one_div_24 (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) : 
  24 ∣ (p^2 - 1) := by
sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_div_24_l3881_388140


namespace NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_given_line_l3881_388199

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y = 0

/-- The equation of the given line -/
def given_line (x y : ℝ) : Prop :=
  2*x - y = 0

/-- The equation of the line we need to prove -/
def target_line (x y : ℝ) : Prop :=
  2*x - y - 3 = 0

/-- Theorem stating that the line passing through the center of the circle
    and parallel to the given line has the equation 2x - y - 3 = 0 -/
theorem line_through_circle_center_parallel_to_given_line :
  ∃ (cx cy : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - cx)^2 + (y - cy)^2 = cx^2 + cy^2) ∧
    (given_line cx cy → target_line cx cy) ∧
    (∀ (x y : ℝ), given_line x y → ∃ (k : ℝ), target_line (x + k) (y + 2*k)) :=
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_given_line_l3881_388199


namespace NUMINAMATH_CALUDE_green_square_area_percentage_l3881_388161

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  side : ℝ
  crossWidth : ℝ
  crossArea : ℝ
  greenSide : ℝ

/-- The cross is symmetric and occupies 49% of the flag's area -/
def isValidCrossFlag (flag : CrossFlag) : Prop :=
  flag.crossArea = 0.49 * flag.side^2 ∧
  flag.greenSide = 2 * flag.crossWidth

/-- Theorem: The green square occupies 6.01% of the flag's area -/
theorem green_square_area_percentage (flag : CrossFlag) 
  (h : isValidCrossFlag flag) : 
  (flag.greenSide^2) / (flag.side^2) = 0.0601 := by
  sorry

end NUMINAMATH_CALUDE_green_square_area_percentage_l3881_388161


namespace NUMINAMATH_CALUDE_geoffreys_birthday_money_l3881_388156

/-- The amount of money Geoffrey received from his grandmother -/
def grandmothers_gift : ℤ := 70

/-- The amount of money Geoffrey received from his aunt -/
def aunts_gift : ℤ := 25

/-- The amount of money Geoffrey received from his uncle -/
def uncles_gift : ℤ := 30

/-- The total amount Geoffrey had in his wallet after receiving gifts -/
def total_in_wallet : ℤ := 125

/-- The cost of each video game -/
def game_cost : ℤ := 35

/-- The number of games Geoffrey bought -/
def number_of_games : ℤ := 3

/-- The amount of money Geoffrey had left after buying the games -/
def money_left : ℤ := 20

theorem geoffreys_birthday_money :
  grandmothers_gift + aunts_gift + uncles_gift = total_in_wallet - (game_cost * number_of_games - money_left) :=
by sorry

end NUMINAMATH_CALUDE_geoffreys_birthday_money_l3881_388156


namespace NUMINAMATH_CALUDE_box_volume_increase_l3881_388197

-- Define the properties of the rectangular box
def rectangular_box (l w h : ℝ) : Prop :=
  l * w * h = 5400 ∧
  2 * (l * w + w * h + h * l) = 2352 ∧
  4 * (l + w + h) = 240

-- State the theorem
theorem box_volume_increase (l w h : ℝ) :
  rectangular_box l w h →
  (l + 2) * (w + 2) * (h + 2) = 8054 :=
by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l3881_388197


namespace NUMINAMATH_CALUDE_min_product_reciprocal_sum_l3881_388106

theorem min_product_reciprocal_sum (a b : ℕ+) (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = 1/9) : 
  (∀ c d : ℕ+, (c : ℚ)⁻¹ + (3 * d : ℚ)⁻¹ = 1/9 → c * d ≥ a * b) ∧ a * b = 108 := by
  sorry

end NUMINAMATH_CALUDE_min_product_reciprocal_sum_l3881_388106


namespace NUMINAMATH_CALUDE_edward_picked_three_l3881_388126

/-- The number of pieces of paper Olivia picked up -/
def olivia_pieces : ℕ := 16

/-- The total number of pieces of paper picked up by Olivia and Edward -/
def total_pieces : ℕ := 19

/-- The number of pieces of paper Edward picked up -/
def edward_pieces : ℕ := total_pieces - olivia_pieces

theorem edward_picked_three : edward_pieces = 3 := by
  sorry

end NUMINAMATH_CALUDE_edward_picked_three_l3881_388126


namespace NUMINAMATH_CALUDE_white_tiles_count_l3881_388134

theorem white_tiles_count (total : ℕ) (yellow : ℕ) (purple : ℕ) 
  (h_total : total = 20)
  (h_yellow : yellow = 3)
  (h_purple : purple = 6) :
  total - (yellow + (yellow + 1) + purple) = 7 := by
  sorry

end NUMINAMATH_CALUDE_white_tiles_count_l3881_388134


namespace NUMINAMATH_CALUDE_parallel_sides_implies_parallelogram_equal_sides_implies_parallelogram_one_pair_parallel_equal_implies_parallelogram_equal_diagonals_implies_parallelogram_l3881_388141

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the conditions
def opposite_sides_parallel (q : Quadrilateral) : Prop := sorry
def opposite_sides_equal (q : Quadrilateral) : Prop := sorry
def one_pair_parallel_and_equal (q : Quadrilateral) : Prop := sorry
def diagonals_equal (q : Quadrilateral) : Prop := sorry

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Theorem statements
theorem parallel_sides_implies_parallelogram (q : Quadrilateral) :
  opposite_sides_parallel q → is_parallelogram q := by sorry

theorem equal_sides_implies_parallelogram (q : Quadrilateral) :
  opposite_sides_equal q → is_parallelogram q := by sorry

theorem one_pair_parallel_equal_implies_parallelogram (q : Quadrilateral) :
  one_pair_parallel_and_equal q → is_parallelogram q := by sorry

theorem equal_diagonals_implies_parallelogram (q : Quadrilateral) :
  diagonals_equal q → is_parallelogram q := by sorry

end NUMINAMATH_CALUDE_parallel_sides_implies_parallelogram_equal_sides_implies_parallelogram_one_pair_parallel_equal_implies_parallelogram_equal_diagonals_implies_parallelogram_l3881_388141


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_digits_l3881_388114

def is_divisible_by_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10).filter (· ≠ 0) → n % d = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_digits :
  ∀ n, is_four_digit n → is_divisible_by_digits n → n ≥ 1362 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_digits_l3881_388114


namespace NUMINAMATH_CALUDE_investment_interest_rate_l3881_388105

theorem investment_interest_rate 
  (total_investment : ℝ) 
  (rate1 rate2 : ℝ) 
  (h1 : total_investment = 6000)
  (h2 : rate1 = 0.05)
  (h3 : rate2 = 0.07)
  (h4 : ∃ (part1 part2 : ℝ), 
    part1 + part2 = total_investment ∧ 
    part1 * rate1 = part2 * rate2) :
  (rate1 * (total_investment - (rate2 * total_investment) / (rate1 + rate2)) + 
   rate2 * ((rate1 * total_investment) / (rate1 + rate2))) / total_investment = 0.05833 :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l3881_388105


namespace NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l3881_388195

theorem longest_segment_in_quarter_circle (d : ℝ) (h : d = 16) :
  let r := d / 2
  let chord_length := r * Real.sqrt 2
  chord_length ^ 2 = 128 := by sorry

end NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l3881_388195


namespace NUMINAMATH_CALUDE_arrange_four_math_four_history_l3881_388162

/-- The number of ways to arrange books on a shelf --/
def arrange_books (n_math : ℕ) (n_history : ℕ) : ℕ :=
  if n_math ≥ 2 then
    n_math * (n_math - 1) * (n_math + n_history - 2).factorial
  else
    0

/-- Theorem: Arranging 4 math books and 4 history books with math books on both ends --/
theorem arrange_four_math_four_history :
  arrange_books 4 4 = 8640 := by
  sorry

end NUMINAMATH_CALUDE_arrange_four_math_four_history_l3881_388162


namespace NUMINAMATH_CALUDE_quadratic_solution_set_implies_coefficients_l3881_388177

-- Define the quadratic function
def f (a c x : ℝ) : ℝ := a * x^2 + 5 * x + c

-- Define the solution set condition
def solution_set (a c : ℝ) : Prop :=
  ∀ x : ℝ, f a c x > 0 ↔ 1/3 < x ∧ x < 1/2

-- Theorem statement
theorem quadratic_solution_set_implies_coefficients :
  ∀ a c : ℝ, solution_set a c → a = -6 ∧ c = -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_implies_coefficients_l3881_388177


namespace NUMINAMATH_CALUDE_largest_non_sum_30_composite_l3881_388172

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_sum_of_multiple_30_and_composite (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_sum_30_composite : 
  (∀ n > 93, is_sum_of_multiple_30_and_composite n) ∧
  ¬is_sum_of_multiple_30_and_composite 93 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_30_composite_l3881_388172


namespace NUMINAMATH_CALUDE_road_trip_time_calculation_l3881_388111

/-- Represents the road trip problem -/
theorem road_trip_time_calculation 
  (freeway_distance : ℝ) 
  (mountain_distance : ℝ) 
  (mountain_time : ℝ) 
  (speed_ratio : ℝ) :
  freeway_distance = 120 →
  mountain_distance = 25 →
  mountain_time = 75 →
  speed_ratio = 4 →
  let mountain_speed := mountain_distance / mountain_time
  let freeway_speed := speed_ratio * mountain_speed
  let freeway_time := freeway_distance / freeway_speed
  freeway_time + mountain_time = 165 :=
by sorry

end NUMINAMATH_CALUDE_road_trip_time_calculation_l3881_388111


namespace NUMINAMATH_CALUDE_age_difference_l3881_388110

def arun_age : ℕ := 60

def gokul_age (a : ℕ) : ℕ := (a - 6) / 18

def madan_age (g : ℕ) : ℕ := g + 5

theorem age_difference : 
  madan_age (gokul_age arun_age) - gokul_age arun_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3881_388110
