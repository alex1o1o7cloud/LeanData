import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l251_25166

theorem simplify_expression (x y : ℝ) : 3*x + 4*x + 5*y + 2*y = 7*x + 7*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l251_25166


namespace NUMINAMATH_CALUDE_problem_solution_l251_25165

def A : Set ℝ := {x | x^2 - 2*x - 15 > 0}
def B : Set ℝ := {x | x - 6 < 0}

theorem problem_solution :
  (∀ m : ℝ, m ∈ A ↔ (m < -3 ∨ m > 5)) ∧
  (∀ m : ℝ, (m ∈ A ∨ m ∈ B) ∧ (m ∈ A ∧ m ∈ B) ↔ (m < -3 ∨ (5 < m ∧ m < 6))) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l251_25165


namespace NUMINAMATH_CALUDE_power_function_through_point_l251_25155

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_through_point :
  ∀ f : ℝ → ℝ, isPowerFunction f → f 2 = Real.sqrt 2 →
  ∀ x : ℝ, x ≥ 0 → f x = Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l251_25155


namespace NUMINAMATH_CALUDE_problem_statement_l251_25185

theorem problem_statement : (3150 - 3030)^2 / 144 = 100 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l251_25185


namespace NUMINAMATH_CALUDE_fairCoin_threeFlips_oneTwoTails_l251_25107

/-- Probability of getting k successes in n trials with probability p for each trial -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)

/-- A fair coin has probability 0.5 of landing tails -/
def fairCoinProbability : ℝ := 0.5

/-- The number of consecutive coin flips -/
def numberOfFlips : ℕ := 3

theorem fairCoin_threeFlips_oneTwoTails :
  binomialProbability numberOfFlips 1 fairCoinProbability +
  binomialProbability numberOfFlips 2 fairCoinProbability = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_fairCoin_threeFlips_oneTwoTails_l251_25107


namespace NUMINAMATH_CALUDE_onion_harvest_bags_per_trip_l251_25195

/-- Calculates the number of bags carried per trip given the total harvest weight,
    weight per bag, and number of trips. -/
def bagsPerTrip (totalHarvest : ℕ) (weightPerBag : ℕ) (numTrips : ℕ) : ℕ :=
  (totalHarvest / weightPerBag) / numTrips

/-- Theorem stating that given the specific conditions of Titan's father's onion harvest,
    the number of bags carried per trip is 10. -/
theorem onion_harvest_bags_per_trip :
  bagsPerTrip 10000 50 20 = 10 := by
  sorry

#eval bagsPerTrip 10000 50 20

end NUMINAMATH_CALUDE_onion_harvest_bags_per_trip_l251_25195


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l251_25117

def population : ℕ := 1203
def sample_size : ℕ := 40

theorem systematic_sampling_interval :
  ∃ (k : ℕ) (eliminated : ℕ),
    eliminated ≤ sample_size ∧
    (population - eliminated) % sample_size = 0 ∧
    k = (population - eliminated) / sample_size ∧
    k = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l251_25117


namespace NUMINAMATH_CALUDE_fraction_problem_l251_25103

theorem fraction_problem (w x y F : ℝ) 
  (h1 : 5 / w + F = 5 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) : 
  F = 10 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l251_25103


namespace NUMINAMATH_CALUDE_min_value_expression_l251_25196

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_prod : x * y * z = 108) : 
  x^2 + 9*x*y + 9*y^2 + 3*z^2 ≥ 324 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l251_25196


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l251_25123

/-- Given a repeating decimal 3.565656..., prove it equals 353/99 -/
theorem repeating_decimal_to_fraction : 
  ∀ (x : ℚ), (∃ (n : ℕ), x = 3 + (56 : ℚ) / (10^2 - 1) / 10^n) → x = 353 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l251_25123


namespace NUMINAMATH_CALUDE_some_value_theorem_l251_25110

theorem some_value_theorem (w x y : ℝ) (h1 : (w + x) / 2 = 0.5) (h2 : w * x = y) :
  ∃ some_value : ℝ, 3 / w + some_value = 3 / y ∧ some_value = 6 := by
  sorry

end NUMINAMATH_CALUDE_some_value_theorem_l251_25110


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_l251_25177

/-- The cost of paint per kg for a cube painting problem -/
theorem paint_cost_per_kg (coverage : ℝ) (total_cost : ℝ) (side_length : ℝ) : 
  coverage = 20 →
  total_cost = 10800 →
  side_length = 30 →
  (total_cost / (6 * side_length^2 / coverage)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_l251_25177


namespace NUMINAMATH_CALUDE_composition_computation_stages_l251_25142

/-- A structure representing a linear function f(x) = px + q -/
structure LinearFunction where
  p : ℝ
  q : ℝ

/-- Computes the composition of n linear functions -/
def compose_linear_functions (fs : List LinearFunction) (x : ℝ) : ℝ := sorry

/-- Represents a computation stage -/
inductive Stage
  | init : Stage
  | next : Stage → Stage

/-- Counts the number of stages -/
def stage_count : Stage → Nat
  | Stage.init => 0
  | Stage.next s => stage_count s + 1

/-- Theorem stating that the composition can be computed in no more than 30 stages -/
theorem composition_computation_stages
  (fs : List LinearFunction)
  (h_length : fs.length = 1000)
  (x₀ : ℝ) :
  ∃ (s : Stage), stage_count s ≤ 30 ∧ compose_linear_functions fs x₀ = sorry :=
by sorry

end NUMINAMATH_CALUDE_composition_computation_stages_l251_25142


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l251_25199

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15) ∧ 
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 30 :=
by sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l251_25199


namespace NUMINAMATH_CALUDE_factor_expression_l251_25112

theorem factor_expression (y : ℝ) : 64 - 16 * y^3 = 16 * (2 - y) * (4 + 2*y + y^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l251_25112


namespace NUMINAMATH_CALUDE_seventeen_in_base_three_l251_25156

/-- Represents a number in base 3 as a list of digits (least significant digit first) -/
def BaseThreeRepresentation := List Nat

/-- Converts a base 3 representation to its decimal value -/
def toDecimal (rep : BaseThreeRepresentation) : Nat :=
  rep.enum.foldl (fun acc (i, digit) => acc + digit * (3^i)) 0

/-- Theorem: The base-3 representation of 17 is [2, 2, 1] (which represents 122₃) -/
theorem seventeen_in_base_three :
  ∃ (rep : BaseThreeRepresentation), toDecimal rep = 17 ∧ rep = [2, 2, 1] := by
  sorry

end NUMINAMATH_CALUDE_seventeen_in_base_three_l251_25156


namespace NUMINAMATH_CALUDE_slope_of_line_l251_25160

/-- The slope of the line (x/4) + (y/5) = 1 is -5/4 -/
theorem slope_of_line (x y : ℝ) : 
  (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5/4) := by
sorry

end NUMINAMATH_CALUDE_slope_of_line_l251_25160


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l251_25144

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 = 4 →
  a 4 = 2 →
  a 8 = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l251_25144


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l251_25134

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/11) 
  (h2 : x - y = 1/55) : 
  x^2 - y^2 = 1/121 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l251_25134


namespace NUMINAMATH_CALUDE_no_arithmetic_sqrt_neg_nine_l251_25198

-- Define the concept of an arithmetic square root
def arithmetic_sqrt (x : ℝ) : Prop :=
  ∃ y : ℝ, y * y = x ∧ y ≥ 0

-- Theorem stating that the arithmetic square root of -9 does not exist
theorem no_arithmetic_sqrt_neg_nine :
  ¬ arithmetic_sqrt (-9) :=
sorry

end NUMINAMATH_CALUDE_no_arithmetic_sqrt_neg_nine_l251_25198


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l251_25194

/-- The line equation mx - y + 2m + 1 = 0 passes through the point (-2, 1) for all values of m. -/
theorem fixed_point_of_line (m : ℝ) : m * (-2) - 1 + 2 * m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l251_25194


namespace NUMINAMATH_CALUDE_total_sticks_is_2310_l251_25170

/-- Represents the number of sticks of gum in 12 brown boxes, including two with only 3 cartons each --/
def total_sticks : ℕ :=
  let sticks_per_pack : ℕ := 5
  let packs_per_carton : ℕ := 7
  let cartons_per_full_box : ℕ := 6
  let cartons_per_partial_box : ℕ := 3
  let total_boxes : ℕ := 12
  let partial_boxes : ℕ := 2
  let full_boxes : ℕ := total_boxes - partial_boxes
  
  let sticks_per_carton : ℕ := sticks_per_pack * packs_per_carton
  let sticks_per_full_box : ℕ := sticks_per_carton * cartons_per_full_box
  let sticks_per_partial_box : ℕ := sticks_per_carton * cartons_per_partial_box
  
  (full_boxes * sticks_per_full_box) + (partial_boxes * sticks_per_partial_box)

theorem total_sticks_is_2310 : total_sticks = 2310 := by
  sorry

end NUMINAMATH_CALUDE_total_sticks_is_2310_l251_25170


namespace NUMINAMATH_CALUDE_product_sum_theorem_l251_25121

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a + b + c = 14) : 
  a*b + b*c + a*c = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l251_25121


namespace NUMINAMATH_CALUDE_divides_or_divides_l251_25167

theorem divides_or_divides (m n : ℤ) (h : m.lcm n + m.gcd n = m + n) :
  n ∣ m ∨ m ∣ n := by sorry

end NUMINAMATH_CALUDE_divides_or_divides_l251_25167


namespace NUMINAMATH_CALUDE_expression_simplification_l251_25132

theorem expression_simplification (x y : ℚ) 
  (hx : x = 1/2) (hy : y = 8) : 
  y * (5 * x - 4 * y) + (x - 2 * y)^2 = 17/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l251_25132


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l251_25184

theorem six_digit_numbers_with_zero (total_six_digit : ℕ) (six_digit_no_zero : ℕ) :
  total_six_digit = 9 * 10^5 →
  six_digit_no_zero = 9^6 →
  total_six_digit - six_digit_no_zero = 368559 := by
sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l251_25184


namespace NUMINAMATH_CALUDE_no_number_decreases_by_1981_l251_25173

theorem no_number_decreases_by_1981 : 
  ¬ ∃ (N : ℕ), 
    ∃ (M : ℕ), 
      (N ≠ 0) ∧ 
      (M ≠ 0) ∧
      (N = 1981 * M) ∧
      (∃ (d : ℕ) (k : ℕ), N = d * 10^k + M ∧ 1 ≤ d ∧ d ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_no_number_decreases_by_1981_l251_25173


namespace NUMINAMATH_CALUDE_new_supervisor_salary_is_960_l251_25197

/-- Represents the monthly salary structure of a factory -/
structure FactorySalary where
  initial_avg : ℝ
  old_supervisor_salary : ℝ
  old_supervisor_bonus_rate : ℝ
  worker_increment_rate : ℝ
  old_supervisor_increment_rate : ℝ
  new_avg : ℝ
  new_supervisor_bonus_rate : ℝ
  new_supervisor_increment_rate : ℝ

/-- Calculates the new supervisor's monthly salary -/
def calculate_new_supervisor_salary (fs : FactorySalary) : ℝ :=
  sorry

/-- Theorem stating that given the factory salary conditions, 
    the new supervisor's monthly salary is $960 -/
theorem new_supervisor_salary_is_960 (fs : FactorySalary) 
  (h1 : fs.initial_avg = 430)
  (h2 : fs.old_supervisor_salary = 870)
  (h3 : fs.old_supervisor_bonus_rate = 0.05)
  (h4 : fs.worker_increment_rate = 0.03)
  (h5 : fs.old_supervisor_increment_rate = 0.04)
  (h6 : fs.new_avg = 450)
  (h7 : fs.new_supervisor_bonus_rate = 0.03)
  (h8 : fs.new_supervisor_increment_rate = 0.035) :
  calculate_new_supervisor_salary fs = 960 :=
sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_is_960_l251_25197


namespace NUMINAMATH_CALUDE_inequality_proof_l251_25171

theorem inequality_proof (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l251_25171


namespace NUMINAMATH_CALUDE_range_of_m_l251_25148

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x-1)/3| < 2
def q (x m : ℝ) : Prop := (x-1)^2 < m^2

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (sufficient_not_necessary (p) (q m)) ↔ (-5 < m ∧ m < 5) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l251_25148


namespace NUMINAMATH_CALUDE_sum_value_l251_25183

theorem sum_value (a b : ℝ) (h1 : |a| = 1) (h2 : b = -2) : 
  a + b = -3 ∨ a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_value_l251_25183


namespace NUMINAMATH_CALUDE_range_of_k_l251_25163

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8 = 0

-- Define the line
def line (k x : ℝ) : ℝ := 2*k*x - 2

-- Define the condition for a point on the line to be a valid center
def valid_center (k x : ℝ) : Prop :=
  ∃ (y : ℝ), y = line k x ∧ 
  ∃ (x' y' : ℝ), circle_C x' y' ∧ (x' - x)^2 + (y' - y)^2 ≤ 4

-- Theorem statement
theorem range_of_k :
  ∀ k : ℝ, (∃ x : ℝ, valid_center k x) ↔ 0 ≤ k ∧ k ≤ 6/5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l251_25163


namespace NUMINAMATH_CALUDE_exactly_one_pass_probability_l251_25186

theorem exactly_one_pass_probability (p : ℝ) (hp : p = 1 / 2) :
  let q := 1 - p
  p * q + q * p = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_exactly_one_pass_probability_l251_25186


namespace NUMINAMATH_CALUDE_part_I_part_II_l251_25157

-- Define the function f
def f (a b x : ℝ) := 2 * x^2 - 2 * a * x + b

-- Define set A
def A (a b : ℝ) := {x : ℝ | f a b x > 0}

-- Define set B
def B (t : ℝ) := {x : ℝ | |x - t| ≤ 1}

-- Theorem for part (I)
theorem part_I (a b : ℝ) (h1 : f a b (-1) = -8) (h2 : ∀ x : ℝ, f a b x ≥ f a b (-1)) :
  (Set.univ \ A a b) ∪ B 1 = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
sorry

-- Theorem for part (II)
theorem part_II (a b : ℝ) (h1 : f a b (-1) = -8) (h2 : ∀ x : ℝ, f a b x ≥ f a b (-1)) :
  {t : ℝ | A a b ∩ B t = ∅} = {t : ℝ | -2 ≤ t ∧ t ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_part_I_part_II_l251_25157


namespace NUMINAMATH_CALUDE_equation_solution_l251_25141

theorem equation_solution : ∃ y : ℝ, 
  (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt y)) = (2 + Real.sqrt y) ^ (1/4)) ∧ 
  y = 81/256 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l251_25141


namespace NUMINAMATH_CALUDE_charlies_first_week_usage_l251_25119

/-- Represents Charlie's cell phone plan and usage --/
structure CellPhonePlan where
  baseData : ℝ  -- Base data allowance in GB
  extraCost : ℝ  -- Cost per extra GB
  week2Usage : ℝ  -- Data usage in week 2
  week3Usage : ℝ  -- Data usage in week 3
  week4Usage : ℝ  -- Data usage in week 4
  extraCharge : ℝ  -- Extra charge on the bill

/-- Theorem stating Charlie's first week data usage --/
theorem charlies_first_week_usage (plan : CellPhonePlan) 
  (h1 : plan.baseData = 8)
  (h2 : plan.extraCost = 10)
  (h3 : plan.week2Usage = 3)
  (h4 : plan.week3Usage = 5)
  (h5 : plan.week4Usage = 10)
  (h6 : plan.extraCharge = 120) :
  ∃ (week1Usage : ℝ), week1Usage = 2 ∧ 
    week1Usage + plan.week2Usage + plan.week3Usage + plan.week4Usage = 
    plan.baseData + plan.extraCharge / plan.extraCost :=
  sorry

end NUMINAMATH_CALUDE_charlies_first_week_usage_l251_25119


namespace NUMINAMATH_CALUDE_five_level_pieces_l251_25168

/-- Calculates the number of pieces in a square-based pyramid -/
def pyramid_pieces (levels : ℕ) : ℕ :=
  let rods := levels * (levels + 1) * 2
  let connectors := levels * 4
  rods + connectors

/-- Properties of a two-level square-based pyramid -/
axiom two_level_total : pyramid_pieces 2 = 20
axiom two_level_rods : 2 * (2 + 1) * 2 = 12
axiom two_level_connectors : 2 * 4 = 8

/-- Theorem: A five-level square-based pyramid requires 80 pieces -/
theorem five_level_pieces : pyramid_pieces 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_five_level_pieces_l251_25168


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l251_25147

-- Problem 1
theorem problem_one : Real.sqrt 3 - Real.sqrt 3 * (1 - Real.sqrt 3) = 3 := by sorry

-- Problem 2
theorem problem_two : (Real.sqrt 3 - 2)^2 + Real.sqrt 12 + 6 * Real.sqrt (1/3) = 7 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l251_25147


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l251_25151

def grape_purchase : ℕ := 8
def mango_purchase : ℕ := 10
def mango_rate : ℕ := 55
def total_paid : ℕ := 1110

theorem grape_rate_calculation :
  ∃ (grape_rate : ℕ),
    grape_rate * grape_purchase + mango_rate * mango_purchase = total_paid ∧
    grape_rate = 70 := by
  sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l251_25151


namespace NUMINAMATH_CALUDE_prob_at_least_one_heart_or_king_l251_25106

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of cards that are either hearts or kings
def heart_or_king : ℕ := 16

-- Define the probability of not choosing a heart or king in one draw
def prob_not_heart_or_king : ℚ := (total_cards - heart_or_king) / total_cards

-- Theorem statement
theorem prob_at_least_one_heart_or_king :
  1 - prob_not_heart_or_king ^ 2 = 88 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_heart_or_king_l251_25106


namespace NUMINAMATH_CALUDE_pen_cost_l251_25178

theorem pen_cost (pen_price pencil_price : ℚ) 
  (h1 : 6 * pen_price + 2 * pencil_price = 348/100)
  (h2 : 3 * pen_price + 4 * pencil_price = 234/100) :
  pen_price = 51/100 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l251_25178


namespace NUMINAMATH_CALUDE_probability_ones_not_adjacent_l251_25100

def total_arrangements : ℕ := 10

def favorable_arrangements : ℕ := 6

theorem probability_ones_not_adjacent :
  (favorable_arrangements : ℚ) / total_arrangements = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_ones_not_adjacent_l251_25100


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l251_25192

theorem sum_of_two_numbers (s l : ℝ) : 
  s = 3.5 →
  l = 3 * s →
  s + l = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l251_25192


namespace NUMINAMATH_CALUDE_degree_of_polynomial_l251_25136

def p (x : ℝ) : ℝ := (2*x^5 - 3*x^3 + x^2 - 14) * (3*x^11 - 9*x^8 + 9*x^5 + 30) - (x^3 + 5)^7

theorem degree_of_polynomial : 
  ∃ (a : ℝ) (q : ℝ → ℝ), a ≠ 0 ∧ 
  (∀ (x : ℝ), p x = a * x^21 + q x) ∧ 
  (∃ (N : ℝ), ∀ (x : ℝ), |x| > N → |q x| < |a| * |x|^21) :=
sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_l251_25136


namespace NUMINAMATH_CALUDE_bong_paint_time_l251_25128

def jay_time : ℝ := 2
def combined_time : ℝ := 1.2

theorem bong_paint_time :
  ∀ bong_time : ℝ,
  (1 / jay_time + 1 / bong_time = 1 / combined_time) →
  bong_time = 3 := by
sorry

end NUMINAMATH_CALUDE_bong_paint_time_l251_25128


namespace NUMINAMATH_CALUDE_alchemists_less_than_half_l251_25101

theorem alchemists_less_than_half (k : ℕ) (c a : ℕ) : 
  k > 0 → 
  k = c + a → 
  c > a → 
  a < k / 2 := by
sorry

end NUMINAMATH_CALUDE_alchemists_less_than_half_l251_25101


namespace NUMINAMATH_CALUDE_equation_equivalence_implies_mnp_30_l251_25124

theorem equation_equivalence_implies_mnp_30 
  (b x z c : ℝ) (m n p : ℤ) 
  (h : ∀ x z c, b^8*x*z - b^7*z - b^6*x = b^5*(c^5 - 1) ↔ (b^m*x-b^n)*(b^p*z-b^3)=b^5*c^5) : 
  m * n * p = 30 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_implies_mnp_30_l251_25124


namespace NUMINAMATH_CALUDE_linear_function_translation_l251_25153

/-- A linear function passing through a specific point -/
def passes_through (b : ℝ) : Prop :=
  3 = 2 + b

/-- The correct value of b for the translated line -/
def correct_b : ℝ := 1

/-- Theorem stating that the linear function passes through (2, 3) iff b = 1 -/
theorem linear_function_translation :
  passes_through correct_b ∧ 
  (∀ b : ℝ, passes_through b → b = correct_b) :=
sorry

end NUMINAMATH_CALUDE_linear_function_translation_l251_25153


namespace NUMINAMATH_CALUDE_optimal_shelf_arrangement_l251_25115

def math_books : ℕ := 130
def portuguese_books : ℕ := 195

theorem optimal_shelf_arrangement :
  ∃ (n : ℕ), n > 0 ∧
  n ∣ math_books ∧
  n ∣ portuguese_books ∧
  (∀ m : ℕ, m > n → ¬(m ∣ math_books ∧ m ∣ portuguese_books)) ∧
  n = 65 := by
  sorry

end NUMINAMATH_CALUDE_optimal_shelf_arrangement_l251_25115


namespace NUMINAMATH_CALUDE_f_value_at_ln_one_third_l251_25174

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x) / (2^x + 1) + a * x

theorem f_value_at_ln_one_third (a : ℝ) :
  (f a (Real.log 3) = 3) → (f a (Real.log (1/3)) = -2) := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_ln_one_third_l251_25174


namespace NUMINAMATH_CALUDE_second_rate_is_five_percent_l251_25193

-- Define the total amount, first part, and interest rates
def total_amount : ℚ := 3200
def first_part : ℚ := 800
def first_rate : ℚ := 3 / 100
def total_interest : ℚ := 144

-- Define the second part
def second_part : ℚ := total_amount - first_part

-- Define the interest from the first part
def interest_first : ℚ := first_part * first_rate

-- Define the interest from the second part
def interest_second : ℚ := total_interest - interest_first

-- Define the interest rate of the second part
def second_rate : ℚ := interest_second / second_part

-- Theorem to prove
theorem second_rate_is_five_percent : second_rate = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_second_rate_is_five_percent_l251_25193


namespace NUMINAMATH_CALUDE_book_club_meeting_lcm_l251_25109

/-- The least common multiple of 5, 6, 8, 9, and 10 is 360 -/
theorem book_club_meeting_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 10))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_book_club_meeting_lcm_l251_25109


namespace NUMINAMATH_CALUDE_pauls_reading_rate_l251_25111

theorem pauls_reading_rate (total_books : ℕ) (total_weeks : ℕ) 
  (h1 : total_books = 20) (h2 : total_weeks = 5) : 
  total_books / total_weeks = 4 := by
sorry

end NUMINAMATH_CALUDE_pauls_reading_rate_l251_25111


namespace NUMINAMATH_CALUDE_tangent_line_equation_l251_25139

theorem tangent_line_equation (x y : ℝ) :
  x < 0 ∧ y > 0 ∧  -- P is in the second quadrant
  y = x^3 - 10*x + 3 ∧  -- P is on the curve
  3*x^2 - 10 = 2  -- Slope of tangent line is 2
  →
  ∃ (a b : ℝ), a = 2 ∧ b = 19 ∧ ∀ (x' y' : ℝ), y' = a*x' + b  -- Equation of tangent line
  :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l251_25139


namespace NUMINAMATH_CALUDE_F_2_f_3_equals_341_l251_25182

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 2
def F (a b : ℝ) : ℝ := b^3 - a

-- State the theorem
theorem F_2_f_3_equals_341 : F 2 (f 3) = 341 := by sorry

end NUMINAMATH_CALUDE_F_2_f_3_equals_341_l251_25182


namespace NUMINAMATH_CALUDE_circle_radius_from_intersecting_line_l251_25164

/-- Given a line intersecting a circle, prove the radius of the circle --/
theorem circle_radius_from_intersecting_line (r : ℝ) :
  let line := {(x, y) : ℝ × ℝ | x - Real.sqrt 3 * y + 8 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  ∃ (A B : ℝ × ℝ), A ∈ line ∧ A ∈ circle ∧ B ∈ line ∧ B ∈ circle ∧
    ‖A - B‖ = 6 →
  r = 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_from_intersecting_line_l251_25164


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_inequality_l251_25135

/-- An arithmetic sequence of 8 terms with positive values and non-zero common difference -/
structure ArithmeticSequence8 where
  a : Fin 8 → ℝ
  positive : ∀ i, a i > 0
  common_diff : ℝ
  common_diff_neq_zero : common_diff ≠ 0
  is_arithmetic : ∀ i j, i < j → a j - a i = common_diff * (j - i)

/-- For an arithmetic sequence of 8 terms with positive values and non-zero common difference,
    the product of the first and last terms is less than the product of the fourth and fifth terms -/
theorem arithmetic_sequence_product_inequality (seq : ArithmeticSequence8) :
  seq.a 0 * seq.a 7 < seq.a 3 * seq.a 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_inequality_l251_25135


namespace NUMINAMATH_CALUDE_factor_expression_l251_25127

theorem factor_expression (x : ℝ) : 100 * x^23 + 225 * x^46 = 25 * x^23 * (4 + 9 * x^23) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l251_25127


namespace NUMINAMATH_CALUDE_work_completion_time_l251_25180

/-- The number of days it takes person A to complete the work alone -/
def days_A : ℝ := 18

/-- The number of days it takes person B to complete the work alone -/
def days_B : ℝ := 24

/-- The fraction of work completed when A and B work together for 2 days -/
def work_completed : ℝ := 0.19444444444444442

theorem work_completion_time :
  (2 * (1 / days_A + 1 / days_B) = work_completed) →
  days_A = 18 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l251_25180


namespace NUMINAMATH_CALUDE_solve_equation_l251_25138

theorem solve_equation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 / x) + (4 / y) = 1) : x = (3 * y) / (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l251_25138


namespace NUMINAMATH_CALUDE_adult_attraction_cost_is_four_l251_25120

/-- Represents the cost structure and family composition for a park visit -/
structure ParkVisit where
  entrance_fee : ℕ
  child_attraction_fee : ℕ
  num_children : ℕ
  num_parents : ℕ
  num_grandparents : ℕ
  total_cost : ℕ

/-- Calculates the cost of an adult attraction ticket given the park visit details -/
def adult_attraction_cost (visit : ParkVisit) : ℕ :=
  let total_people := visit.num_children + visit.num_parents + visit.num_grandparents
  let entrance_cost := total_people * visit.entrance_fee
  let children_attraction_cost := visit.num_children * visit.child_attraction_fee
  let adult_attraction_total := visit.total_cost - entrance_cost - children_attraction_cost
  let num_adults := visit.num_parents + visit.num_grandparents
  adult_attraction_total / num_adults

theorem adult_attraction_cost_is_four : 
  adult_attraction_cost ⟨5, 2, 4, 2, 1, 55⟩ = 4 := by
  sorry

end NUMINAMATH_CALUDE_adult_attraction_cost_is_four_l251_25120


namespace NUMINAMATH_CALUDE_veg_eaters_count_l251_25175

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  onlyVeg : ℕ
  bothVegNonVeg : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def totalVegEaters (fd : FamilyDiet) : ℕ :=
  fd.onlyVeg + fd.bothVegNonVeg

/-- Theorem: The number of people who eat veg in the family is 21 -/
theorem veg_eaters_count (fd : FamilyDiet) 
  (h1 : fd.onlyVeg = 13)
  (h2 : fd.bothVegNonVeg = 8) : 
  totalVegEaters fd = 21 := by
  sorry

end NUMINAMATH_CALUDE_veg_eaters_count_l251_25175


namespace NUMINAMATH_CALUDE_helen_cookies_l251_25113

/-- The number of cookies Helen baked in total -/
def total_cookies : ℕ := 574

/-- The number of cookies Helen baked this morning -/
def morning_cookies : ℕ := 139

/-- The number of cookies Helen baked yesterday -/
def yesterday_cookies : ℕ := total_cookies - morning_cookies

theorem helen_cookies : yesterday_cookies = 435 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l251_25113


namespace NUMINAMATH_CALUDE_flowers_purchase_l251_25161

theorem flowers_purchase (dozen_bought : ℕ) : 
  (∀ d : ℕ, 12 * d + 2 * d = 14 * d) →
  12 * dozen_bought + 2 * dozen_bought = 42 →
  dozen_bought = 3 := by
  sorry

end NUMINAMATH_CALUDE_flowers_purchase_l251_25161


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2016th_term_l251_25191

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_2016th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_incr : ∀ n : ℕ, a (n + 1) > a n) 
  (h_first : a 1 = 1) 
  (h_geom : (a 4)^2 = a 2 * a 8) :
  a 2016 = 2016 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2016th_term_l251_25191


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l251_25104

open Set

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {1, 3, 5}

-- Define set B
def B : Set Nat := {2, 5, 7}

-- Theorem statement
theorem complement_A_intersect_B : (U \ A) ∩ B = {2, 7} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l251_25104


namespace NUMINAMATH_CALUDE_hen_duck_speed_ratio_l251_25172

/-- The number of leaps a hen makes for every 8 leaps of a duck -/
def hen_leaps : ℕ := 6

/-- The number of duck leaps that equal 3 hen leaps -/
def duck_leaps_equal : ℕ := 4

/-- The number of hen leaps that equal 4 duck leaps -/
def hen_leaps_equal : ℕ := 3

/-- The number of duck leaps for which we compare hen leaps -/
def duck_comparison_leaps : ℕ := 8

theorem hen_duck_speed_ratio :
  (hen_leaps : ℚ) / duck_comparison_leaps = 1 := by
  sorry

end NUMINAMATH_CALUDE_hen_duck_speed_ratio_l251_25172


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l251_25190

theorem inequality_and_minimum_value (a b m n : ℝ) (x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hm : m > 0) (hn : n > 0) 
  (hx : 0 < x ∧ x < 1/2) : 
  (m^2 / a + n^2 / b ≥ (m + n)^2 / (a + b)) ∧
  (∃ (min_val : ℝ), min_val = 25 ∧ 
    ∀ y, 0 < y ∧ y < 1/2 → 2/y + 9/(1-2*y) ≥ min_val) ∧
  (2/((1:ℝ)/5) + 9/(1-2*((1:ℝ)/5)) = 25) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l251_25190


namespace NUMINAMATH_CALUDE_valid_12_letter_words_mod_1000_l251_25118

/-- Represents a letter in Zuminglish -/
inductive ZumLetter
| M
| O
| P

/-- Represents a Zuminglish word -/
def ZumWord := List ZumLetter

/-- Checks if a letter is a vowel -/
def isVowel (l : ZumLetter) : Bool :=
  match l with
  | ZumLetter.O => true
  | _ => false

/-- Checks if a Zuminglish word is valid -/
def isValidWord (w : ZumWord) : Bool :=
  sorry

/-- Counts the number of valid n-letter Zuminglish words -/
def countValidWords (n : Nat) : Nat :=
  sorry

/-- The main theorem: number of valid 12-letter Zuminglish words modulo 1000 -/
theorem valid_12_letter_words_mod_1000 :
  countValidWords 12 % 1000 = 416 := by
  sorry

end NUMINAMATH_CALUDE_valid_12_letter_words_mod_1000_l251_25118


namespace NUMINAMATH_CALUDE_expression_value_l251_25102

theorem expression_value (x y : ℝ) (h1 : x ≠ y) 
  (h2 : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (x * y + 1)) : 
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (x * y + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l251_25102


namespace NUMINAMATH_CALUDE_indira_cricket_time_l251_25130

def total_minutes : ℕ := 15000

def amaya_cricket_minutes : ℕ := 75 * 5 * 4
def amaya_basketball_minutes : ℕ := 45 * 2 * 4
def sean_cricket_minutes : ℕ := 65 * 14
def sean_basketball_minutes : ℕ := 55 * 4

def amaya_sean_total : ℕ := amaya_cricket_minutes + amaya_basketball_minutes + sean_cricket_minutes + sean_basketball_minutes

theorem indira_cricket_time :
  total_minutes - amaya_sean_total = 12010 :=
sorry

end NUMINAMATH_CALUDE_indira_cricket_time_l251_25130


namespace NUMINAMATH_CALUDE_box_comparison_l251_25181

-- Define a structure for a box with three dimensions
structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the relation "smaller than" for boxes
def smaller (k p : Box) : Prop :=
  (k.x ≤ p.x ∧ k.y ≤ p.y ∧ k.z ≤ p.z) ∨
  (k.x ≤ p.x ∧ k.y ≤ p.z ∧ k.z ≤ p.y) ∨
  (k.x ≤ p.y ∧ k.y ≤ p.x ∧ k.z ≤ p.z) ∨
  (k.x ≤ p.y ∧ k.y ≤ p.z ∧ k.z ≤ p.x) ∨
  (k.x ≤ p.z ∧ k.y ≤ p.x ∧ k.z ≤ p.y) ∨
  (k.x ≤ p.z ∧ k.y ≤ p.y ∧ k.z ≤ p.x)

-- Define boxes A, B, and C
def A : Box := ⟨5, 6, 3⟩
def B : Box := ⟨1, 5, 4⟩
def C : Box := ⟨2, 2, 3⟩

-- Theorem to prove A > B and C < A
theorem box_comparison : smaller B A ∧ smaller C A := by
  sorry


end NUMINAMATH_CALUDE_box_comparison_l251_25181


namespace NUMINAMATH_CALUDE_subset_relation_l251_25189

theorem subset_relation (x y : ℝ) :
  (abs x + abs y < 1) →
  (Real.sqrt ((x - 1/2)^2 + (y + 1/2)^2) + Real.sqrt ((x + 1/2)^2 + (y - 1/2)^2) < 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_subset_relation_l251_25189


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l251_25143

/-- An arithmetic sequence with the given properties has a common difference of 2 -/
theorem arithmetic_geometric_sequence_common_difference :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence property
  d ≠ 0 →
  a 1 = 18 →
  (a 1) * (a 8) = (a 4)^2 →  -- geometric sequence property
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l251_25143


namespace NUMINAMATH_CALUDE_largest_whole_number_times_nine_less_than_150_l251_25126

theorem largest_whole_number_times_nine_less_than_150 :
  (∀ n : ℕ, 9 * n < 150 → n ≤ 16) ∧
  (9 * 16 < 150) := by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_times_nine_less_than_150_l251_25126


namespace NUMINAMATH_CALUDE_compare_exponential_expressions_l251_25133

theorem compare_exponential_expressions :
  let a := 4^(1/4)
  let b := 5^(1/5)
  let c := 16^(1/16)
  let d := 25^(1/25)
  (a > b ∧ a > c ∧ a > d) ∧
  (b > c ∧ b > d) :=
by sorry

end NUMINAMATH_CALUDE_compare_exponential_expressions_l251_25133


namespace NUMINAMATH_CALUDE_initial_typists_count_initial_typists_count_proof_l251_25179

/-- Given that some typists can type 40 letters in 20 minutes and 30 typists working at the same rate can complete 180 letters in 1 hour, prove that the number of typists in the initial group is 20. -/
theorem initial_typists_count : ℕ :=
  let some_typists_letters := 40 -- letters typed by some typists in 20 minutes
  let some_typists_time := 20 -- minutes
  let known_typists_count := 30 -- known number of typists
  let known_typists_letters := 180 -- letters typed by known typists in 1 hour
  let known_typists_time := 60 -- minutes (1 hour)
  20

theorem initial_typists_count_proof (some_typists_letters : ℕ) (some_typists_time : ℕ) 
  (known_typists_count : ℕ) (known_typists_letters : ℕ) (known_typists_time : ℕ) :
  some_typists_letters = 40 →
  some_typists_time = 20 →
  known_typists_count = 30 →
  known_typists_letters = 180 →
  known_typists_time = 60 →
  initial_typists_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_initial_typists_count_initial_typists_count_proof_l251_25179


namespace NUMINAMATH_CALUDE_six_term_sequence_count_l251_25154

def sequence_count (n : ℕ) (a b c : ℕ) : ℕ :=
  n.choose c * (n - c).choose b

theorem six_term_sequence_count : sequence_count 6 3 2 1 = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_term_sequence_count_l251_25154


namespace NUMINAMATH_CALUDE_ali_seashells_left_l251_25140

/-- The number of seashells Ali has left after all transactions -/
def seashells_left (initial : ℝ) (given_friends : ℝ) (given_brothers : ℝ) (sold_fraction : ℝ) (traded_fraction : ℝ) : ℝ :=
  let remaining_after_giving := initial - (given_friends + given_brothers)
  let remaining_after_selling := remaining_after_giving * (1 - sold_fraction)
  remaining_after_selling * (1 - traded_fraction)

/-- Theorem stating that Ali has 76.375 seashells left after all transactions -/
theorem ali_seashells_left : 
  seashells_left 385.5 45.75 34.25 (2/3) (1/4) = 76.375 := by sorry

end NUMINAMATH_CALUDE_ali_seashells_left_l251_25140


namespace NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l251_25105

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagon_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - 
  (2 * Q.quadrilateral_faces + 5 * Q.pentagon_faces)

/-- Theorem: A specific convex polyhedron Q has 321 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 70 ∧
    Q.faces = 42 ∧
    Q.triangular_faces = 26 ∧
    Q.quadrilateral_faces = 12 ∧
    Q.pentagon_faces = 4 ∧
    space_diagonals Q = 321 := by
  sorry

end NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l251_25105


namespace NUMINAMATH_CALUDE_ellipse_parabola_properties_l251_25114

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola E
def parabola_E (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the line y = k(x - 4)
def line_k (x y k : ℝ) : Prop := y = k * (x - 4)

-- Define the line x = 1
def line_x1 (x : ℝ) : Prop := x = 1

theorem ellipse_parabola_properties 
  (a b c : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_ecc : c / a = Real.sqrt 3 / 2) 
  (h_focus : ∃ x y, ellipse_C x y a b ∧ parabola_E x y ∧ (x = a ∨ x = -a ∨ y = b ∨ y = -b)) :
  -- 1. Equation of C
  (∀ x y, ellipse_C x y a b ↔ x^2 / 4 + y^2 = 1) ∧
  -- 2. Collinearity of A, P, and N
  (∀ k x_M y_M x_N y_N x_P, 
    k ≠ 0 →
    ellipse_C x_M y_M a b →
    ellipse_C x_N y_N a b →
    line_k x_M y_M k →
    line_k x_N y_N k →
    line_x1 x_P →
    ∃ y_P, line_k x_P y_P k →
    ∃ t, t * (x_P + 2) = 3 ∧ t * y_P = k * (x_N + 2)) ∧
  -- 3. Maximum area of triangle OMN
  (∃ S : ℝ, 
    (∀ k x_M y_M x_N y_N, 
      k ≠ 0 →
      ellipse_C x_M y_M a b →
      ellipse_C x_N y_N a b →
      line_k x_M y_M k →
      line_k x_N y_N k →
      (1/2) * abs (x_M * y_N - x_N * y_M) ≤ S) ∧
    (∃ k x_M y_M x_N y_N,
      k ≠ 0 →
      ellipse_C x_M y_M a b →
      ellipse_C x_N y_N a b →
      line_k x_M y_M k →
      line_k x_N y_N k →
      (1/2) * abs (x_M * y_N - x_N * y_M) = S) ∧
    S = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_properties_l251_25114


namespace NUMINAMATH_CALUDE_functional_equation_solution_l251_25129

/-- A function from rational numbers to rational numbers -/
def RationalFunction := ℚ → ℚ

/-- The functional equation property -/
def SatisfiesEquation (f : RationalFunction) : Prop :=
  ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y

/-- The theorem statement -/
theorem functional_equation_solution :
  ∀ f : RationalFunction, SatisfiesEquation f →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l251_25129


namespace NUMINAMATH_CALUDE_num_ten_bills_is_two_payment_equation_l251_25169

/-- Represents the number of $10 bills used -/
def num_ten_bills : ℕ := sorry

/-- Represents the number of $20 bills used -/
def num_twenty_bills : ℕ := num_ten_bills + 1

/-- The total amount spent in dollars -/
def total_spent : ℕ := 80

/-- Theorem stating that the number of $10 bills used is 2 -/
theorem num_ten_bills_is_two : num_ten_bills = 2 := by
  sorry

/-- The payment equation -/
theorem payment_equation : 
  10 * num_ten_bills + 20 * num_twenty_bills = total_spent := by
  sorry

end NUMINAMATH_CALUDE_num_ten_bills_is_two_payment_equation_l251_25169


namespace NUMINAMATH_CALUDE_calculate_expression_l251_25137

theorem calculate_expression (x y : ℝ) : 3 * x^2 * y * (-2 * x * y)^2 = 12 * x^4 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l251_25137


namespace NUMINAMATH_CALUDE_paving_cost_l251_25176

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 400) :
  length * width * rate = 8250 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l251_25176


namespace NUMINAMATH_CALUDE_book_e_chapters_l251_25150

/-- Represents the number of chapters in each book --/
structure BookChapters where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The problem statement --/
def book_chapters_problem (total : ℕ) (books : BookChapters) : Prop :=
  books.a = 17 ∧
  books.b = books.a + 5 ∧
  books.c = books.b - 7 ∧
  books.d = 2 * books.c ∧
  total = 97 ∧
  total = books.a + books.b + books.c + books.d + books.e

/-- The theorem to prove --/
theorem book_e_chapters (total : ℕ) (books : BookChapters) 
  (h : book_chapters_problem total books) : books.e = 13 := by
  sorry


end NUMINAMATH_CALUDE_book_e_chapters_l251_25150


namespace NUMINAMATH_CALUDE_computer_operations_per_hour_l251_25162

/-- Represents the number of operations a computer can perform per second -/
structure ComputerSpeed :=
  (multiplications_per_second : ℕ)
  (additions_per_second : ℕ)

/-- Calculates the total number of operations per hour -/
def operations_per_hour (speed : ComputerSpeed) : ℕ :=
  (speed.multiplications_per_second + speed.additions_per_second) * 3600

/-- Theorem: A computer with the given speed performs 72 million operations per hour -/
theorem computer_operations_per_hour :
  let speed := ComputerSpeed.mk 15000 5000
  operations_per_hour speed = 72000000 := by
  sorry

end NUMINAMATH_CALUDE_computer_operations_per_hour_l251_25162


namespace NUMINAMATH_CALUDE_cone_volume_from_sector_l251_25158

/-- Given a cone whose lateral surface area is a sector with central angle 120° and area 3π,
    prove that the volume of the cone is (2√2π)/3 -/
theorem cone_volume_from_sector (θ : Real) (A : Real) (V : Real) : 
  θ = 2 * π / 3 →  -- 120° in radians
  A = 3 * π →
  V = (2 * Real.sqrt 2 * π) / 3 →
  ∃ (r l h : Real),
    r > 0 ∧ l > 0 ∧ h > 0 ∧
    A = (1/2) * l^2 * θ ∧  -- Area of sector
    r = l * θ / (2 * π) ∧  -- Relation between radius and arc length
    h^2 = l^2 - r^2 ∧     -- Pythagorean theorem
    V = (1/3) * π * r^2 * h  -- Volume of cone
    := by sorry

end NUMINAMATH_CALUDE_cone_volume_from_sector_l251_25158


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l251_25131

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to arrange n crates with 3 possible orientations each -/
def totalArrangements (n : ℕ) : ℕ :=
  3^n

/-- Calculates the number of ways to arrange crates to reach a specific height -/
def validArrangements (n : ℕ) (target_height : ℕ) : ℕ :=
  sorry  -- Placeholder for the actual calculation

/-- The probability of achieving the target height -/
def probability (n : ℕ) (target_height : ℕ) : ℚ :=
  (validArrangements n target_height : ℚ) / (totalArrangements n : ℚ)

theorem crate_stacking_probability :
  let crate_dims : CrateDimensions := ⟨3, 5, 7⟩
  let num_crates : ℕ := 10
  let target_height : ℕ := 43
  probability num_crates target_height = 10 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_crate_stacking_probability_l251_25131


namespace NUMINAMATH_CALUDE_sphere_cylinder_volumes_l251_25116

/-- Given a sphere with surface area 144π cm² that fits exactly inside a cylinder
    with height equal to the sphere's diameter, prove that the volume of the sphere
    is 288π cm³ and the volume of the cylinder is 432π cm³. -/
theorem sphere_cylinder_volumes (r : ℝ) (h : 4 * Real.pi * r^2 = 144 * Real.pi) :
  (4/3 : ℝ) * Real.pi * r^3 = 288 * Real.pi ∧
  Real.pi * r^2 * (2*r) = 432 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volumes_l251_25116


namespace NUMINAMATH_CALUDE_investment_average_interest_rate_l251_25152

/-- Prove that given a total investment split between two interest rates with equal annual returns, the average rate of interest is as calculated. -/
theorem investment_average_interest_rate 
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (h1 : total_investment = 6000)
  (h2 : rate1 = 0.03)
  (h3 : rate2 = 0.07)
  (h4 : ∃ (x : ℝ), x * rate2 = (total_investment - x) * rate1) :
  (rate1 * (total_investment - (180 / 0.1)) + rate2 * (180 / 0.1)) / total_investment = 0.042 := by
  sorry

end NUMINAMATH_CALUDE_investment_average_interest_rate_l251_25152


namespace NUMINAMATH_CALUDE_claras_weight_l251_25125

theorem claras_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight + clara_weight = 240)
  (h2 : clara_weight - alice_weight = (2/3) * clara_weight) : 
  clara_weight = 180 := by
sorry

end NUMINAMATH_CALUDE_claras_weight_l251_25125


namespace NUMINAMATH_CALUDE_shoe_price_calculation_shoe_price_proof_l251_25188

theorem shoe_price_calculation (initial_price : ℝ) 
  (increase_percentage : ℝ) (discount_percentage : ℝ) : ℝ :=
  let price_after_increase := initial_price * (1 + increase_percentage)
  let final_price := price_after_increase * (1 - discount_percentage)
  final_price

theorem shoe_price_proof :
  shoe_price_calculation 50 0.2 0.15 = 51 := by
  sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_shoe_price_proof_l251_25188


namespace NUMINAMATH_CALUDE_shipment_weight_change_l251_25149

theorem shipment_weight_change (total_boxes : Nat) (initial_avg : ℝ) (light_weight heavy_weight : ℝ) (removed_boxes : Nat) (new_avg : ℝ) : 
  total_boxes = 30 →
  light_weight = 10 →
  heavy_weight = 20 →
  initial_avg = 18 →
  removed_boxes = 18 →
  new_avg = 15 →
  ∃ (light_count heavy_count : Nat),
    light_count + heavy_count = total_boxes ∧
    (light_count * light_weight + heavy_count * heavy_weight) / total_boxes = initial_avg ∧
    ((light_count * light_weight + (heavy_count - removed_boxes) * heavy_weight) / (total_boxes - removed_boxes) = new_avg) :=
by sorry

end NUMINAMATH_CALUDE_shipment_weight_change_l251_25149


namespace NUMINAMATH_CALUDE_rectangular_box_diagonal_l251_25108

/-- Proves that a rectangular box with given surface area and edge length sum has a specific longest diagonal --/
theorem rectangular_box_diagonal (x y z : ℝ) : 
  (2 * (x*y + y*z + z*x) = 150) →  -- Total surface area
  (4 * (x + y + z) = 60) →         -- Sum of all edge lengths
  Real.sqrt (x^2 + y^2 + z^2) = 5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonal_l251_25108


namespace NUMINAMATH_CALUDE_range_of_a_l251_25159

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x + a ≥ 0) → a ≥ -8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l251_25159


namespace NUMINAMATH_CALUDE_min_unique_points_is_eight_l251_25145

/-- A square with points marked on its sides -/
structure MarkedSquare where
  /-- The number of points marked on each side of the square -/
  pointsPerSide : ℕ
  /-- Condition: Each side has exactly 3 points -/
  threePointsPerSide : pointsPerSide = 3

/-- The minimum number of unique points marked on the square -/
def minUniquePoints (s : MarkedSquare) : ℕ :=
  s.pointsPerSide * 4 - 4

/-- Theorem: The minimum number of unique points marked on the square is 8 -/
theorem min_unique_points_is_eight (s : MarkedSquare) : 
  minUniquePoints s = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_unique_points_is_eight_l251_25145


namespace NUMINAMATH_CALUDE_first_discount_percentage_l251_25187

/-- Proves that the first discount percentage is 30% given the initial and final prices --/
theorem first_discount_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 400)
  (h2 : final_price = 224) : 
  ∃ (x : ℝ), 
    (initial_price * (1 - x / 100) * (1 - 0.20) = final_price) ∧ 
    (x = 30) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l251_25187


namespace NUMINAMATH_CALUDE_total_chocolate_pieces_l251_25146

theorem total_chocolate_pieces (num_boxes : ℕ) (pieces_per_box : ℕ) 
  (h1 : num_boxes = 6) 
  (h2 : pieces_per_box = 500) : 
  num_boxes * pieces_per_box = 3000 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolate_pieces_l251_25146


namespace NUMINAMATH_CALUDE_two_distinct_real_roots_l251_25122

variable (a : ℝ)
variable (x : ℝ)

def f (a x : ℝ) : ℝ := (a+1)*(x^2+1)^2-(2*a+3)*(x^2+1)*x+(a+2)*x^2

theorem two_distinct_real_roots :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧
    ∀ x₃ : ℝ, f a x₃ = 0 → x₃ = x₁ ∨ x₃ = x₂) ↔ a ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_real_roots_l251_25122
