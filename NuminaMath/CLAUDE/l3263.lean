import Mathlib

namespace marble_probability_l3263_326308

theorem marble_probability (total_marbles : ℕ) 
  (prob_both_black : ℚ) (prob_both_white : ℚ) :
  total_marbles = 30 →
  prob_both_black = 4/9 →
  prob_both_white = 4/25 :=
by sorry

end marble_probability_l3263_326308


namespace geometric_sequence_fifth_term_l3263_326346

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geo : geometric_sequence a)
  (h_fourth : a 4 = (a 2) ^ 2)
  (h_sum : a 2 + a 4 = 5 / 16) :
  a 5 = 1 / 32 := by
sorry

end geometric_sequence_fifth_term_l3263_326346


namespace quadratic_equation_at_most_one_solution_l3263_326385

theorem quadratic_equation_at_most_one_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a ≥ 9/8 ∨ a = 0) :=
by sorry

end quadratic_equation_at_most_one_solution_l3263_326385


namespace only_seven_has_integer_solution_solutions_for_seven_l3263_326318

/-- The product of terms (1 + 1/(x+k)) from k = 0 to n -/
def productTerm (x : ℤ) (n : ℕ) : ℚ :=
  (List.range (n + 1)).foldl (fun acc k => acc * (1 + 1 / (x + k))) 1

/-- The main theorem stating that 7 is the only positive integer solution -/
theorem only_seven_has_integer_solution :
  ∀ a : ℕ+, (∃ x : ℤ, productTerm x a = a - x) ↔ a = 7 := by
  sorry

/-- Verification of the two integer solutions for a = 7 -/
theorem solutions_for_seven :
  productTerm 2 7 = 5 ∧ productTerm 4 7 = 3 := by
  sorry

end only_seven_has_integer_solution_solutions_for_seven_l3263_326318


namespace sequence_properties_l3263_326305

def b (n : ℕ) : ℝ := 2 * n - 1

def c (n : ℕ) : ℝ := 3 * n - 2

def a (n : ℕ) (x y : ℝ) : ℝ := x * b n + y * c n

theorem sequence_properties
  (x y : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : x + y = 1) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) x y - a n x y = d) ∧
  (∃ y' : ℝ, ∀ n : ℕ, a n x y' = (b n + c n) / 2) ∧
  (∀ n : ℕ, n ≥ 2 → b n < a n x y ∧ a n x y < c n) ∧
  (∀ n : ℕ, n ≥ 2 → a n x y + b n > c n ∧ a n x y + c n > b n ∧ b n + c n > a n x y) :=
by sorry

end sequence_properties_l3263_326305


namespace train_distance_problem_l3263_326383

theorem train_distance_problem (speed1 speed2 distance_diff : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 25)
  (h3 : distance_diff = 55)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0) :
  ∃ (time distance1 distance2 : ℝ),
    time > 0 ∧
    distance1 = speed1 * time ∧
    distance2 = speed2 * time ∧
    distance2 = distance1 + distance_diff ∧
    distance1 + distance2 = 495 := by
  sorry

end train_distance_problem_l3263_326383


namespace least_marbles_divisible_l3263_326358

theorem least_marbles_divisible (n : ℕ) : n > 0 ∧ 
  (∀ k ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ), n % k = 0) →
  n ≥ 420 :=
by sorry

end least_marbles_divisible_l3263_326358


namespace log_of_expression_l3263_326312

theorem log_of_expression (x : ℝ) : 
  x = 125 * Real.rpow 25 (1/3) * Real.sqrt 25 → 
  Real.log x / Real.log 5 = 14/3 := by
sorry

end log_of_expression_l3263_326312


namespace cube_root_equation_solution_l3263_326331

theorem cube_root_equation_solution :
  ∃ x : ℝ, x = 1/11 ∧ (5 + 2/x)^(1/3) = 3 := by
  sorry

end cube_root_equation_solution_l3263_326331


namespace monogramming_cost_is_17_69_l3263_326309

/-- Calculates the monogramming cost per stocking --/
def monogramming_cost_per_stocking (grandchildren children : ℕ) 
  (stockings_per_grandchild : ℕ) (stocking_price : ℚ) (discount_percent : ℚ) 
  (total_cost : ℚ) : ℚ :=
  let total_stockings := grandchildren * stockings_per_grandchild + children
  let discounted_price := stocking_price * (1 - discount_percent / 100)
  let stockings_cost := total_stockings * discounted_price
  let total_monogramming_cost := total_cost - stockings_cost
  total_monogramming_cost / total_stockings

/-- Theorem stating that the monogramming cost per stocking is $17.69 --/
theorem monogramming_cost_is_17_69 :
  monogramming_cost_per_stocking 5 4 5 20 10 1035 = 1769 / 100 := by
  sorry

end monogramming_cost_is_17_69_l3263_326309


namespace shooting_scenarios_correct_l3263_326352

/-- Represents a shooting scenario with a total number of shots and hits -/
structure ShootingScenario where
  totalShots : Nat
  totalHits : Nat

/-- Calculates the number of possible situations for Scenario 1 -/
def scenario1Situations (s : ShootingScenario) : Nat :=
  if s.totalShots = 10 ∧ s.totalHits = 7 then
    12
  else
    0

/-- Calculates the number of possible situations for Scenario 2 -/
def scenario2Situations (s : ShootingScenario) : Nat :=
  if s.totalShots = 10 then
    144
  else
    0

/-- Calculates the number of possible situations for Scenario 3 -/
def scenario3Situations (s : ShootingScenario) : Nat :=
  if s.totalShots = 10 ∧ s.totalHits = 6 then
    50
  else
    0

theorem shooting_scenarios_correct :
  ∀ s : ShootingScenario,
    (scenario1Situations s = 12 ∨ scenario1Situations s = 0) ∧
    (scenario2Situations s = 144 ∨ scenario2Situations s = 0) ∧
    (scenario3Situations s = 50 ∨ scenario3Situations s = 0) :=
by sorry

end shooting_scenarios_correct_l3263_326352


namespace consecutive_integers_divisible_by_three_l3263_326375

theorem consecutive_integers_divisible_by_three (a b c d e : ℕ) : 
  (70 < a) ∧ (a < 100) ∧
  (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1) ∧ (e = d + 1) ∧
  (a % 3 = 0) ∧ (b % 3 = 0) ∧ (c % 3 = 0) ∧ (d % 3 = 0) ∧ (e % 3 = 0) →
  e = 84 := by
sorry

end consecutive_integers_divisible_by_three_l3263_326375


namespace infinite_series_solution_l3263_326374

theorem infinite_series_solution : ∃! x : ℝ, x = (1 : ℝ) / (1 + x) ∧ |x| < 1 := by
  sorry

end infinite_series_solution_l3263_326374


namespace alcohol_mixture_concentration_l3263_326324

-- Define the concentrations and volumes
def x_concentration : ℝ := 0.10
def y_concentration : ℝ := 0.30
def target_concentration : ℝ := 0.22
def x_volume : ℝ := 300
def y_volume : ℝ := 450

-- Theorem statement
theorem alcohol_mixture_concentration :
  (x_concentration * x_volume + y_concentration * y_volume) / (x_volume + y_volume) = target_concentration := by
  sorry

end alcohol_mixture_concentration_l3263_326324


namespace min_moves_to_equalize_l3263_326342

/-- Represents a stack of coins -/
structure CoinStack :=
  (coins : ℕ)

/-- Represents the state of all coin stacks -/
structure CoinStacks :=
  (stacks : Fin 4 → CoinStack)

/-- Represents a move that adds one coin to three different stacks -/
structure Move :=
  (targets : Fin 3 → Fin 4)
  (different : targets 0 ≠ targets 1 ∧ targets 0 ≠ targets 2 ∧ targets 1 ≠ targets 2)

/-- The initial state of the coin stacks -/
def initial_stacks : CoinStacks :=
  CoinStacks.mk (fun i => match i with
    | 0 => CoinStack.mk 9
    | 1 => CoinStack.mk 7
    | 2 => CoinStack.mk 5
    | 3 => CoinStack.mk 10)

/-- Applies a move to a given state of coin stacks -/
def apply_move (stacks : CoinStacks) (move : Move) : CoinStacks :=
  sorry

/-- Checks if all stacks have an equal number of coins -/
def all_equal (stacks : CoinStacks) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem min_moves_to_equalize :
  ∃ (moves : List Move),
    moves.length = 11 ∧
    all_equal (moves.foldl apply_move initial_stacks) ∧
    ∀ (other_moves : List Move),
      all_equal (other_moves.foldl apply_move initial_stacks) →
      other_moves.length ≥ 11 :=
  sorry

end min_moves_to_equalize_l3263_326342


namespace reciprocal_sum_theorem_l3263_326361

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 8 * x * y) : 1 / x + 1 / y = 8 := by
  sorry

end reciprocal_sum_theorem_l3263_326361


namespace function_minimum_condition_l3263_326337

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem function_minimum_condition (a : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, f a x₀ ≤ f a x) ∧ 
  (∀ x : ℝ, f a x ≥ 2 * a^2 - a - 1) ↔ 
  0 < a ∧ a ≤ 1 :=
sorry

end function_minimum_condition_l3263_326337


namespace second_discarded_number_l3263_326354

theorem second_discarded_number 
  (n₁ : ℕ) (a₁ : ℚ) (n₂ : ℕ) (a₂ : ℚ) (x₁ : ℚ) :
  n₁ = 50 →
  a₁ = 38 →
  n₂ = 48 →
  a₂ = 37.5 →
  x₁ = 45 →
  ∃ x₂ : ℚ, x₂ = 55 ∧ n₁ * a₁ = n₂ * a₂ + x₁ + x₂ :=
by sorry

end second_discarded_number_l3263_326354


namespace regular_polygon_is_pentagon_with_perimeter_125_l3263_326313

/-- A regular polygon where the length of a side is 25 when the perimeter is divided by 5 -/
structure RegularPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  h1 : perimeter = sides * side_length
  h2 : perimeter / 5 = side_length
  h3 : side_length = 25

theorem regular_polygon_is_pentagon_with_perimeter_125 (p : RegularPolygon) :
  p.sides = 5 ∧ p.perimeter = 125 := by
  sorry

#check regular_polygon_is_pentagon_with_perimeter_125

end regular_polygon_is_pentagon_with_perimeter_125_l3263_326313


namespace gift_packaging_combinations_l3263_326320

/-- The number of varieties of packaging paper -/
def paper_varieties : ℕ := 10

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 4

/-- The number of types of decorative stickers -/
def sticker_types : ℕ := 5

/-- The total number of gift packaging combinations -/
def total_combinations : ℕ := paper_varieties * ribbon_colors * sticker_types

theorem gift_packaging_combinations :
  total_combinations = 200 := by sorry

end gift_packaging_combinations_l3263_326320


namespace p_twelve_equals_neg_five_l3263_326338

/-- A quadratic function with specific properties -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

/-- Theorem stating that p(12) = -5 given certain conditions -/
theorem p_twelve_equals_neg_five 
  (d e f : ℝ) 
  (h1 : ∀ x, p d e f (3.5 + x) = p d e f (3.5 - x)) -- axis of symmetry at x = 3.5
  (h2 : p d e f (-5) = -5) -- p(-5) = -5
  : p d e f 12 = -5 := by
  sorry

end p_twelve_equals_neg_five_l3263_326338


namespace consecutive_integers_average_l3263_326384

theorem consecutive_integers_average (x : ℝ) : 
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 25 →
  ((x - 9) + (x + 1 - 8) + (x + 2 - 7) + (x + 3 - 6) + (x + 4 - 5) + (x + 5 - 4) + (x + 6 - 3) + (x + 7 - 2) + (x + 8 - 1) + (x + 9)) / 10 = 20.5 := by
  sorry

end consecutive_integers_average_l3263_326384


namespace missing_number_solution_l3263_326387

theorem missing_number_solution : 
  ∃ x : ℝ, 0.72 * 0.43 + 0.12 * x = 0.3504 ∧ x = 0.34 := by
  sorry

end missing_number_solution_l3263_326387


namespace smallest_common_multiple_of_8_and_6_l3263_326336

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, 8 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 8 ∣ n ∧ 6 ∣ n :=
by
  -- The proof goes here
  sorry

end smallest_common_multiple_of_8_and_6_l3263_326336


namespace joshua_share_l3263_326347

/-- Given that Joshua and Justin shared $40 and Joshua's share was thrice as much as Justin's,
    prove that Joshua's share is $30. -/
theorem joshua_share (total : ℕ) (justin : ℕ) (joshua : ℕ) : 
  total = 40 → joshua = 3 * justin → total = joshua + justin → joshua = 30 := by
  sorry

end joshua_share_l3263_326347


namespace max_value_abc_l3263_326357

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_abc : a + b + c = 3) :
  a^2 * b^3 * c^4 ≤ 2048/19683 ∧ ∃ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 ∧ a^2 * b^3 * c^4 = 2048/19683 :=
by sorry

end max_value_abc_l3263_326357


namespace volunteer_selection_theorem_l3263_326319

/-- The number of ways to select three volunteers from five for three specific roles -/
def select_volunteers (n : ℕ) (k : ℕ) (excluded : ℕ) : ℕ :=
  (n - 1) * (n - 1) * (n - 2)

/-- The theorem stating that selecting three volunteers from five for three specific roles,
    where one volunteer cannot serve in a particular role, results in 48 different ways -/
theorem volunteer_selection_theorem :
  select_volunteers 5 3 1 = 48 := by
  sorry

end volunteer_selection_theorem_l3263_326319


namespace f_properties_l3263_326341

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_properties (a : ℝ) :
  (∀ x, a ≤ 0 → deriv (f a) x ≤ 0) ∧
  (a > 0 → ∀ x, x < Real.log (1/a) → deriv (f a) x < 0) ∧
  (a > 0 → ∀ x, x > Real.log (1/a) → deriv (f a) x > 0) ∧
  (a > 0 → ∀ x, f a x > 2 * Real.log a + 3/2) :=
by sorry

end f_properties_l3263_326341


namespace sufficient_but_not_necessary_l3263_326380

-- Define the proposition
def P (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0

-- Define the sufficient condition
def sufficient_condition (a : ℝ) : Prop := a ≥ 5

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, sufficient_condition a → P a) ∧
  ¬(∀ a : ℝ, P a → sufficient_condition a) :=
sorry

end sufficient_but_not_necessary_l3263_326380


namespace bridge_length_calculation_l3263_326316

/-- Proves that given a train of length 360 meters traveling at 36 km/hour,
    if it takes 50 seconds to pass a bridge, then the length of the bridge is 140 meters. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  train_length = 360 →
  train_speed_kmh = 36 →
  time_to_pass = 50 →
  (train_speed_kmh * 1000 / 3600) * time_to_pass - train_length = 140 :=
by sorry

end bridge_length_calculation_l3263_326316


namespace polynomial_equality_l3263_326363

theorem polynomial_equality (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, a + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 = x^3) →
  (a = 1 ∧ a₂ = 3) := by
  sorry

end polynomial_equality_l3263_326363


namespace sum_m_n_equals_five_l3263_326351

theorem sum_m_n_equals_five (m n : ℕ) (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : a = n * b) (h4 : a + b = m * (a - b)) : 
  m + n = 5 :=
sorry

end sum_m_n_equals_five_l3263_326351


namespace polynomial_factorization_l3263_326349

theorem polynomial_factorization (a b : ℝ) :
  2 * a^3 - 3 * a^2 * b - 3 * a * b^2 + 2 * b^3 = (a + b) * (a - 2*b) * (2*a - b) := by
  sorry

end polynomial_factorization_l3263_326349


namespace game_value_proof_l3263_326364

def super_nintendo_value : ℝ := 150
def store_credit_percentage : ℝ := 0.8
def tom_payment : ℝ := 80
def tom_change : ℝ := 10
def nes_sale_price : ℝ := 160

theorem game_value_proof :
  let credit := super_nintendo_value * store_credit_percentage
  let tom_actual_payment := tom_payment - tom_change
  let credit_used := nes_sale_price - tom_actual_payment
  credit - credit_used = 30 := by sorry

end game_value_proof_l3263_326364


namespace geometric_sum_proof_l3263_326321

/-- The sum of a geometric sequence with first term 9, common ratio 3, and 7 terms -/
def geometric_sum : ℕ := 9827

/-- The first term of the geometric sequence -/
def a : ℕ := 9

/-- The common ratio of the geometric sequence -/
def r : ℕ := 3

/-- The number of terms in the geometric sequence -/
def n : ℕ := 7

/-- Theorem stating that the sum of the geometric sequence equals 9827 -/
theorem geometric_sum_proof : 
  a * (r^n - 1) / (r - 1) = geometric_sum :=
sorry

end geometric_sum_proof_l3263_326321


namespace interest_rate_problem_l3263_326377

/-- 
Given a principal amount P and an interest rate R,
if increasing the rate by 1% for 3 years results in Rs. 63 more interest,
then P = Rs. 2100.
-/
theorem interest_rate_problem (P R : ℚ) : 
  (P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 63) → P = 2100 := by
  sorry

end interest_rate_problem_l3263_326377


namespace evaluate_expression_l3263_326326

theorem evaluate_expression : -(14 / 2 * 9 - 60 + 3 * 9) = -30 := by
  sorry

end evaluate_expression_l3263_326326


namespace solution_set_f_leq_5_range_of_m_for_nonempty_solution_l3263_326378

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -7/4 ≤ x ∧ x ≤ 3/4} := by sorry

-- Theorem for the range of m when the solution set of f(x) < |m-1| is non-empty
theorem range_of_m_for_nonempty_solution (m : ℝ) :
  (∃ x : ℝ, f x < |m - 1|) → (m > 5 ∨ m < -3) := by sorry

end solution_set_f_leq_5_range_of_m_for_nonempty_solution_l3263_326378


namespace cylinder_volume_l3263_326397

theorem cylinder_volume (r : ℝ) (h : ℝ) : 
  r > 0 → h > 0 → h = 2 * r → 2 * π * r * h = π → π * r^2 * h = π / 4 := by
  sorry

end cylinder_volume_l3263_326397


namespace average_salary_calculation_l3263_326330

theorem average_salary_calculation (officer_salary : ℕ) (non_officer_salary : ℕ)
  (num_officers : ℕ) (num_non_officers : ℕ) :
  officer_salary = 430 →
  non_officer_salary = 110 →
  num_officers = 15 →
  num_non_officers = 465 →
  (officer_salary * num_officers + non_officer_salary * num_non_officers) / (num_officers + num_non_officers) = 120 := by
  sorry

#eval (430 * 15 + 110 * 465) / (15 + 465)

end average_salary_calculation_l3263_326330


namespace carbonated_water_percent_in_specific_mixture_l3263_326359

/-- Represents a solution with a given percentage of carbonated water -/
structure Solution :=
  (carbonated_water_percent : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (solution1_volume_percent : ℝ)

/-- Calculates the percentage of carbonated water in a mixture -/
def carbonated_water_percent_in_mixture (m : Mixture) : ℝ :=
  m.solution1.carbonated_water_percent * m.solution1_volume_percent +
  m.solution2.carbonated_water_percent * (1 - m.solution1_volume_percent)

/-- Theorem stating that the percentage of carbonated water in the specific mixture is 67.5% -/
theorem carbonated_water_percent_in_specific_mixture :
  let solution1 : Solution := ⟨0.8⟩
  let solution2 : Solution := ⟨0.55⟩
  let mixture : Mixture := ⟨solution1, solution2, 0.5⟩
  carbonated_water_percent_in_mixture mixture = 0.675 := by
  sorry

end carbonated_water_percent_in_specific_mixture_l3263_326359


namespace triangle_angle_measure_l3263_326301

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧ 
  b / Real.sin B = c / Real.sin C ∧
  Real.sin B ^ 2 + Real.sin C ^ 2 = Real.sin A ^ 2 - Real.sqrt 3 * Real.sin B * Real.sin C →
  Real.cos A = -Real.sqrt 3 / 2 := by
sorry

end triangle_angle_measure_l3263_326301


namespace three_planes_max_regions_l3263_326360

/-- The maximum number of regions into which n planes can divide 3D space -/
def maxRegions (n : ℕ) : ℕ := sorry

/-- Theorem: Three planes can divide 3D space into at most 8 regions -/
theorem three_planes_max_regions :
  maxRegions 3 = 8 := by sorry

end three_planes_max_regions_l3263_326360


namespace sufficient_not_necessary_l3263_326303

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x^2 + y^2 = 0 → x * y = 0) ∧
  (∃ x y, x * y = 0 ∧ x^2 + y^2 ≠ 0) := by
  sorry

end sufficient_not_necessary_l3263_326303


namespace cranberry_harvest_percentage_l3263_326386

/-- Given the initial number of cranberries, the number eaten by elk, and the number left,
    prove that the percentage of cranberries harvested by humans is 40%. -/
theorem cranberry_harvest_percentage
  (total : ℕ)
  (eaten_by_elk : ℕ)
  (left : ℕ)
  (h1 : total = 60000)
  (h2 : eaten_by_elk = 20000)
  (h3 : left = 16000) :
  (total - eaten_by_elk - left) / total * 100 = 40 := by
  sorry

end cranberry_harvest_percentage_l3263_326386


namespace simplify_and_evaluate_solve_equation_l3263_326373

-- Problem 1
theorem simplify_and_evaluate : 
  let f (x : ℝ) := (x^2 - 6*x + 9) / (x^2 - 1) / ((x^2 - 3*x) / (x + 1))
  f (-3) = -1/2 := by sorry

-- Problem 2
theorem solve_equation :
  ∃ (x : ℝ), x / (x + 1) = 2*x / (3*x + 3) - 1 ∧ x = -3/4 := by sorry

end simplify_and_evaluate_solve_equation_l3263_326373


namespace rationalize_denominator_l3263_326329

theorem rationalize_denominator : 
  ∃ (a b : ℝ) (h : b ≠ 0), (7 / (2 * Real.sqrt 98)) = a / b ∧ b * Real.sqrt b = b := by
  sorry

end rationalize_denominator_l3263_326329


namespace smallest_even_three_digit_multiple_of_17_l3263_326323

theorem smallest_even_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 
    n % 17 = 0 ∧ 
    n % 2 = 0 ∧ 
    100 ≤ n ∧ n ≤ 999 → 
    n ≥ 136 :=
by sorry

end smallest_even_three_digit_multiple_of_17_l3263_326323


namespace present_giving_property_l3263_326333

/-- Represents a child in the class -/
structure Child where
  id : Nat

/-- Represents a triple of children -/
structure Triple where
  a : Child
  b : Child
  c : Child

/-- The main theorem to be proved -/
theorem present_giving_property (n : Nat) (h : Odd n) :
  ∃ (children : Finset Child) (S : Finset Triple),
    (children.card = 3 * n) ∧
    (∀ (x y : Child), x ∈ children → y ∈ children → x ≠ y →
      ∃! (t : Triple), t ∈ S ∧ (t.a = x ∧ t.b = y ∨ t.a = x ∧ t.c = y ∨ t.b = x ∧ t.c = y)) ∧
    (∀ (t : Triple), t ∈ S →
      ∃ (t' : Triple), t' ∈ S ∧ t'.a = t.a ∧ t'.b = t.c ∧ t'.c = t.b) := by
  sorry

end present_giving_property_l3263_326333


namespace sqrt_floor_equality_l3263_326370

theorem sqrt_floor_equality (n : ℕ) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end sqrt_floor_equality_l3263_326370


namespace animal_path_distance_l3263_326393

theorem animal_path_distance : 
  let outer_radius : ℝ := 25
  let middle_radius : ℝ := 15
  let inner_radius : ℝ := 5
  let outer_arc : ℝ := (1/4) * 2 * Real.pi * outer_radius
  let middle_to_outer : ℝ := outer_radius - middle_radius
  let middle_arc : ℝ := (1/4) * 2 * Real.pi * middle_radius
  let to_center_and_back : ℝ := 2 * middle_radius
  let middle_to_inner : ℝ := middle_radius - inner_radius
  outer_arc + middle_to_outer + middle_arc + to_center_and_back + middle_arc + middle_to_inner = 27.5 * Real.pi + 50 := by
  sorry

end animal_path_distance_l3263_326393


namespace parabolas_equal_if_equal_segments_l3263_326304

/-- Two non-parallel lines in the plane -/
structure NonParallelLines where
  l₁ : ℝ → ℝ
  l₂ : ℝ → ℝ
  not_parallel : l₁ ≠ l₂

/-- A parabola of the form f(x) = x² + px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- The length of the segment cut by a parabola on a line -/
def segment_length (para : Parabola) (line : ℝ → ℝ) : ℝ := sorry

/-- Two parabolas cut equal segments on two non-parallel lines -/
def equal_segments (f₁ f₂ : Parabola) (lines : NonParallelLines) : Prop :=
  segment_length f₁ lines.l₁ = segment_length f₂ lines.l₁ ∧
  segment_length f₁ lines.l₂ = segment_length f₂ lines.l₂

/-- Main theorem: If two parabolas cut equal segments on two non-parallel lines, 
    then the parabolas are identical -/
theorem parabolas_equal_if_equal_segments (f₁ f₂ : Parabola) (lines : NonParallelLines) :
  equal_segments f₁ f₂ lines → f₁ = f₂ := by sorry

end parabolas_equal_if_equal_segments_l3263_326304


namespace proper_subsets_of_B_l3263_326317

-- Define the sets A and B
def A (b : ℝ) : Set ℝ := {x | x^2 + (b+2)*x + b + 1 = 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

-- State the theorem
theorem proper_subsets_of_B (b : ℝ) (a : ℝ) (h : A b = {a}) :
  {s : Set ℝ | s ⊂ B a b ∧ s ≠ B a b} = {∅, {1}, {0}} := by
  sorry

end proper_subsets_of_B_l3263_326317


namespace curve_self_intersection_l3263_326376

/-- A point on the curve defined by t --/
def curve_point (t : ℝ) : ℝ × ℝ :=
  (t^2 - 4, t^3 - 6*t + 7)

/-- The curve crosses itself if there exist two distinct real numbers that map to the same point --/
def self_intersection (p : ℝ × ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ curve_point a = p ∧ curve_point b = p

theorem curve_self_intersection :
  self_intersection (2, 7) := by
  sorry

end curve_self_intersection_l3263_326376


namespace total_volume_of_cubes_total_volume_is_114_l3263_326348

theorem total_volume_of_cubes : ℕ → ℕ → ℕ → ℕ → ℕ
  | carl_cube_count, carl_cube_side, kate_cube_count, kate_cube_side =>
    let carl_cube_volume := carl_cube_side ^ 3
    let kate_cube_volume := kate_cube_side ^ 3
    carl_cube_count * carl_cube_volume + kate_cube_count * kate_cube_volume

theorem total_volume_is_114 : 
  total_volume_of_cubes 4 3 6 1 = 114 := by
  sorry

end total_volume_of_cubes_total_volume_is_114_l3263_326348


namespace notebook_pencil_cost_l3263_326356

/-- Given the prices of notebooks and pencils in two scenarios, prove the cost of one notebook and one pencil. -/
theorem notebook_pencil_cost
  (scenario1 : 6 * notebook_price + 4 * pencil_price = 9.2)
  (scenario2 : 3 * notebook_price + pencil_price = 3.8)
  : notebook_price + pencil_price = 1.8 := by
  sorry

end notebook_pencil_cost_l3263_326356


namespace tetrahedron_division_l3263_326396

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- A plane passing through one edge and the midpoint of the opposite edge of a tetrahedron -/
structure DividingPlane where
  tetrahedron : RegularTetrahedron

/-- The parts into which a tetrahedron is divided by the planes -/
structure TetrahedronPart where
  tetrahedron : RegularTetrahedron
  planes : Finset DividingPlane

/-- The theorem stating the division of a regular tetrahedron by six specific planes -/
theorem tetrahedron_division (t : RegularTetrahedron) 
  (h_volume : t.volume = 1) 
  (planes : Finset DividingPlane) 
  (h_planes : planes.card = 6) 
  (h_plane_position : ∀ p ∈ planes, p.tetrahedron = t) : 
  ∃ (parts : Finset TetrahedronPart), 
    (parts.card = 24) ∧ 
    (∀ part ∈ parts, part.tetrahedron = t ∧ part.planes = planes) ∧
    (∀ part ∈ parts, ∃ v : ℝ, v = 1 / 24) :=
sorry

end tetrahedron_division_l3263_326396


namespace sufficient_not_necessary_l3263_326392

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) := by
  sorry

end sufficient_not_necessary_l3263_326392


namespace initial_number_of_persons_l3263_326328

theorem initial_number_of_persons (n : ℕ) 
  (avg_weight_increase : ℝ) 
  (weight_difference : ℝ) : 
  avg_weight_increase = 2.5 →
  weight_difference = 20 →
  weight_difference = n * avg_weight_increase →
  n = 8 := by
  sorry

end initial_number_of_persons_l3263_326328


namespace cuboid_colored_cubes_theorem_l3263_326391

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of cubes colored on only one side when a cuboid is cut into unit cubes -/
def cubesColoredOnOneSide (c : Cuboid) : ℕ :=
  2 * ((c.width - 2) * (c.length - 2) + (c.width - 2) * (c.height - 2) + (c.length - 2) * (c.height - 2))

theorem cuboid_colored_cubes_theorem (c : Cuboid) 
    (h_width : c.width = 5)
    (h_length : c.length = 4)
    (h_height : c.height = 3) :
  cubesColoredOnOneSide c = 22 := by
  sorry

end cuboid_colored_cubes_theorem_l3263_326391


namespace sector_area_one_radian_unit_radius_l3263_326398

/-- The area of a circular sector with central angle 1 radian and radius 1 unit is 1/2 square units. -/
theorem sector_area_one_radian_unit_radius : 
  let θ : Real := 1  -- Central angle in radians
  let r : Real := 1  -- Radius
  let sector_area := (1/2) * r * r * θ
  sector_area = 1/2 := by sorry

end sector_area_one_radian_unit_radius_l3263_326398


namespace jacket_cost_calculation_l3263_326388

def total_spent : ℚ := 33.56
def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14

theorem jacket_cost_calculation : 
  total_spent - shorts_cost - shirt_cost = 7.43 := by sorry

end jacket_cost_calculation_l3263_326388


namespace sheridan_cats_bought_l3263_326345

/-- The number of cats Mrs. Sheridan bought -/
def cats_bought (initial final : ℝ) : ℝ := final - initial

/-- Theorem stating that Mrs. Sheridan bought 43 cats -/
theorem sheridan_cats_bought :
  cats_bought 11.0 54 = 43 := by
  sorry

end sheridan_cats_bought_l3263_326345


namespace number_comparisons_l3263_326300

theorem number_comparisons :
  (-7 / 8 : ℚ) > (-8 / 9 : ℚ) ∧ -|(-5)| < -(-4) := by sorry

end number_comparisons_l3263_326300


namespace apple_pie_problem_l3263_326353

/-- The number of apples needed per pie -/
def apples_per_pie (total_pies : ℕ) (apples_from_garden : ℕ) (apples_to_buy : ℕ) : ℕ :=
  (apples_from_garden + apples_to_buy) / total_pies

theorem apple_pie_problem :
  apples_per_pie 10 50 30 = 8 := by
  sorry

end apple_pie_problem_l3263_326353


namespace possible_m_values_l3263_326302

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem possible_m_values (m : ℝ) : A ∪ B m = A → m = 0 ∨ m = -1/3 ∨ m = 1 := by
  sorry

end possible_m_values_l3263_326302


namespace total_apples_l3263_326382

/-- Given 37 baskets with 17 apples each, prove that the total number of apples is 629. -/
theorem total_apples (num_baskets : ℕ) (apples_per_basket : ℕ) (h1 : num_baskets = 37) (h2 : apples_per_basket = 17) :
  num_baskets * apples_per_basket = 629 := by
  sorry

end total_apples_l3263_326382


namespace intercept_triangle_area_zero_l3263_326311

/-- The cubic function f(x) = x³ - x --/
def f (x : ℝ) : ℝ := x^3 - x

/-- The set of x-intercepts of the curve y = x³ - x --/
def x_intercepts : Set ℝ := {x : ℝ | f x = 0}

/-- The y-intercept of the curve y = x³ - x --/
def y_intercept : ℝ × ℝ := (0, f 0)

/-- The area of the triangle formed by the intercepts of the curve y = x³ - x --/
def triangle_area : ℝ := sorry

/-- Theorem: The area of the triangle formed by the intercepts of y = x³ - x is 0 --/
theorem intercept_triangle_area_zero : triangle_area = 0 := by sorry

end intercept_triangle_area_zero_l3263_326311


namespace divisible_by_eleven_l3263_326314

theorem divisible_by_eleven (m : ℕ) : 
  m < 10 →
  (864 * 10^7 + m * 10^6 + 5 * 10^5 + 3 * 10^4 + 7 * 10^3 + 9 * 10^2 + 7 * 10 + 9) % 11 = 0 →
  m = 9 := by
sorry

end divisible_by_eleven_l3263_326314


namespace goldfish_preference_l3263_326371

theorem goldfish_preference (total_students : ℕ) (johnson_fraction : ℚ) (henderson_fraction : ℚ) (total_preference : ℕ) :
  total_students = 30 →
  johnson_fraction = 1/6 →
  henderson_fraction = 1/5 →
  total_preference = 31 →
  ∃ feldstein_fraction : ℚ,
    feldstein_fraction = 2/3 ∧
    total_preference = johnson_fraction * total_students + henderson_fraction * total_students + feldstein_fraction * total_students :=
by
  sorry

end goldfish_preference_l3263_326371


namespace diorama_time_factor_l3263_326343

def total_time : ℕ := 67
def building_time : ℕ := 49

theorem diorama_time_factor :
  ∃ (planning_time : ℕ) (factor : ℚ),
    planning_time + building_time = total_time ∧
    building_time = planning_time * factor - 5 ∧
    factor = 3 := by
  sorry

end diorama_time_factor_l3263_326343


namespace quadratic_one_solution_l3263_326362

theorem quadratic_one_solution (m : ℚ) : 
  (∃! y, 3 * y^2 - 7 * y + m = 0) ↔ m = 49/12 := by
sorry

end quadratic_one_solution_l3263_326362


namespace cow_chicken_problem_l3263_326367

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 12) → cows = 6 := by
  sorry

end cow_chicken_problem_l3263_326367


namespace addition_subtraction_equality_l3263_326335

theorem addition_subtraction_equality : 147 + 31 - 19 + 21 = 180 := by sorry

end addition_subtraction_equality_l3263_326335


namespace proposition_p_false_l3263_326365

theorem proposition_p_false : 
  ¬(∀ x : ℝ, x > 0 → x^2 - 3*x + 12 < 0) ∧ 
  (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 12 ≥ 0) := by
  sorry

end proposition_p_false_l3263_326365


namespace rectangle_area_difference_l3263_326369

def is_valid_rectangle (l w : ℕ) : Prop :=
  2 * l + 2 * w = 56 ∧ (l ≥ w + 5 ∨ w ≥ l + 5)

def rectangle_area (l w : ℕ) : ℕ := l * w

theorem rectangle_area_difference : 
  ∃ (l₁ w₁ l₂ w₂ : ℕ),
    is_valid_rectangle l₁ w₁ ∧
    is_valid_rectangle l₂ w₂ ∧
    ∀ (l w : ℕ),
      is_valid_rectangle l w →
      rectangle_area l w ≤ rectangle_area l₁ w₁ ∧
      rectangle_area l w ≥ rectangle_area l₂ w₂ ∧
      rectangle_area l₁ w₁ - rectangle_area l₂ w₂ = 5 :=
sorry

end rectangle_area_difference_l3263_326369


namespace sum_of_fractions_l3263_326315

theorem sum_of_fractions : (3 / 30) + (4 / 40) + (5 / 50) = 0.3 := by
  sorry

end sum_of_fractions_l3263_326315


namespace circle_intersection_range_l3263_326327

theorem circle_intersection_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x - 2*Real.sqrt 3*y - m = 0 ∧ x^2 + y^2 = 1) ↔ 
  -3 ≤ m ∧ m ≤ 5 :=
by sorry

end circle_intersection_range_l3263_326327


namespace b2f_to_decimal_l3263_326366

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Converts a hexadecimal digit to its decimal value --/
def hexToDecimal (d : HexDigit) : ℕ :=
  match d with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Converts a list of hexadecimal digits to its decimal value --/
def hexListToDecimal (l : List HexDigit) : ℕ :=
  l.enum.foldr (fun (i, d) acc => acc + (hexToDecimal d) * (16 ^ i)) 0

theorem b2f_to_decimal :
  hexListToDecimal [HexDigit.B, HexDigit.D2, HexDigit.F] = 2863 := by
  sorry

end b2f_to_decimal_l3263_326366


namespace adult_meal_cost_l3263_326307

/-- Proves that the cost of each adult meal is $3 given the specified conditions -/
theorem adult_meal_cost (total_people : Nat) (kids : Nat) (total_cost : Nat) :
  total_people = 12 →
  kids = 7 →
  total_cost = 15 →
  (total_cost / (total_people - kids) : ℚ) = 3 := by
  sorry

end adult_meal_cost_l3263_326307


namespace line_slope_l3263_326395

theorem line_slope (x y : ℝ) :
  x + Real.sqrt 3 * y + 2 = 0 →
  (y - (-2 / Real.sqrt 3)) / (x - 0) = - Real.sqrt 3 / 3 :=
by sorry

end line_slope_l3263_326395


namespace emily_cell_phone_cost_l3263_326399

/-- Calculates the total cost of a cell phone plan based on given parameters. -/
def calculate_total_cost (base_cost : ℕ) (included_hours : ℕ) (extra_hour_cost : ℕ)
  (base_message_cost : ℕ) (base_message_limit : ℕ) (hours_used : ℕ) (messages_sent : ℕ) : ℕ :=
  let extra_hours := max (hours_used - included_hours) 0
  let extra_hour_charge := extra_hours * extra_hour_cost
  let base_message_charge := min messages_sent base_message_limit * base_message_cost
  let extra_messages := max (messages_sent - base_message_limit) 0
  let extra_message_charge := extra_messages * (2 * base_message_cost)
  base_cost + extra_hour_charge + base_message_charge + extra_message_charge

/-- Emily's cell phone plan cost theorem -/
theorem emily_cell_phone_cost :
  calculate_total_cost 30 50 15 10 150 52 200 = 8500 :=
by sorry

end emily_cell_phone_cost_l3263_326399


namespace original_price_calculation_l3263_326379

theorem original_price_calculation (final_price : ℝ) : 
  final_price = 1120 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.3) * (1 - 0.2) = final_price ∧ 
    original_price = 2000 := by
  sorry

end original_price_calculation_l3263_326379


namespace total_cups_sold_l3263_326350

def plastic_cups : ℕ := 284
def ceramic_cups : ℕ := 284

theorem total_cups_sold : plastic_cups + ceramic_cups = 568 := by
  sorry

end total_cups_sold_l3263_326350


namespace local_minimum_at_one_l3263_326306

/-- The function f(x) = ax³ - 2x² + a²x has a local minimum at x=1 if and only if a = 1 -/
theorem local_minimum_at_one (a : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), 
    a*x^3 - 2*x^2 + a^2*x ≥ a*1^3 - 2*1^2 + a^2*1) ↔ a = 1 := by
  sorry


end local_minimum_at_one_l3263_326306


namespace jose_wandering_time_l3263_326332

/-- Given a distance of 4 kilometers and a speed of 2 kilometers per hour,
    the time taken is 2 hours. -/
theorem jose_wandering_time :
  let distance : ℝ := 4  -- Distance in kilometers
  let speed : ℝ := 2     -- Speed in kilometers per hour
  let time := distance / speed
  time = 2 := by sorry

end jose_wandering_time_l3263_326332


namespace polynomial_roots_magnitude_l3263_326394

theorem polynomial_roots_magnitude (c : ℂ) : 
  let p : ℂ → ℂ := λ x => (x^2 - 2*x + 2) * (x^2 - c*x + 4) * (x^2 - 4*x + 8)
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ z ∈ s, p z = 0) ∧ (∀ z, p z = 0 → z ∈ s)) →
  Complex.abs c = Real.sqrt 10 := by
  sorry

end polynomial_roots_magnitude_l3263_326394


namespace berry_tuesday_temperature_l3263_326344

def berry_temperatures : List Float := [99.1, 98.2, 99.3, 99.8, 99, 98.9]
def average_temperature : Float := 99
def days_in_week : Nat := 7

theorem berry_tuesday_temperature :
  let total_sum : Float := average_temperature * days_in_week.toFloat
  let known_sum : Float := berry_temperatures.sum
  let tuesday_temp : Float := total_sum - known_sum
  tuesday_temp = 98.7 := by sorry

end berry_tuesday_temperature_l3263_326344


namespace quadratic_equation_prime_solutions_l3263_326390

theorem quadratic_equation_prime_solutions :
  ∀ (p q x₁ x₂ : ℤ),
    Prime p →
    Prime q →
    x₁^2 + p*x₁ + 3*q = 0 →
    x₂^2 + p*x₂ + 3*q = 0 →
    x₁ + x₂ = -p →
    x₁ * x₂ = 3*q →
    ((p = 7 ∧ q = 2 ∧ x₁ = -1 ∧ x₂ = -6) ∨
     (p = 5 ∧ q = 2 ∧ x₁ = -3 ∧ x₂ = -2)) :=
by sorry

end quadratic_equation_prime_solutions_l3263_326390


namespace skate_cost_theorem_l3263_326310

/-- The cost of a new pair of skates is equal to 26 times the rental fee. -/
theorem skate_cost_theorem (admission_fee : ℝ) (rental_fee : ℝ) (visits : ℕ) 
  (h1 : admission_fee = 5)
  (h2 : rental_fee = 2.5)
  (h3 : visits = 26) :
  visits * rental_fee = 65 := by
  sorry

#check skate_cost_theorem

end skate_cost_theorem_l3263_326310


namespace charlie_cleaning_time_l3263_326381

theorem charlie_cleaning_time (alice_time bob_time charlie_time : ℚ) : 
  alice_time = 30 →
  bob_time = (3 / 4) * alice_time →
  charlie_time = (1 / 3) * bob_time →
  charlie_time = 7.5 := by
  sorry

end charlie_cleaning_time_l3263_326381


namespace subtracted_number_l3263_326322

theorem subtracted_number (x : ℝ) (y : ℝ) : 
  x = 62.5 → ((x + 5) * 2 / 5 - y = 22) → y = 5 := by
  sorry

end subtracted_number_l3263_326322


namespace equation_holds_iff_specific_values_l3263_326355

theorem equation_holds_iff_specific_values :
  ∀ (a b p q : ℝ),
  (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
  ((b = (1/2) * (2^20 - 1)^(1/20) ∧ a = -(2^20 - 1)^(1/20)) ∨
   (b = -(1/2) * (2^20 - 1)^(1/20) ∧ a = (2^20 - 1)^(1/20))) ∧
  p = -1 ∧ q = 1/4 :=
by sorry

end equation_holds_iff_specific_values_l3263_326355


namespace coefficient_expansion_l3263_326372

theorem coefficient_expansion (m : ℝ) : 
  (∃ c : ℝ, c = -160 ∧ c = 20 * m^3) → m = -2 := by
  sorry

end coefficient_expansion_l3263_326372


namespace shade_in_three_folds_l3263_326325

/-- Represents a square grid -/
structure Grid :=
  (size : Nat)
  (shaded : Set (Nat × Nat))

/-- Represents a fold along a grid line -/
inductive Fold
  | Vertical (col : Nat)
  | Horizontal (row : Nat)

/-- Apply a fold to a grid -/
def applyFold (g : Grid) (f : Fold) : Grid :=
  sorry

/-- Check if the entire grid is shaded -/
def isFullyShaded (g : Grid) : Prop :=
  sorry

/-- Theorem stating that it's possible to shade the entire grid in 3 or fewer folds -/
theorem shade_in_three_folds (g : Grid) :
  ∃ (folds : List Fold), folds.length ≤ 3 ∧ isFullyShaded (folds.foldl applyFold g) :=
sorry

end shade_in_three_folds_l3263_326325


namespace smallest_sum_of_a_and_b_l3263_326368

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) :
  a + b ≥ 16 + (4/3) * Real.rpow 3 (1/3) :=
sorry

end smallest_sum_of_a_and_b_l3263_326368


namespace arithmetic_sequence_property_l3263_326389

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given condition: a_2 + a_5 + a_8 = 6 -/
def GivenCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 5 + a 8 = 6

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : GivenCondition a) : a 5 = 2 := by
  sorry

end arithmetic_sequence_property_l3263_326389


namespace square_side_length_l3263_326340

theorem square_side_length (perimeter : ℝ) (h : perimeter = 100) : 
  ∃ (side_length : ℝ), side_length = 25 ∧ 4 * side_length = perimeter := by
  sorry

end square_side_length_l3263_326340


namespace intersection_M_N_l3263_326334

def M : Set ℝ := {x | 0 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l3263_326334


namespace b_visited_zhougong_l3263_326339

-- Define the celebrities
inductive Celebrity
| A
| B
| C

-- Define the places
inductive Place
| ZhougongTemple
| FamenTemple
| Wuzhangyuan

-- Define a function to represent whether a celebrity visited a place
def visited : Celebrity → Place → Prop := sorry

-- A visited more places than B
axiom a_visited_more : ∃ (p : Place), visited Celebrity.A p ∧ ¬visited Celebrity.B p

-- A did not visit Famen Temple
axiom a_not_famen : ¬visited Celebrity.A Place.FamenTemple

-- B did not visit Wuzhangyuan
axiom b_not_wuzhangyuan : ¬visited Celebrity.B Place.Wuzhangyuan

-- The three celebrities visited the same place
axiom same_place : ∃ (p : Place), visited Celebrity.A p ∧ visited Celebrity.B p ∧ visited Celebrity.C p

-- Theorem to prove
theorem b_visited_zhougong : visited Celebrity.B Place.ZhougongTemple := by sorry

end b_visited_zhougong_l3263_326339
