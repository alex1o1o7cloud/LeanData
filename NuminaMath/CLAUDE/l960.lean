import Mathlib

namespace NUMINAMATH_CALUDE_selling_price_calculation_l960_96030

def cost_price : ℝ := 1500
def loss_percentage : ℝ := 14.000000000000002

theorem selling_price_calculation (cost_price : ℝ) (loss_percentage : ℝ) :
  let loss_amount := (loss_percentage / 100) * cost_price
  let selling_price := cost_price - loss_amount
  selling_price = 1290 := by sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l960_96030


namespace NUMINAMATH_CALUDE_greatest_x_given_lcm_l960_96090

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem greatest_x_given_lcm (x : ℕ) :
  lcm x 15 21 = 105 → x ≤ 105 ∧ ∃ y : ℕ, lcm y 15 21 = 105 ∧ y = 105 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_given_lcm_l960_96090


namespace NUMINAMATH_CALUDE_integral_inequality_l960_96037

theorem integral_inequality (n : ℕ) (hn : n ≥ 2) :
  (1 : ℝ) / n < ∫ x in (0 : ℝ)..(π / 2), 1 / (1 + Real.cos x) ^ n ∧
  ∫ x in (0 : ℝ)..(π / 2), 1 / (1 + Real.cos x) ^ n < (n + 5 : ℝ) / (n * (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_integral_inequality_l960_96037


namespace NUMINAMATH_CALUDE_expected_value_is_three_halves_l960_96082

/-- The number of white balls in the bag -/
def white_balls : ℕ := 1

/-- The number of red balls in the bag -/
def red_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 2

/-- X represents the number of red balls drawn -/
def X : Finset ℕ := {1, 2}

/-- The probability mass function of X -/
def prob_X (x : ℕ) : ℚ :=
  if x = 1 then 1/2
  else if x = 2 then 1/2
  else 0

/-- The expected value of X -/
def expected_value_X : ℚ := (1 : ℚ) * (prob_X 1) + (2 : ℚ) * (prob_X 2)

theorem expected_value_is_three_halves :
  expected_value_X = 3/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_three_halves_l960_96082


namespace NUMINAMATH_CALUDE_telephone_network_connections_l960_96062

/-- The number of distinct connections in a network of telephones -/
def distinct_connections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 7 telephones, where each telephone is connected to 6 others,
    the total number of distinct connections is 21. -/
theorem telephone_network_connections :
  distinct_connections 7 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_telephone_network_connections_l960_96062


namespace NUMINAMATH_CALUDE_distribution_five_to_three_l960_96069

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least min_per_group objects and
    at most max_per_group objects. -/
def distribution_count (n k min_per_group max_per_group : ℕ) : ℕ := sorry

/-- Theorem: There are 30 ways to distribute 5 distinct objects into 3 distinct groups,
    where each group must contain at least 1 object and at most 2 objects. -/
theorem distribution_five_to_three : distribution_count 5 3 1 2 = 30 := by sorry

end NUMINAMATH_CALUDE_distribution_five_to_three_l960_96069


namespace NUMINAMATH_CALUDE_line_quadrants_l960_96054

/-- A line passing through the second and fourth quadrants has a negative slope -/
def passes_through_second_and_fourth_quadrants (k : ℝ) : Prop :=
  k < 0

/-- A line y = kx + b passes through the first and third quadrants if k > 0 -/
def passes_through_first_and_third_quadrants (k : ℝ) : Prop :=
  k > 0

/-- A line y = kx + b passes through the fourth quadrant if k > 0 and b < 0 -/
def passes_through_fourth_quadrant (k b : ℝ) : Prop :=
  k > 0 ∧ b < 0

theorem line_quadrants (k : ℝ) :
  passes_through_second_and_fourth_quadrants k →
  passes_through_first_and_third_quadrants (-k) ∧
  passes_through_fourth_quadrant (-k) (-1) :=
by sorry

end NUMINAMATH_CALUDE_line_quadrants_l960_96054


namespace NUMINAMATH_CALUDE_function_square_evaluation_l960_96061

theorem function_square_evaluation (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2
  f (a + 1) = a^2 + 2*a + 1 := by sorry

end NUMINAMATH_CALUDE_function_square_evaluation_l960_96061


namespace NUMINAMATH_CALUDE_q_factor_change_l960_96048

theorem q_factor_change (w m z : ℝ) (hw : w ≠ 0) (hm : m ≠ 0) (hz : z ≠ 0) :
  let q := 5 * w / (4 * m * z^2)
  let q_new := 5 * (4*w) / (4 * (2*m) * (3*z)^2)
  q_new = (2/9) * q := by
sorry

end NUMINAMATH_CALUDE_q_factor_change_l960_96048


namespace NUMINAMATH_CALUDE_solution_value_l960_96007

theorem solution_value (a : ℝ) (h : a^2 - 5*a - 1 = 0) : 3*a^2 - 15*a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l960_96007


namespace NUMINAMATH_CALUDE_functional_equation_solution_l960_96041

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)

/-- The theorem stating that the only functions satisfying the equation are f(x) = 0 or f(x) = x² -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l960_96041


namespace NUMINAMATH_CALUDE_problem_solution_l960_96045

theorem problem_solution :
  45 / (8 - 3/7) = 315/53 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l960_96045


namespace NUMINAMATH_CALUDE_mod_equation_l960_96085

theorem mod_equation (m : ℕ) (h1 : m < 37) (h2 : (4 * m) % 37 = 1) :
  (3^m)^2 % 37 - 3 % 37 = 19 := by
  sorry

end NUMINAMATH_CALUDE_mod_equation_l960_96085


namespace NUMINAMATH_CALUDE_either_odd_or_even_l960_96070

theorem either_odd_or_even (n : ℤ) : (Odd (2*n - 1)) ∨ (Even (2*n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_either_odd_or_even_l960_96070


namespace NUMINAMATH_CALUDE_sum_of_digits_l960_96017

theorem sum_of_digits (A B C D E : ℕ) : A + B + C + D + E = 32 :=
  by
  have h1 : A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 := by sorry
  have h2 : 3 * E % 10 = 1 := by sorry
  have h3 : 3 * A + (B + C + D + 2) = 20 := by sorry
  have h4 : B + C + D = 19 := by sorry
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l960_96017


namespace NUMINAMATH_CALUDE_no_real_solution_exists_sum_equals_square_plus_twenty_l960_96024

/-- Given three numbers with a difference of 4 between each,
    where their sum is 20 more than the square of the first number,
    prove that no real solution exists for the middle number. -/
theorem no_real_solution_exists (x : ℝ) : ¬ ∃ x : ℝ, x^2 - 3*x + 8 = 0 := by
  sorry

/-- Define the relationship between the three numbers -/
def second_number (x : ℝ) : ℝ := x + 4

/-- Define the relationship between the three numbers -/
def third_number (x : ℝ) : ℝ := x + 8

/-- Define the sum of the three numbers -/
def sum_of_numbers (x : ℝ) : ℝ := x + second_number x + third_number x

/-- Define the relationship between the sum and the square of the first number -/
theorem sum_equals_square_plus_twenty (x : ℝ) : sum_of_numbers x = x^2 + 20 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_exists_sum_equals_square_plus_twenty_l960_96024


namespace NUMINAMATH_CALUDE_joan_balloons_l960_96064

theorem joan_balloons (total sally jessica : ℕ) (h1 : total = 16) (h2 : sally = 5) (h3 : jessica = 2) :
  ∃ joan : ℕ, joan + sally + jessica = total ∧ joan = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l960_96064


namespace NUMINAMATH_CALUDE_shooting_probabilities_l960_96081

/-- Represents the probabilities of hitting each ring in a shooting game. -/
structure RingProbabilities where
  ten : Real
  nine : Real
  eight : Real
  seven : Real
  sub_seven : Real

/-- The probabilities sum to 1. -/
axiom prob_sum_to_one (p : RingProbabilities) : 
  p.ten + p.nine + p.eight + p.seven + p.sub_seven = 1

/-- The given probabilities for each ring. -/
def given_probs : RingProbabilities := {
  ten := 0.24,
  nine := 0.28,
  eight := 0.19,
  seven := 0.16,
  sub_seven := 0.13
}

/-- The probability of hitting 10 or 9 rings. -/
def prob_ten_or_nine (p : RingProbabilities) : Real :=
  p.ten + p.nine

/-- The probability of hitting at least 7 ring. -/
def prob_at_least_seven (p : RingProbabilities) : Real :=
  p.ten + p.nine + p.eight + p.seven

/-- The probability of not hitting 8 ring. -/
def prob_not_eight (p : RingProbabilities) : Real :=
  1 - p.eight

theorem shooting_probabilities (p : RingProbabilities) 
  (h : p = given_probs) : 
  prob_ten_or_nine p = 0.52 ∧ 
  prob_at_least_seven p = 0.87 ∧ 
  prob_not_eight p = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l960_96081


namespace NUMINAMATH_CALUDE_soccer_season_games_l960_96055

/-- The number of months in the soccer season -/
def season_length : ℕ := 3

/-- The number of soccer games played per month -/
def games_per_month : ℕ := 9

/-- The total number of soccer games played during the season -/
def total_games : ℕ := season_length * games_per_month

theorem soccer_season_games : total_games = 27 := by
  sorry

end NUMINAMATH_CALUDE_soccer_season_games_l960_96055


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l960_96033

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem gcd_factorial_problem : Nat.gcd (factorial 7) ((factorial 10) / (factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l960_96033


namespace NUMINAMATH_CALUDE_thirteenth_divisible_by_three_l960_96016

theorem thirteenth_divisible_by_three (start : ℕ) (count : ℕ) : 
  start > 10 → 
  start % 3 = 0 → 
  ∀ n < start, n > 10 → n % 3 ≠ 0 →
  count = 13 →
  (start + 3 * (count - 1) = 48) :=
sorry

end NUMINAMATH_CALUDE_thirteenth_divisible_by_three_l960_96016


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l960_96019

theorem opposite_of_negative_six : -((-6 : ℝ)) = (6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l960_96019


namespace NUMINAMATH_CALUDE_mary_marbles_count_l960_96049

/-- The number of yellow marbles Mary and Joan have in total -/
def total_marbles : ℕ := 12

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := total_marbles - joan_marbles

theorem mary_marbles_count : mary_marbles = 9 := by
  sorry

end NUMINAMATH_CALUDE_mary_marbles_count_l960_96049


namespace NUMINAMATH_CALUDE_percentage_of_difference_l960_96026

theorem percentage_of_difference (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (40 / 100) * (x + y) →
  y = (11.11111111111111 / 100) * x →
  P = 6.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_difference_l960_96026


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l960_96056

theorem log_expression_equals_two :
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 50) + Real.log 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l960_96056


namespace NUMINAMATH_CALUDE_joe_test_scores_l960_96001

theorem joe_test_scores (initial_avg : ℚ) (lowest_score : ℚ) (new_avg : ℚ) 
  (h1 : initial_avg = 90)
  (h2 : lowest_score = 75)
  (h3 : new_avg = 85) :
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_avg - lowest_score = (n - 1 : ℚ) * new_avg ∧
    n = 13 := by
sorry

end NUMINAMATH_CALUDE_joe_test_scores_l960_96001


namespace NUMINAMATH_CALUDE_linear_combination_existence_l960_96089

theorem linear_combination_existence (n : ℕ+) (a b c : ℕ+) 
  (ha : a ≤ 3*n^2 + 4*n) (hb : b ≤ 3*n^2 + 4*n) (hc : c ≤ 3*n^2 + 4*n) :
  ∃ (x y z : ℤ), 
    (abs x ≤ 2*n ∧ abs y ≤ 2*n ∧ abs z ≤ 2*n) ∧ 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    (a*x + b*y + c*z = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_combination_existence_l960_96089


namespace NUMINAMATH_CALUDE_trapezoidal_dam_pressure_l960_96031

/-- 
Represents a vertical trapezoidal dam with water pressure.
-/
structure TrapezoidalDam where
  ρ : ℝ  -- density of water
  g : ℝ  -- acceleration due to gravity
  h : ℝ  -- height of the dam
  a : ℝ  -- top width of the dam
  b : ℝ  -- bottom width of the dam
  h_pos : h > 0
  a_pos : a > 0
  b_pos : b > 0
  a_ge_b : a ≥ b

/-- 
The total water pressure on a vertical trapezoidal dam is ρg(h^2(2a + b))/6.
-/
theorem trapezoidal_dam_pressure (dam : TrapezoidalDam) :
  ∃ P : ℝ, P = dam.ρ * dam.g * (dam.h^2 * (2 * dam.a + dam.b)) / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_trapezoidal_dam_pressure_l960_96031


namespace NUMINAMATH_CALUDE_motorboat_speed_l960_96035

/-- Prove that the maximum speed of a motorboat in still water is 40 km/h given the specified conditions -/
theorem motorboat_speed (flood_rate : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  flood_rate = 10 →
  downstream_distance = 2 →
  upstream_distance = 1.2 →
  (downstream_distance / (v + flood_rate) = upstream_distance / (v - flood_rate)) →
  v = 40 :=
by
  sorry

#check motorboat_speed

end NUMINAMATH_CALUDE_motorboat_speed_l960_96035


namespace NUMINAMATH_CALUDE_rectangle_area_l960_96094

theorem rectangle_area (width height : ℝ) (h1 : width / height = 0.875) (h2 : height = 24) :
  width * height = 504 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l960_96094


namespace NUMINAMATH_CALUDE_rectangle_area_change_l960_96051

theorem rectangle_area_change (original_area : ℝ) (length_decrease : ℝ) (width_increase : ℝ) :
  original_area = 600 →
  length_decrease = 0.2 →
  width_increase = 0.05 →
  original_area * (1 - length_decrease) * (1 + width_increase) = 504 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l960_96051


namespace NUMINAMATH_CALUDE_ball_count_theorem_l960_96079

/-- Represents the number of balls of each color in a box -/
structure BallCount where
  white : ℕ
  red : ℕ
  blue : ℕ

/-- Checks if the given ball counts satisfy the ratio 4:3:2 for white:red:blue -/
def satisfiesRatio (bc : BallCount) : Prop :=
  4 * bc.red = 3 * bc.white ∧ 4 * bc.blue = 2 * bc.white

theorem ball_count_theorem (bc : BallCount) 
    (h_ratio : satisfiesRatio bc) 
    (h_white : bc.white = 12) : 
    bc.red = 9 ∧ bc.blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l960_96079


namespace NUMINAMATH_CALUDE_modulo_equivalence_in_range_l960_96025

theorem modulo_equivalence_in_range : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [MOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_in_range_l960_96025


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_nonequality_l960_96059

theorem negation_of_existence_is_universal_nonequality :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_nonequality_l960_96059


namespace NUMINAMATH_CALUDE_limit_special_function_l960_96038

/-- The limit of (5 - 4/cos(x))^(1/sin^2(3x)) as x approaches 0 is e^(-2/9) -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
      |(5 - 4 / Real.cos x) ^ (1 / Real.sin (3 * x) ^ 2) - Real.exp (-2/9)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_special_function_l960_96038


namespace NUMINAMATH_CALUDE_linear_system_solution_l960_96099

theorem linear_system_solution (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 0)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 10)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 100) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 210 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l960_96099


namespace NUMINAMATH_CALUDE_min_sum_floor_l960_96066

theorem min_sum_floor (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ⌊(a+b+c)/d⌋ + ⌊(a+b+d)/c⌋ + ⌊(a+c+d)/b⌋ + ⌊(b+c+d)/a⌋ ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_floor_l960_96066


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l960_96053

/-- If the solution set of the inequality |ax+2| < 6 is (-1,2), then a = -4 -/
theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |a*x + 2| < 6 ↔ -1 < x ∧ x < 2) →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l960_96053


namespace NUMINAMATH_CALUDE_die_roll_probability_l960_96091

/-- The probability of getting a specific number on a standard six-sided die -/
def prob_match : ℚ := 1 / 6

/-- The probability of not getting a specific number on a standard six-sided die -/
def prob_no_match : ℚ := 5 / 6

/-- The number of rolls -/
def n : ℕ := 12

/-- The number of ways to choose the position of the first pair of consecutive matches -/
def ways_to_choose_first_pair : ℕ := n - 2

theorem die_roll_probability :
  (ways_to_choose_first_pair : ℚ) * prob_no_match^(n - 3) * prob_match^2 = 19531250 / 362797056 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l960_96091


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l960_96078

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (n : ℤ) = 7 ∧ 
    Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n ∧
    ∀ m : ℕ, m > n → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l960_96078


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l960_96040

theorem average_of_remaining_numbers
  (n : ℕ)
  (total : ℕ)
  (subset : ℕ)
  (avg_all : ℚ)
  (avg_subset : ℚ)
  (h_total : n = 5)
  (h_subset : subset = 3)
  (h_avg_all : avg_all = 6)
  (h_avg_subset : avg_subset = 4) :
  (n * avg_all - subset * avg_subset) / (n - subset) = 9 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l960_96040


namespace NUMINAMATH_CALUDE_no_solution_to_system_l960_96042

theorem no_solution_to_system :
  ¬∃ (x y : ℝ), 
    (80 * x + 15 * y - 7) / (78 * x + 12 * y) = 1 ∧
    (2 * x^2 + 3 * y^2 - 11) / (y^2 - x^2 + 3) = 1 ∧
    78 * x + 12 * y ≠ 0 ∧
    y^2 - x^2 + 3 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l960_96042


namespace NUMINAMATH_CALUDE_logarithm_sum_approximation_l960_96068

theorem logarithm_sum_approximation : 
  let expr := (1 / (Real.log 3 / Real.log 8 + 1)) + 
              (1 / (Real.log 2 / Real.log 12 + 1)) + 
              (1 / (Real.log 4 / Real.log 9 + 1))
  ∃ ε > 0, |expr - 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_approximation_l960_96068


namespace NUMINAMATH_CALUDE_chapters_read_l960_96047

/-- Represents the number of pages in each chapter of the book --/
def pages_per_chapter : ℕ := 8

/-- Represents the total number of pages Tom read --/
def total_pages_read : ℕ := 24

/-- Theorem stating that the number of chapters Tom read is 3 --/
theorem chapters_read : (total_pages_read / pages_per_chapter : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chapters_read_l960_96047


namespace NUMINAMATH_CALUDE_sum_greatest_odd_divisors_formula_l960_96095

/-- The greatest odd divisor of a positive integer -/
def greatest_odd_divisor (k : ℕ+) : ℕ+ :=
  sorry

/-- The sum of greatest odd divisors from 1 to 2^n -/
def sum_greatest_odd_divisors (n : ℕ+) : ℕ+ :=
  sorry

theorem sum_greatest_odd_divisors_formula (n : ℕ+) :
  (sum_greatest_odd_divisors n : ℚ) = (4^(n : ℕ) + 5) / 3 :=
sorry

end NUMINAMATH_CALUDE_sum_greatest_odd_divisors_formula_l960_96095


namespace NUMINAMATH_CALUDE_largest_square_area_l960_96075

theorem largest_square_area (original_side : ℝ) (cut_side : ℝ) (largest_area : ℝ) : 
  original_side = 5 →
  cut_side = 1 →
  largest_area = (5 * Real.sqrt 2 / 2) ^ 2 →
  largest_area = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_area_l960_96075


namespace NUMINAMATH_CALUDE_initial_men_count_prove_initial_men_count_l960_96084

/-- The number of men initially working on a project -/
def initial_men : ℕ := 12

/-- The number of hours worked per day by the initial group -/
def initial_hours_per_day : ℕ := 8

/-- The number of days worked by the initial group -/
def initial_days : ℕ := 10

/-- The number of men in the second group -/
def second_group_men : ℕ := 5

/-- The number of hours worked per day by the second group -/
def second_hours_per_day : ℕ := 16

/-- The number of days worked by the second group -/
def second_days : ℕ := 12

theorem initial_men_count : 
  initial_men * initial_hours_per_day * initial_days = 
  second_group_men * second_hours_per_day * second_days :=
by sorry

/-- The main theorem proving the number of men initially working -/
theorem prove_initial_men_count : initial_men = 12 :=
by sorry

end NUMINAMATH_CALUDE_initial_men_count_prove_initial_men_count_l960_96084


namespace NUMINAMATH_CALUDE_lucille_earnings_l960_96043

/-- Represents the number of weeds in different areas of the garden -/
structure GardenWeeds where
  flower_bed : Nat
  vegetable_patch : Nat
  grass : Nat

/-- Represents Lucille's weeding and earnings -/
def LucilleWeeding (garden : GardenWeeds) (soda_cost : Nat) (money_left : Nat) : Prop :=
  let weeds_pulled := garden.flower_bed + garden.vegetable_patch + garden.grass / 2
  let total_earnings := soda_cost + money_left
  total_earnings / weeds_pulled = 6

/-- Theorem: Given the garden conditions and Lucille's spending, she earns 6 cents per weed -/
theorem lucille_earnings (garden : GardenWeeds) 
  (h1 : garden.flower_bed = 11)
  (h2 : garden.vegetable_patch = 14)
  (h3 : garden.grass = 32)
  (h4 : LucilleWeeding garden 99 147) : 
  ∃ (earnings_per_weed : Nat), earnings_per_weed = 6 := by
  sorry

end NUMINAMATH_CALUDE_lucille_earnings_l960_96043


namespace NUMINAMATH_CALUDE_problem_solution_l960_96020

theorem problem_solution (a b c d : ℤ) 
  (h1 : a = d)
  (h2 : b = c)
  (h3 : d + d = c * d)
  (h4 : b = d)
  (h5 : d + d = d * d)
  (h6 : c = 3) :
  a * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l960_96020


namespace NUMINAMATH_CALUDE_stratified_sampling_seniors_l960_96072

theorem stratified_sampling_seniors (total_population : ℕ) (senior_population : ℕ) (sample_size : ℕ) : 
  total_population = 2100 → 
  senior_population = 680 → 
  sample_size = 105 → 
  (sample_size * senior_population) / total_population = 34 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_seniors_l960_96072


namespace NUMINAMATH_CALUDE_sin_75_cos_45_minus_cos_75_sin_45_l960_96039

theorem sin_75_cos_45_minus_cos_75_sin_45 :
  Real.sin (75 * π / 180) * Real.cos (45 * π / 180) -
  Real.cos (75 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_45_minus_cos_75_sin_45_l960_96039


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l960_96077

theorem rectangle_dimensions : ∃ (l w : ℝ), 
  (l = 9 ∧ w = 8) ∧
  (l - 3 = w - 2) ∧
  ((l - 3)^2 = (1/2) * l * w) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l960_96077


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l960_96074

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧
  (∃ a b : ℝ, a + b > 2 ∧ a * b > 1 ∧ ¬(a > 1 ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l960_96074


namespace NUMINAMATH_CALUDE_y_intercept_of_given_line_l960_96076

/-- A line in the form y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis -/
def y_intercept (l : Line) : ℝ := l.b

/-- Given line with equation y = 3x + 2 -/
def given_line : Line := { m := 3, b := 2 }

/-- Theorem: The y-intercept of the given line is 2 -/
theorem y_intercept_of_given_line : 
  y_intercept given_line = 2 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_given_line_l960_96076


namespace NUMINAMATH_CALUDE_y_values_l960_96000

theorem y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  let y := ((x - 3)^2 * (x + 2)) / (2 * x - 4)
  y = 9 ∨ y = 3.6 :=
by sorry

end NUMINAMATH_CALUDE_y_values_l960_96000


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_l960_96002

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 4536 →
  min a b = 54 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_l960_96002


namespace NUMINAMATH_CALUDE_go_square_side_count_l960_96092

/-- Represents a square arrangement of Go stones -/
structure GoSquare where
  side_length : ℕ
  perimeter_stones : ℕ

/-- The number of stones on one side of a GoSquare -/
def stones_on_side (square : GoSquare) : ℕ := square.side_length

/-- The number of stones on the perimeter of a GoSquare -/
def perimeter_count (square : GoSquare) : ℕ := square.perimeter_stones

theorem go_square_side_count (square : GoSquare) 
  (h : perimeter_count square = 84) : 
  stones_on_side square = 22 := by
  sorry

end NUMINAMATH_CALUDE_go_square_side_count_l960_96092


namespace NUMINAMATH_CALUDE_no_real_solution_cos_sin_l960_96012

theorem no_real_solution_cos_sin : ¬∃ (x : ℝ), (Real.cos x = 1/2) ∧ (Real.sin x = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_cos_sin_l960_96012


namespace NUMINAMATH_CALUDE_golden_rabbit_cards_l960_96071

def total_cards : ℕ := 10000
def digits_without_6_8 : ℕ := 8

theorem golden_rabbit_cards :
  (total_cards - digits_without_6_8^4 : ℕ) = 5904 := by sorry

end NUMINAMATH_CALUDE_golden_rabbit_cards_l960_96071


namespace NUMINAMATH_CALUDE_solution_set_min_value_min_value_expression_l960_96021

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Theorem stating the solution set for f(x) ≤ 4
theorem solution_set : {x : ℝ | f x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Theorem stating the minimum value of f(x)
theorem min_value : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ m = 3 := by sorry

-- Theorem for the minimum value of 1/(a-1) + 2/b
theorem min_value_expression :
  ∀ (a b : ℝ), a > 1 → b > 0 → a + 2*b = 3 →
  ∀ (y : ℝ), y = 1/(a-1) + 2/b → y ≥ 9/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_min_value_expression_l960_96021


namespace NUMINAMATH_CALUDE_polynomial_integer_solutions_l960_96097

theorem polynomial_integer_solutions :
  ∀ n : ℤ, n^5 - 2*n^4 - 7*n^2 - 7*n + 3 = 0 ↔ n = -1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_integer_solutions_l960_96097


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_equals_twice_perimeter_l960_96010

/-- Given a triangle with area A, perimeter p, semiperimeter s, and inradius r -/
theorem inscribed_circle_radius_when_area_equals_twice_perimeter 
  (A : ℝ) (p : ℝ) (s : ℝ) (r : ℝ) 
  (h1 : A = 2 * p)  -- Area is twice the perimeter
  (h2 : p = 2 * s)  -- Perimeter is twice the semiperimeter
  (h3 : A = r * s)  -- Area formula for a triangle
  (h4 : s ≠ 0)      -- Semiperimeter is non-zero
  : r = 4 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_equals_twice_perimeter_l960_96010


namespace NUMINAMATH_CALUDE_candy_store_lollipops_l960_96086

/-- The number of milliliters of food coloring used for each lollipop -/
def lollipop_coloring : ℕ := 5

/-- The number of milliliters of food coloring used for each hard candy -/
def hard_candy_coloring : ℕ := 20

/-- The number of hard candies made -/
def hard_candies_made : ℕ := 5

/-- The total amount of food coloring used in milliliters -/
def total_coloring_used : ℕ := 600

/-- The number of lollipops made -/
def lollipops_made : ℕ := 100

theorem candy_store_lollipops :
  lollipops_made * lollipop_coloring + hard_candies_made * hard_candy_coloring = total_coloring_used :=
by sorry

end NUMINAMATH_CALUDE_candy_store_lollipops_l960_96086


namespace NUMINAMATH_CALUDE_rounding_and_multiplication_l960_96032

/-- Round a number to the nearest significant figure -/
def roundToSignificantFigure (x : ℝ) : ℝ := sorry

/-- Round a number up to the nearest hundred -/
def roundUpToHundred (x : ℝ) : ℕ := sorry

/-- The main theorem -/
theorem rounding_and_multiplication :
  let a := 0.000025
  let b := 6546300
  let rounded_a := roundToSignificantFigure a
  let rounded_b := roundToSignificantFigure b
  let product := rounded_a * rounded_b
  roundUpToHundred product = 200 := by sorry

end NUMINAMATH_CALUDE_rounding_and_multiplication_l960_96032


namespace NUMINAMATH_CALUDE_remaining_segments_length_is_23_l960_96067

/-- Represents a polygon with perpendicular adjacent sides -/
structure Polygon where
  vertical_height : ℕ
  top_horizontal : ℕ
  first_descent : ℕ
  middle_horizontal : ℕ
  final_descent : ℕ

/-- Calculates the length of segments in the new figure after removing four sides -/
def remaining_segments_length (p : Polygon) : ℕ :=
  p.vertical_height + (p.top_horizontal + p.middle_horizontal) + 
  (p.first_descent + p.final_descent) + p.middle_horizontal

/-- The original polygon described in the problem -/
def original_polygon : Polygon :=
  { vertical_height := 7
  , top_horizontal := 3
  , first_descent := 2
  , middle_horizontal := 4
  , final_descent := 3 }

theorem remaining_segments_length_is_23 :
  remaining_segments_length original_polygon = 23 := by
  sorry

#eval remaining_segments_length original_polygon

end NUMINAMATH_CALUDE_remaining_segments_length_is_23_l960_96067


namespace NUMINAMATH_CALUDE_smallest_b_for_composite_l960_96022

theorem smallest_b_for_composite (b : ℕ) (h : b = 8) :
  (∀ x : ℤ, ∃ y z : ℤ, y ≠ 1 ∧ z ≠ 1 ∧ y * z = x^4 + b^4) ∧
  (∀ b' : ℕ, 0 < b' ∧ b' < b →
    ∃ x : ℤ, ∀ y z : ℤ, (y * z = x^4 + b'^4) → (y = 1 ∨ z = 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_composite_l960_96022


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l960_96003

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x ≠ 5 ∧ x^2 - 4*x - 5 = 0) ∧ 
  (∀ x : ℝ, x = 5 → x^2 - 4*x - 5 = 0) := by
  sorry

#check sufficient_not_necessary

end NUMINAMATH_CALUDE_sufficient_not_necessary_l960_96003


namespace NUMINAMATH_CALUDE_angle_at_point_l960_96011

theorem angle_at_point (x : ℝ) : 
  x > 0 ∧ x + x + 140 = 360 → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_angle_at_point_l960_96011


namespace NUMINAMATH_CALUDE_balance_four_hearts_l960_96023

/-- Represents the weight of a symbol in the balance game -/
structure Weight (α : Type) where
  value : ℚ

/-- The balance game with three symbols -/
structure BalanceGame where
  star : Weight ℚ
  heart : Weight ℚ
  circle : Weight ℚ

/-- Defines the balance equations for the game -/
def balance_equations (game : BalanceGame) : Prop :=
  4 * game.star.value + 3 * game.heart.value = 12 * game.circle.value ∧
  2 * game.star.value = game.heart.value + 3 * game.circle.value

/-- The main theorem to prove -/
theorem balance_four_hearts (game : BalanceGame) :
  balance_equations game →
  4 * game.heart.value = 5 * game.circle.value :=
by sorry

end NUMINAMATH_CALUDE_balance_four_hearts_l960_96023


namespace NUMINAMATH_CALUDE_sum_of_m_and_n_is_zero_l960_96087

theorem sum_of_m_and_n_is_zero (m n p : ℝ) 
  (h1 : m * n + p^2 + 4 = 0) 
  (h2 : m - n = 4) : 
  m + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_and_n_is_zero_l960_96087


namespace NUMINAMATH_CALUDE_no_alpha_sequence_exists_l960_96005

theorem no_alpha_sequence_exists : ¬ ∃ (α : ℝ) (a : ℕ → ℝ),
  (0 < α ∧ α < 1) ∧
  (∀ n, 0 < a n) ∧
  (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end NUMINAMATH_CALUDE_no_alpha_sequence_exists_l960_96005


namespace NUMINAMATH_CALUDE_hyperbola_center_l960_96014

theorem hyperbola_center (x y : ℝ) :
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 71 = 0 →
  ∃ a b : ℝ, a = 3 ∧ b = 4 ∧
  ∀ x' y' : ℝ, 9 * (x' - a)^2 - 16 * (y' - b)^2 = 9 * x'^2 - 54 * x' - 16 * y'^2 + 128 * y' - 71 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l960_96014


namespace NUMINAMATH_CALUDE_sum_consecutive_triangular_numbers_l960_96013

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_consecutive_triangular_numbers (n : ℕ) :
  triangular_number n + triangular_number (n + 1) = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_triangular_numbers_l960_96013


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l960_96063

/-- A geometric sequence with third term 16 and seventh term 2 has fifth term 8 -/
theorem geometric_sequence_fifth_term (a : ℝ) (r : ℝ) 
  (h1 : a * r^2 = 16)  -- third term is 16
  (h2 : a * r^6 = 2)   -- seventh term is 2
  : a * r^4 = 8 :=     -- fifth term is 8
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l960_96063


namespace NUMINAMATH_CALUDE_ratio_of_sides_l960_96098

/-- A rectangle with a point inside dividing it into four triangles -/
structure DividedRectangle where
  -- The lengths of the sides of the rectangle
  AB : ℝ
  BC : ℝ
  -- The areas of the four triangles
  area_APD : ℝ
  area_BPA : ℝ
  area_CPB : ℝ
  area_DPC : ℝ
  -- Conditions
  positive_AB : 0 < AB
  positive_BC : 0 < BC
  diagonal_condition : AB^2 + BC^2 = (2*AB)^2
  area_condition : area_APD = 1 ∧ area_BPA = 2 ∧ area_CPB = 3 ∧ area_DPC = 4

/-- The theorem stating the ratio of sides in the divided rectangle -/
theorem ratio_of_sides (r : DividedRectangle) : r.AB / r.BC = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sides_l960_96098


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l960_96083

theorem triangle_area_in_circle (R : ℝ) (α β : ℝ) (h_R : R = 2) 
  (h_α : α = π / 3) (h_β : β = π / 4) :
  let γ : ℝ := π - α - β
  let a : ℝ := 2 * R * Real.sin α
  let b : ℝ := 2 * R * Real.sin β
  let c : ℝ := 2 * R * Real.sin γ
  let S : ℝ := (Real.sqrt 3 + 3 : ℝ)
  S = (1 / 2) * a * b * Real.sin γ := by sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l960_96083


namespace NUMINAMATH_CALUDE_power_equation_solution_l960_96004

theorem power_equation_solution (n : Real) : 
  10^n = 10^4 * Real.sqrt (10^155 / 0.0001) → n = 83.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l960_96004


namespace NUMINAMATH_CALUDE_roots_of_equation_l960_96052

theorem roots_of_equation : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x * (x - 1) = x ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 2 ∧ x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l960_96052


namespace NUMINAMATH_CALUDE_tan_theta_equals_five_twelfths_l960_96027

/-- Given a dilation matrix D and a rotation matrix R, prove that tan θ = 5/12 -/
theorem tan_theta_equals_five_twelfths 
  (k : ℝ) 
  (θ : ℝ) 
  (hk : k > 0) 
  (D : Matrix (Fin 2) (Fin 2) ℝ) 
  (R : Matrix (Fin 2) (Fin 2) ℝ) 
  (hD : D = ![![k, 0], ![0, k]]) 
  (hR : R = ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]) 
  (h_prod : R * D = ![![12, -5], ![5, 12]]) : 
  Real.tan θ = 5/12 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_equals_five_twelfths_l960_96027


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l960_96009

/-- A natural number is composite if it has a proper factor -/
def IsComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A natural number can be represented as the sum of two composite numbers -/
def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

/-- 11 is the largest natural number that cannot be represented as the sum of two composite numbers -/
theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l960_96009


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_ge_four_l960_96046

/-- The function f(x) = ax - x^3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

/-- The theorem statement -/
theorem f_increasing_iff_a_ge_four (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 → f a x2 - f a x1 > x2 - x1) ↔ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_ge_four_l960_96046


namespace NUMINAMATH_CALUDE_dagger_example_l960_96008

-- Define the ⋄ operation
def dagger (m n p q : ℚ) : ℚ := m^2 * p * (q / n)

-- Theorem statement
theorem dagger_example : dagger 5 9 4 6 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l960_96008


namespace NUMINAMATH_CALUDE_water_polo_team_selection_result_l960_96028

/-- The number of ways to choose a starting team in water polo -/
def water_polo_team_selection (total_players : ℕ) (team_size : ℕ) (goalie_count : ℕ) : ℕ :=
  Nat.choose total_players goalie_count * Nat.choose (total_players - goalie_count) (team_size - goalie_count)

/-- Theorem: The number of ways to choose a starting team of 9 players (including 2 goalies) from a team of 20 members is 6,046,560 -/
theorem water_polo_team_selection_result :
  water_polo_team_selection 20 9 2 = 6046560 := by
  sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_result_l960_96028


namespace NUMINAMATH_CALUDE_two_times_binomial_seven_choose_four_l960_96057

theorem two_times_binomial_seven_choose_four : 2 * (Nat.choose 7 4) = 70 := by
  sorry

end NUMINAMATH_CALUDE_two_times_binomial_seven_choose_four_l960_96057


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l960_96034

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 + (a - 1) * x - 1 < 0) ↔ -3/5 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l960_96034


namespace NUMINAMATH_CALUDE_lowest_price_for_electronic_component_l960_96036

/-- Calculates the lowest price per component to break even -/
def lowest_break_even_price (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (units_sold : ℕ) : ℚ :=
  (production_cost + shipping_cost + (fixed_costs / units_sold))

theorem lowest_price_for_electronic_component :
  let production_cost : ℚ := 80
  let shipping_cost : ℚ := 3
  let fixed_costs : ℚ := 16500
  let units_sold : ℕ := 150
  lowest_break_even_price production_cost shipping_cost fixed_costs units_sold = 193 := by
sorry

#eval lowest_break_even_price 80 3 16500 150

end NUMINAMATH_CALUDE_lowest_price_for_electronic_component_l960_96036


namespace NUMINAMATH_CALUDE_quadratic_cubic_inequalities_l960_96015

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem quadratic_cubic_inequalities 
  (m n a b : ℝ) 
  (h1 : n = 0)
  (h2 : -2 * m + n = -2)
  (h3 : m * 1^2 + n * 1 = a * 1^3 + b * 1 - 3)
  (h4 : 2 * m * 1 + n = 3 * a * 1^2 + b) :
  ∃ (k p : ℝ), k = 2 ∧ p = -1 ∧ 
  (∀ x > 0, f m n x ≥ k * x + p ∧ g a b x ≤ k * x + p) := by
sorry

end NUMINAMATH_CALUDE_quadratic_cubic_inequalities_l960_96015


namespace NUMINAMATH_CALUDE_roof_dimensions_l960_96073

theorem roof_dimensions (width : ℝ) (length : ℝ) :
  length = 4 * width →
  width * length = 576 →
  length - width = 36 := by
sorry

end NUMINAMATH_CALUDE_roof_dimensions_l960_96073


namespace NUMINAMATH_CALUDE_cheryl_skittles_l960_96080

theorem cheryl_skittles (initial : ℕ) (given : ℕ) (final : ℕ) : 
  given = 89 → final = 97 → initial + given = final → initial = 8 := by
sorry

end NUMINAMATH_CALUDE_cheryl_skittles_l960_96080


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l960_96018

/-- Given a cube with surface area 6a^2 and all its vertices on a sphere,
    prove that the surface area of the sphere is 3πa^2 -/
theorem sphere_surface_area_from_cube (a : ℝ) (h : a > 0) :
  let cube_surface_area := 6 * a^2
  let cube_diagonal := a * Real.sqrt 3
  let sphere_radius := cube_diagonal / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  sphere_surface_area = 3 * Real.pi * a^2 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l960_96018


namespace NUMINAMATH_CALUDE_annual_piano_clarinet_cost_difference_l960_96065

/-- Calculates the difference in annual cost between piano and clarinet lessons --/
def annual_lesson_cost_difference (clarinet_hourly_rate piano_hourly_rate : ℕ) 
  (clarinet_weekly_hours piano_weekly_hours : ℕ) (weeks_per_year : ℕ) : ℕ :=
  ((piano_hourly_rate * piano_weekly_hours) - (clarinet_hourly_rate * clarinet_weekly_hours)) * weeks_per_year

/-- Proves that the difference in annual cost between piano and clarinet lessons is $1040 --/
theorem annual_piano_clarinet_cost_difference : 
  annual_lesson_cost_difference 40 28 3 5 52 = 1040 := by
  sorry

end NUMINAMATH_CALUDE_annual_piano_clarinet_cost_difference_l960_96065


namespace NUMINAMATH_CALUDE_sqrt_equation_l960_96029

theorem sqrt_equation (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l960_96029


namespace NUMINAMATH_CALUDE_arccos_cos_three_l960_96096

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l960_96096


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l960_96060

theorem smaller_number_in_ratio (a b : ℝ) : 
  a / b = 3 / 4 → a + b = 420 → a = 180 := by sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l960_96060


namespace NUMINAMATH_CALUDE_inequality_proof_l960_96044

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  b * c^2 + c * a^2 + a * b^2 < b^2 * c + c^2 * a + a^2 * b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l960_96044


namespace NUMINAMATH_CALUDE_smallest_number_l960_96093

theorem smallest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -3) (hc : c = 1) (hd : d = -1) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by sorry

end NUMINAMATH_CALUDE_smallest_number_l960_96093


namespace NUMINAMATH_CALUDE_carls_membership_number_l960_96050

/-- A predicate to check if a number is a two-digit prime -/
def isTwoDigitPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

/-- The main theorem -/
theorem carls_membership_number
  (a b c d : ℕ)
  (ha : isTwoDigitPrime a)
  (hb : isTwoDigitPrime b)
  (hc : isTwoDigitPrime c)
  (hd : isTwoDigitPrime d)
  (sum_all : a + b + c + d = 100)
  (sum_no_ben : b + c + d = 30)
  (sum_no_carl : a + b + d = 29)
  (sum_no_david : a + b + c = 23) :
  c = 23 := by
  sorry


end NUMINAMATH_CALUDE_carls_membership_number_l960_96050


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l960_96006

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 145 → ∃ (a b : ℕ), a^2 - b^2 = 145 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 433 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l960_96006


namespace NUMINAMATH_CALUDE_wage_increase_percentage_l960_96088

theorem wage_increase_percentage (original_wage new_wage : ℝ) 
  (h1 : original_wage = 34)
  (h2 : new_wage = 51) :
  (new_wage - original_wage) / original_wage * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_percentage_l960_96088


namespace NUMINAMATH_CALUDE_flower_purchase_solution_l960_96058

/-- Represents the flower purchase problem -/
structure FlowerPurchase where
  priceA : ℕ  -- Price of type A flower
  priceB : ℕ  -- Price of type B flower
  totalPlants : ℕ  -- Total number of plants to purchase
  (first_purchase : 30 * priceA + 15 * priceB = 675)
  (second_purchase : 12 * priceA + 5 * priceB = 265)
  (total_constraint : totalPlants = 31)
  (type_b_constraint : ∀ m : ℕ, m ≤ totalPlants → totalPlants - m < 2 * m)

/-- Theorem stating the solution to the flower purchase problem -/
theorem flower_purchase_solution (fp : FlowerPurchase) :
  fp.priceA = 20 ∧ fp.priceB = 5 ∧
  ∃ (m : ℕ), m = 11 ∧ fp.totalPlants - m = 20 ∧
  20 * m + 5 * (fp.totalPlants - m) = 320 ∧
  ∀ (n : ℕ), n ≤ fp.totalPlants → 
    20 * n + 5 * (fp.totalPlants - n) ≥ 20 * m + 5 * (fp.totalPlants - m) :=
by sorry


end NUMINAMATH_CALUDE_flower_purchase_solution_l960_96058
