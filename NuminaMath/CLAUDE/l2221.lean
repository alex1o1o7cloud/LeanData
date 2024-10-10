import Mathlib

namespace value_of_M_l2221_222120

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.55 * 1500) ∧ (M = 3300) := by
  sorry

end value_of_M_l2221_222120


namespace fourth_rectangle_area_l2221_222171

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: Area of the fourth rectangle in a divided large rectangle -/
theorem fourth_rectangle_area
  (a b : ℝ)
  (r1 r2 r3 r4 : Rectangle)
  (h1 : r1.width = 2*a ∧ r1.height = b)
  (h2 : r2.width = 3*a ∧ r2.height = b)
  (h3 : r3.width = 2*a ∧ r3.height = 2*b)
  (h4 : r4.width = 3*a ∧ r4.height = 2*b)
  (area1 : area r1 = 2*a*b)
  (area2 : area r2 = 6*a*b)
  (area3 : area r3 = 4*a*b) :
  area r4 = 6*a*b :=
sorry

end fourth_rectangle_area_l2221_222171


namespace bumper_car_line_problem_l2221_222160

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 6 + 3 = 6) → initial_people = 9 := by
  sorry

end bumper_car_line_problem_l2221_222160


namespace prob_one_of_A_or_B_is_two_thirds_l2221_222186

/-- The number of study groups -/
def num_groups : ℕ := 4

/-- The number of groups to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly one of group A and group B -/
def prob_one_of_A_or_B : ℚ := 2/3

/-- Theorem stating that the probability of selecting exactly one of group A and group B
    when randomly selecting two groups out of four groups is 2/3 -/
theorem prob_one_of_A_or_B_is_two_thirds :
  prob_one_of_A_or_B = (num_groups - 2) / (Nat.choose num_groups num_selected) := by
  sorry

end prob_one_of_A_or_B_is_two_thirds_l2221_222186


namespace even_sum_necessary_not_sufficient_l2221_222137

theorem even_sum_necessary_not_sufficient :
  (∀ a b : ℤ, (Even a ∧ Even b) → Even (a + b)) ∧
  (∃ a b : ℤ, Even (a + b) ∧ ¬(Even a ∧ Even b)) :=
by sorry

end even_sum_necessary_not_sufficient_l2221_222137


namespace candy_making_time_l2221_222102

/-- Candy-making process time calculation -/
theorem candy_making_time
  (initial_temp : ℝ)
  (target_temp : ℝ)
  (final_temp : ℝ)
  (heating_rate : ℝ)
  (cooling_rate : ℝ)
  (h1 : initial_temp = 60)
  (h2 : target_temp = 240)
  (h3 : final_temp = 170)
  (h4 : heating_rate = 5)
  (h5 : cooling_rate = 7) :
  (target_temp - initial_temp) / heating_rate + (target_temp - final_temp) / cooling_rate = 46 :=
by sorry

end candy_making_time_l2221_222102


namespace absolute_value_fraction_l2221_222124

theorem absolute_value_fraction (x y : ℝ) 
  (h : y < Real.sqrt (x - 1) + Real.sqrt (1 - x) + 1) : 
  |y - 1| / (y - 1) = -1 := by
  sorry

end absolute_value_fraction_l2221_222124


namespace man_speed_in_still_water_l2221_222107

/-- The speed of a man rowing in still water, given his upstream and downstream speeds -/
theorem man_speed_in_still_water (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 34)
  (h2 : downstream_speed = 48) :
  (upstream_speed + downstream_speed) / 2 = 41 := by
  sorry

end man_speed_in_still_water_l2221_222107


namespace first_term_of_ap_l2221_222190

def arithmetic_progression (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_progression (a₁ d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem first_term_of_ap (a₁ d : ℚ) :
  sum_arithmetic_progression a₁ d 22 = 1045 ∧
  sum_arithmetic_progression (arithmetic_progression a₁ d 23) d 22 = 2013 →
  a₁ = 53 / 2 := by
sorry

end first_term_of_ap_l2221_222190


namespace line_direction_vector_l2221_222162

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := (2, -1) + t • d
  let d : ℝ × ℝ := direction_vector line
  let y (x : ℝ) : ℝ := (2 * x + 3) / 5
  (∀ x ≥ 2, (x - 2) ^ 2 + (y x + 1) ^ 2 = t ^ 2 → 
    line t = (x, y x)) →
  d = (5 / Real.sqrt 29, 2 / Real.sqrt 29) :=
sorry

end line_direction_vector_l2221_222162


namespace f_properties_l2221_222104

open Real

noncomputable def f (x : ℝ) := Real.log (Real.exp (2 * x) + 1) - x

theorem f_properties :
  (∀ x, ∃ y, f x = y) ∧
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 ≤ x ∧ x < y → f x < f y) :=
by sorry

end f_properties_l2221_222104


namespace solve_equation_one_solve_equation_two_l2221_222184

-- Equation 1
theorem solve_equation_one : 
  ∃ x : ℝ, (3 * x - 4 = -2 * (x - 1)) ∧ (x = 1.2) := by sorry

-- Equation 2
theorem solve_equation_two :
  ∃ x : ℝ, (1 + (2 * x + 1) / 3 = (3 * x - 2) / 2) ∧ (x = 14 / 5) := by sorry

end solve_equation_one_solve_equation_two_l2221_222184


namespace initial_ducks_l2221_222183

theorem initial_ducks (initial final additional : ℕ) 
  (h1 : final = initial + additional)
  (h2 : final = 33)
  (h3 : additional = 20) : 
  initial = 13 := by
sorry

end initial_ducks_l2221_222183


namespace pyramid_volume_no_conditional_l2221_222176

/-- Algorithm to calculate triangle area from three side lengths -/
def triangle_area (a b c : ℝ) : ℝ := sorry

/-- Algorithm to calculate line slope from two points' coordinates -/
def line_slope (x1 y1 x2 y2 : ℝ) : ℝ := sorry

/-- Algorithm to calculate common logarithm of a number -/
noncomputable def common_log (x : ℝ) : ℝ := sorry

/-- Algorithm to calculate pyramid volume from base area and height -/
def pyramid_volume (base_area height : ℝ) : ℝ := sorry

/-- Predicate to check if an algorithm contains conditional statements -/
def has_conditional {α β : Type} (f : α → β) : Prop := sorry

theorem pyramid_volume_no_conditional :
  ¬ has_conditional pyramid_volume ∧
  has_conditional triangle_area ∧
  has_conditional line_slope ∧
  has_conditional common_log :=
by sorry

end pyramid_volume_no_conditional_l2221_222176


namespace simplify_expression_l2221_222156

theorem simplify_expression (n : ℕ) : (3^(n+4) - 3*(3^n)) / (3*(3^(n+3))) = 26 / 27 := by
  sorry

end simplify_expression_l2221_222156


namespace seating_arrangement_l2221_222111

theorem seating_arrangement (total_people : ℕ) (max_rows : ℕ) 
  (h1 : total_people = 57)
  (h2 : max_rows = 8) : 
  ∃ (rows_with_9 rows_with_6 : ℕ),
    rows_with_9 + rows_with_6 ≤ max_rows ∧
    9 * rows_with_9 + 6 * rows_with_6 = total_people ∧
    rows_with_9 = 5 := by
  sorry

end seating_arrangement_l2221_222111


namespace parabola_focus_and_directrix_l2221_222138

/-- Represents a parabola with equation x^2 = 4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = 4*y

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (0, 1)

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  fun y => y = -1

theorem parabola_focus_and_directrix (p : Parabola) :
  (focus p = (0, 1)) ∧ (directrix p = fun y => y = -1) := by
  sorry

end parabola_focus_and_directrix_l2221_222138


namespace crabapple_sequences_count_l2221_222117

/-- The number of ways to select 5 students from a group of 13 students,
    where the order matters and no student is selected more than once. -/
def crabapple_sequences : ℕ :=
  13 * 12 * 11 * 10 * 9

/-- Theorem stating that the number of crabapple recipient sequences is 154,440. -/
theorem crabapple_sequences_count : crabapple_sequences = 154440 := by
  sorry

end crabapple_sequences_count_l2221_222117


namespace fifteenth_prime_l2221_222154

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def nth_prime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime :
  (nth_prime 6 = 13) → (nth_prime 15 = 47) :=
sorry

end fifteenth_prime_l2221_222154


namespace largest_five_digit_divisible_by_five_l2221_222147

theorem largest_five_digit_divisible_by_five : 
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 5 = 0 → n ≤ 99995 :=
by sorry

end largest_five_digit_divisible_by_five_l2221_222147


namespace geometric_sequence_problem_l2221_222192

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_increasing : IsIncreasingGeometricSequence a)
    (h_a3 : a 3 = 4)
    (h_sum : 1 / a 1 + 1 / a 5 = 5 / 8) :
  a 7 = 16 := by
  sorry

end geometric_sequence_problem_l2221_222192


namespace third_quadrant_angle_sum_l2221_222152

theorem third_quadrant_angle_sum (θ : Real) : 
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.tan (θ - π/4) = 1/3) → 
  (Real.sin θ + Real.cos θ = -3/5 * Real.sqrt 5) := by
sorry

end third_quadrant_angle_sum_l2221_222152


namespace solution_pairs_l2221_222144

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def satisfies_conditions (a b : ℕ) : Prop :=
  (is_divisible_by (a - b) 3 ∨
   is_prime (a + 2*b) ∧
   a = 4*b - 1 ∧
   is_divisible_by (a + 7) b) ∧
  (¬(is_divisible_by (a - b) 3) ∨
   ¬(is_prime (a + 2*b)) ∨
   ¬(a = 4*b - 1) ∨
   ¬(is_divisible_by (a + 7) b))

theorem solution_pairs :
  ∀ a b : ℕ, satisfies_conditions a b ↔ (a = 3 ∧ b = 1) ∨ (a = 7 ∧ b = 2) ∨ (a = 11 ∧ b = 3) :=
sorry

end solution_pairs_l2221_222144


namespace clothing_production_solution_l2221_222195

/-- Represents the solution to the clothing production problem -/
def clothingProduction (totalFabric : ℝ) (topsPerUnit : ℝ) (pantsPerUnit : ℝ) (unitFabric : ℝ) 
  (fabricForTops : ℝ) (fabricForPants : ℝ) : Prop :=
  totalFabric > 0 ∧
  topsPerUnit > 0 ∧
  pantsPerUnit > 0 ∧
  unitFabric > 0 ∧
  fabricForTops ≥ 0 ∧
  fabricForPants ≥ 0 ∧
  fabricForTops + fabricForPants = totalFabric ∧
  (fabricForTops / unitFabric) * topsPerUnit = (fabricForPants / unitFabric) * pantsPerUnit

theorem clothing_production_solution :
  clothingProduction 600 2 3 3 360 240 := by
  sorry

end clothing_production_solution_l2221_222195


namespace trigonometric_roots_theorem_l2221_222125

-- Define the equation and its roots
def equation (m : ℝ) (x : ℝ) : Prop := 8 * x^2 + 6 * m * x + 2 * m + 1 = 0

-- Define the interval for α
def alpha_in_interval (α : ℝ) : Prop := 0 < α ∧ α < Real.pi

-- Theorem statement
theorem trigonometric_roots_theorem (α : ℝ) (m : ℝ) 
  (h1 : alpha_in_interval α)
  (h2 : equation m (Real.sin α))
  (h3 : equation m (Real.cos α)) :
  m = -10/9 ∧ 
  (Real.cos α + Real.sin α) * Real.tan α / (1 - Real.tan α^2) = 11 * Real.sqrt 47 / 564 :=
sorry

end trigonometric_roots_theorem_l2221_222125


namespace expression_square_l2221_222179

theorem expression_square (x y : ℕ) 
  (h : (1 : ℚ) / x + 1 / y + 1 / (x * y) = 1 / (x + 4) + 1 / (y - 4) + 1 / ((x + 4) * (y - 4))) : 
  ∃ n : ℕ, x * y + 4 = n^2 := by
sorry

end expression_square_l2221_222179


namespace min_both_brown_eyes_and_lunch_box_example_l2221_222141

/-- Given a class of students, calculates the minimum number of students
    who have both brown eyes and a lunch box. -/
def min_both_brown_eyes_and_lunch_box (total : ℕ) (brown_eyes : ℕ) (lunch_box : ℕ) : ℕ :=
  max 0 (brown_eyes + lunch_box - total)

/-- Theorem stating that in a class of 35 students, where 18 have brown eyes
    and 25 have a lunch box, at least 8 students have both brown eyes and a lunch box. -/
theorem min_both_brown_eyes_and_lunch_box_example :
  min_both_brown_eyes_and_lunch_box 35 18 25 = 8 := by
  sorry

end min_both_brown_eyes_and_lunch_box_example_l2221_222141


namespace sum_has_five_digits_l2221_222143

/-- A nonzero digit is a natural number between 1 and 9. -/
def NonzeroDigit : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- Convert a three-digit number represented by three digits to a natural number. -/
def threeDigitToNat (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Convert a two-digit number represented by two digits to a natural number. -/
def twoDigitToNat (a b : ℕ) : ℕ := 10 * a + b

/-- The main theorem: the sum of the four numbers always has 5 digits. -/
theorem sum_has_five_digits (A B C : NonzeroDigit) :
  ∃ (n : ℕ), 10000 ≤ 21478 + threeDigitToNat A.val 5 9 + twoDigitToNat B.val 4 + twoDigitToNat C.val 6 ∧
             21478 + threeDigitToNat A.val 5 9 + twoDigitToNat B.val 4 + twoDigitToNat C.val 6 < 100000 := by
  sorry

end sum_has_five_digits_l2221_222143


namespace number_of_dimes_l2221_222100

/-- Given a total of 11 coins, including 2 nickels and 7 quarters, prove that the number of dimes is 2 -/
theorem number_of_dimes (total : ℕ) (nickels : ℕ) (quarters : ℕ) (h1 : total = 11) (h2 : nickels = 2) (h3 : quarters = 7) :
  total - nickels - quarters = 2 := by
sorry

end number_of_dimes_l2221_222100


namespace smallest_sum_of_squares_l2221_222194

theorem smallest_sum_of_squares (x y : ℕ+) : 
  (x.val * (x.val + 1) ∣ y.val * (y.val + 1)) ∧ 
  (¬ (x.val ∣ y.val) ∧ ¬ (x.val ∣ (y.val + 1)) ∧ ¬ ((x.val + 1) ∣ y.val) ∧ ¬ ((x.val + 1) ∣ (y.val + 1))) →
  (∀ a b : ℕ+, 
    (a.val * (a.val + 1) ∣ b.val * (b.val + 1)) ∧ 
    (¬ (a.val ∣ b.val) ∧ ¬ (a.val ∣ (b.val + 1)) ∧ ¬ ((a.val + 1) ∣ b.val) ∧ ¬ ((a.val + 1) ∣ (b.val + 1))) →
    x.val^2 + y.val^2 ≤ a.val^2 + b.val^2) →
  x.val^2 + y.val^2 = 1421 := by sorry

end smallest_sum_of_squares_l2221_222194


namespace garden_length_l2221_222166

/-- Given a rectangular garden with perimeter 1200 meters and breadth 240 meters, 
    prove that its length is 360 meters. -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (length : ℝ) : 
  perimeter = 1200 ∧ 
  breadth = 240 ∧ 
  perimeter = 2 * (length + breadth) →
  length = 360 := by
sorry

end garden_length_l2221_222166


namespace problem_solution_l2221_222178

def f (x : ℝ) := |2*x - 3| + |2*x + 3|

theorem problem_solution :
  (∃ (M : ℝ),
    (∀ x, f x ≥ M) ∧
    (∃ x, f x = M) ∧
    (M = 6)) ∧
  ({x : ℝ | f x ≤ 8} = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) ∧
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    1/a + 1/(2*b) + 1/(3*c) = 1 →
    a + 2*b + 3*c ≥ 9) := by
  sorry

end problem_solution_l2221_222178


namespace consecutive_integers_square_sum_l2221_222112

theorem consecutive_integers_square_sum : 
  ∃ (n : ℤ), 
    (n + 1)^2 + (n + 2)^2 = (n - 2)^2 + (n - 1)^2 + n^2 ∧
    n = 12 :=
by sorry

end consecutive_integers_square_sum_l2221_222112


namespace final_balance_is_450_l2221_222170

/-- Calculates the final balance after withdrawal and deposit --/
def finalBalance (initialBalance : ℚ) : ℚ :=
  let remainingBalance := initialBalance - 200
  let depositAmount := remainingBalance / 2
  remainingBalance + depositAmount

/-- Theorem: The final balance is $450 given the conditions --/
theorem final_balance_is_450 :
  ∃ (initialBalance : ℚ),
    (initialBalance - 200 = initialBalance * (3/5)) ∧
    (finalBalance initialBalance = 450) :=
by sorry

end final_balance_is_450_l2221_222170


namespace factorization_equality_l2221_222164

theorem factorization_equality (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end factorization_equality_l2221_222164


namespace statement_A_necessary_not_sufficient_l2221_222133

theorem statement_A_necessary_not_sufficient :
  (∀ x y : ℝ, (x + y ≠ 5 → (x ≠ 2 ∨ y ≠ 3))) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 3) ∧ (x + y = 5)) := by
  sorry

end statement_A_necessary_not_sufficient_l2221_222133


namespace all_pairs_successful_probability_expected_successful_pairs_gt_half_l2221_222153

-- Define the number of sock pairs
variable (n : ℕ)

-- Define a successful pair
def successful_pair (pair : ℕ × ℕ) : Prop := pair.1 = pair.2

-- Define the probability of all pairs being successful
def all_pairs_successful_prob : ℚ := (2^n * n.factorial) / (2*n).factorial

-- Define the expected number of successful pairs
def expected_successful_pairs : ℚ := n / (2*n - 1)

-- Theorem 1: Probability of all pairs being successful
theorem all_pairs_successful_probability :
  all_pairs_successful_prob n = (2^n * n.factorial) / (2*n).factorial :=
sorry

-- Theorem 2: Expected number of successful pairs is greater than 0.5
theorem expected_successful_pairs_gt_half :
  expected_successful_pairs n > 1/2 :=
sorry

end all_pairs_successful_probability_expected_successful_pairs_gt_half_l2221_222153


namespace david_pushups_count_l2221_222174

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 59

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 44

/-- The difference between David's and Zachary's push-ups -/
def pushup_difference : ℕ := 19

/-- The difference between Zachary's and David's crunches -/
def crunch_difference : ℕ := 27

/-- David's push-ups -/
def david_pushups : ℕ := zachary_pushups + pushup_difference

theorem david_pushups_count : david_pushups = 78 := by
  sorry

end david_pushups_count_l2221_222174


namespace arctan_sum_equals_pi_fourth_l2221_222119

theorem arctan_sum_equals_pi_fourth (m : ℕ+) : 
  (Real.arctan (1/7) + Real.arctan (1/8) + Real.arctan (1/9) + Real.arctan (1/m.val : ℝ) = π/4) → m = 133 := by
  sorry

end arctan_sum_equals_pi_fourth_l2221_222119


namespace college_strength_l2221_222116

theorem college_strength (cricket_players basketball_players both : ℕ) 
  (h1 : cricket_players = 500)
  (h2 : basketball_players = 600)
  (h3 : both = 220) :
  cricket_players + basketball_players - both = 880 :=
by sorry

end college_strength_l2221_222116


namespace reservoir_duration_l2221_222177

theorem reservoir_duration (x y z : ℝ) 
  (h1 : 40 * (y - x) = z)
  (h2 : 40 * (1.1 * y - 1.2 * x) = z)
  : z / (y - 1.2 * x) = 50 := by
  sorry

end reservoir_duration_l2221_222177


namespace absolute_value_inequality_l2221_222110

theorem absolute_value_inequality (x a : ℝ) (ha : a > 0) :
  (|x - 3| + |x - 4| + |x - 5| < a) ↔ (a > 4) := by
  sorry

end absolute_value_inequality_l2221_222110


namespace total_cost_after_discounts_l2221_222181

/-- Calculate the total cost of items after applying discounts --/
theorem total_cost_after_discounts :
  let board_game_cost : ℚ := 2
  let action_figure_cost : ℚ := 7
  let action_figure_count : ℕ := 4
  let puzzle_cost : ℚ := 6
  let deck_cost : ℚ := 3.5
  let toy_car_cost : ℚ := 4
  let toy_car_count : ℕ := 2
  let action_figure_discount : ℚ := 0.15
  let puzzle_toy_car_discount : ℚ := 0.10
  let deck_discount : ℚ := 0.05

  let total_cost : ℚ := 
    board_game_cost +
    (action_figure_cost * action_figure_count) * (1 - action_figure_discount) +
    puzzle_cost * (1 - puzzle_toy_car_discount) +
    deck_cost * (1 - deck_discount) +
    (toy_car_cost * toy_car_count) * (1 - puzzle_toy_car_discount)

  total_cost = 41.73 := by
  sorry

end total_cost_after_discounts_l2221_222181


namespace sam_watermelons_l2221_222168

def total_watermelons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

theorem sam_watermelons : 
  let initial := 4
  let additional := 3
  total_watermelons initial additional = 7 := by
  sorry

end sam_watermelons_l2221_222168


namespace combined_collection_size_l2221_222135

/-- The number of books in Tim's collection -/
def tim_books : ℕ := 44

/-- The number of books in Sam's collection -/
def sam_books : ℕ := 52

/-- The number of books in Alex's collection -/
def alex_books : ℕ := 65

/-- The number of books in Katie's collection -/
def katie_books : ℕ := 37

/-- The total number of books in the combined collections -/
def total_books : ℕ := tim_books + sam_books + alex_books + katie_books

theorem combined_collection_size : total_books = 198 := by
  sorry

end combined_collection_size_l2221_222135


namespace room_length_is_19_l2221_222193

/-- Represents the dimensions of a rectangular room with a surrounding veranda. -/
structure RoomWithVeranda where
  roomLength : ℝ
  roomWidth : ℝ
  verandaWidth : ℝ

/-- Calculates the area of the veranda given the room dimensions. -/
def verandaArea (room : RoomWithVeranda) : ℝ :=
  (room.roomLength + 2 * room.verandaWidth) * (room.roomWidth + 2 * room.verandaWidth) -
  room.roomLength * room.roomWidth

/-- Theorem: The length of the room is 19 meters given the specified conditions. -/
theorem room_length_is_19 (room : RoomWithVeranda)
  (h1 : room.roomWidth = 12)
  (h2 : room.verandaWidth = 2)
  (h3 : verandaArea room = 140) :
  room.roomLength = 19 := by
  sorry

end room_length_is_19_l2221_222193


namespace bing_position_guimao_in_cycle_l2221_222165

-- Define the cyclic arrangement
def heavenly_stems := 10
def earthly_branches := 12
def cycle_length := 60

-- Define the position of 丙 (bǐng)
def bing_first_appearance := 3

-- Define the function for the nth appearance of 丙 (bǐng)
def bing_column (n : ℕ) : ℕ := 10 * n - 7

-- Define the position of 癸卯 (guǐ mǎo) in the cycle
def guimao_position := 40

-- Theorem for the position of 丙 (bǐng)
theorem bing_position (n : ℕ) : 
  bing_column n ≡ bing_first_appearance [MOD cycle_length] :=
sorry

-- Theorem for the position of 癸卯 (guǐ mǎo)
theorem guimao_in_cycle : 
  guimao_position > 0 ∧ guimao_position ≤ cycle_length :=
sorry

end bing_position_guimao_in_cycle_l2221_222165


namespace remainder_67_power_67_plus_67_mod_68_l2221_222114

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end remainder_67_power_67_plus_67_mod_68_l2221_222114


namespace cuboid_surface_area_l2221_222199

/-- Given two cubes with side length b joined to form a cuboid, 
    the surface area of the resulting cuboid is 10b^2 -/
theorem cuboid_surface_area (b : ℝ) (h : b > 0) : 
  2 * (2*b*b + b*b + b*(2*b)) = 10 * b^2 := by
  sorry

end cuboid_surface_area_l2221_222199


namespace intersection_M_N_l2221_222196

def M : Set ℝ := { x | -3 < x ∧ x < 1 }
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by
  sorry

end intersection_M_N_l2221_222196


namespace quadratic_real_roots_l2221_222123

theorem quadratic_real_roots (p1 p2 q1 q2 : ℝ) 
  (h : p1 * p2 = 2 * (q1 + q2)) : 
  ∃ x : ℝ, (x^2 + p1*x + q1 = 0) ∨ (x^2 + p2*x + q2 = 0) :=
by sorry

end quadratic_real_roots_l2221_222123


namespace border_collie_grooming_time_l2221_222159

/-- Represents the time in minutes Karen takes to groom different dog breeds -/
structure GroomingTimes where
  rottweiler : ℕ
  borderCollie : ℕ
  chihuahua : ℕ

/-- Represents the number of dogs Karen grooms in a specific session -/
structure DogCounts where
  rottweilers : ℕ
  borderCollies : ℕ
  chihuahuas : ℕ

/-- Given Karen's grooming times and dog counts, calculates the total grooming time -/
def totalGroomingTime (times : GroomingTimes) (counts : DogCounts) : ℕ :=
  times.rottweiler * counts.rottweilers +
  times.borderCollie * counts.borderCollies +
  times.chihuahua * counts.chihuahuas

/-- Theorem stating that Karen takes 10 minutes to groom a border collie -/
theorem border_collie_grooming_time :
  ∀ (times : GroomingTimes) (counts : DogCounts),
    times.rottweiler = 20 →
    times.chihuahua = 45 →
    counts.rottweilers = 6 →
    counts.borderCollies = 9 →
    counts.chihuahuas = 1 →
    totalGroomingTime times counts = 255 →
    times.borderCollie = 10 := by
  sorry

end border_collie_grooming_time_l2221_222159


namespace parallel_vectors_x_value_l2221_222106

/-- Given two 2D vectors a and b, where a = (2, -1) and b = (-4, x),
    if a and b are parallel, then x = 2. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![-4, x]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  x = 2 := by
sorry

end parallel_vectors_x_value_l2221_222106


namespace existence_of_special_sequence_l2221_222130

theorem existence_of_special_sequence :
  ∃ (a : Fin 100 → ℕ),
    (∀ i j, i < j → a i < a j) ∧
    (∀ i : Fin 98, Nat.gcd (a i) (a (i + 1)) > Nat.gcd (a (i + 1)) (a (i + 2))) :=
by sorry

end existence_of_special_sequence_l2221_222130


namespace triangle_side_length_l2221_222118

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end triangle_side_length_l2221_222118


namespace sum_of_roots_l2221_222101

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 12*c^2 + 15*c - 36 = 0) 
  (hd : 6*d^3 - 36*d^2 - 150*d + 1350 = 0) : 
  c + d = 7 := by
  sorry

end sum_of_roots_l2221_222101


namespace power_of_power_l2221_222146

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l2221_222146


namespace integral_equation_solution_l2221_222161

theorem integral_equation_solution (k : ℝ) : 
  (∫ x in (0:ℝ)..2, (3 * x^2 + k)) = 16 → k = 4 := by
sorry

end integral_equation_solution_l2221_222161


namespace no_real_roots_for_geometric_sequence_quadratic_l2221_222173

/-- If a, b, and c form a geometric sequence, then ax^2 + bx + c = 0 has no real roots -/
theorem no_real_roots_for_geometric_sequence_quadratic 
  (a b c : ℝ) (h_geom : b^2 = a*c ∧ a*c > 0) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
sorry

end no_real_roots_for_geometric_sequence_quadratic_l2221_222173


namespace complex_modulus_squared_l2221_222122

theorem complex_modulus_squared (z : ℂ) (h1 : z + Complex.abs z = 6 + 2*I) 
  (h2 : z.re ≥ 0) : Complex.abs z ^ 2 = 100 / 9 := by
  sorry

end complex_modulus_squared_l2221_222122


namespace police_force_female_officers_l2221_222158

theorem police_force_female_officers :
  ∀ (total_female : ℕ) (first_shift_total : ℕ) (first_shift_female_percent : ℚ),
    first_shift_total = 204 →
    first_shift_female_percent = 17 / 100 →
    (first_shift_total / 2 : ℚ) = first_shift_female_percent * total_female →
    total_female = 600 := by
  sorry

end police_force_female_officers_l2221_222158


namespace hyperbola_equation_l2221_222151

/-- The equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- General form of hyperbola
  (∃ x y : ℝ, y^2 = -4*x) →  -- Parabola equation
  ((-1 : ℝ) = a) →  -- Real axis endpoint coincides with parabola focus
  ((a + b) / a = 2) →  -- Eccentricity is 2
  (∀ x y : ℝ, x^2 - y^2 / 3 = 1) :=  -- Resulting hyperbola equation
by sorry

end hyperbola_equation_l2221_222151


namespace all_analogies_correct_correct_analogies_count_l2221_222108

-- Define the structure for a hyperbola
structure Hyperbola where
  focal_length : ℝ
  real_axis_length : ℝ
  eccentricity : ℝ

-- Define the structure for an ellipse
structure Ellipse where
  focal_length : ℝ
  major_axis_length : ℝ
  eccentricity : ℝ

-- Define the structure for an arithmetic sequence
structure ArithmeticSequence where
  first_term : ℝ
  second_term : ℝ
  third_term : ℝ

-- Define the structure for a geometric sequence
structure GeometricSequence where
  first_term : ℝ
  second_term : ℝ
  third_term : ℝ

-- Define the structure for an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  area : ℝ

-- Define the structure for a regular tetrahedron
structure RegularTetrahedron where
  edge_length : ℝ
  volume : ℝ

def analogy1_correct (h : Hyperbola) (e : Ellipse) : Prop :=
  (h.focal_length = 2 * h.real_axis_length → h.eccentricity = 2) →
  (e.focal_length = 1/2 * e.major_axis_length → e.eccentricity = 1/2)

def analogy2_correct (a : ArithmeticSequence) (g : GeometricSequence) : Prop :=
  (a.first_term + a.second_term + a.third_term = 1 → a.second_term = 1/3) →
  (g.first_term * g.second_term * g.third_term = 1 → g.second_term = 1)

def analogy3_correct (t1 t2 : EquilateralTriangle) (tet1 tet2 : RegularTetrahedron) : Prop :=
  (t2.side_length = 2 * t1.side_length → t2.area = 4 * t1.area) →
  (tet2.edge_length = 2 * tet1.edge_length → tet2.volume = 8 * tet1.volume)

theorem all_analogies_correct 
  (h : Hyperbola) (e : Ellipse) 
  (a : ArithmeticSequence) (g : GeometricSequence)
  (t1 t2 : EquilateralTriangle) (tet1 tet2 : RegularTetrahedron) : 
  analogy1_correct h e ∧ analogy2_correct a g ∧ analogy3_correct t1 t2 tet1 tet2 := by
  sorry

theorem correct_analogies_count : ∃ (n : ℕ), n = 3 ∧ 
  ∀ (h : Hyperbola) (e : Ellipse) 
     (a : ArithmeticSequence) (g : GeometricSequence)
     (t1 t2 : EquilateralTriangle) (tet1 tet2 : RegularTetrahedron),
  (analogy1_correct h e → n ≥ 1) ∧
  (analogy2_correct a g → n ≥ 2) ∧
  (analogy3_correct t1 t2 tet1 tet2 → n = 3) := by
  sorry

end all_analogies_correct_correct_analogies_count_l2221_222108


namespace lulu_piggy_bank_l2221_222175

theorem lulu_piggy_bank (initial_amount : ℝ) : 
  (4/5 * (1/2 * (initial_amount - 5))) = 24 → initial_amount = 65 := by
  sorry

end lulu_piggy_bank_l2221_222175


namespace fifth_power_sum_l2221_222157

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^5 + b * y^5 = -16 := by
sorry

end fifth_power_sum_l2221_222157


namespace composite_polynomial_l2221_222129

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 6*n^2 + 12*n + 16 = a * b :=
by sorry

end composite_polynomial_l2221_222129


namespace set_equation_solution_l2221_222191

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def B (a b : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + b = 0}

-- State the theorem
theorem set_equation_solution (a b : ℝ) : 
  B a b ≠ ∅ ∧ B a b ⊆ A → 
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) := by
  sorry

end set_equation_solution_l2221_222191


namespace sandwich_cost_is_181_l2221_222132

/-- The cost in cents for Joe to make a deluxe ham and cheese sandwich -/
def sandwich_cost : ℕ :=
  let bread_cost : ℕ := 15 -- Cost of one slice of bread in cents
  let ham_cost : ℕ := 25 -- Cost of one slice of ham in cents
  let cheese_cost : ℕ := 35 -- Cost of one slice of cheese in cents
  let mayo_cost : ℕ := 10 -- Cost of one tablespoon of mayonnaise in cents
  let lettuce_cost : ℕ := 5 -- Cost of one lettuce leaf in cents
  let tomato_cost : ℕ := 8 -- Cost of one tomato slice in cents
  
  let bread_slices : ℕ := 2 -- Number of bread slices used
  let ham_slices : ℕ := 2 -- Number of ham slices used
  let cheese_slices : ℕ := 2 -- Number of cheese slices used
  let mayo_tbsp : ℕ := 1 -- Number of tablespoons of mayonnaise used
  let lettuce_leaves : ℕ := 1 -- Number of lettuce leaves used
  let tomato_slices : ℕ := 2 -- Number of tomato slices used
  
  bread_cost * bread_slices +
  ham_cost * ham_slices +
  cheese_cost * cheese_slices +
  mayo_cost * mayo_tbsp +
  lettuce_cost * lettuce_leaves +
  tomato_cost * tomato_slices

theorem sandwich_cost_is_181 : sandwich_cost = 181 := by
  sorry

end sandwich_cost_is_181_l2221_222132


namespace log_expression_simplification_l2221_222169

theorem log_expression_simplification :
  (1/2) * Real.log (32/49) - (4/3) * Real.log (Real.sqrt 8) + Real.log (Real.sqrt 245) = 1/2 := by
  sorry

end log_expression_simplification_l2221_222169


namespace square_area_11cm_l2221_222148

/-- The area of a square with side length 11 cm is 121 cm². -/
theorem square_area_11cm (side_length : ℝ) (h : side_length = 11) :
  side_length * side_length = 121 := by
  sorry

end square_area_11cm_l2221_222148


namespace roast_cost_is_17_l2221_222167

/-- Calculates the cost of a roast given initial money, vegetable cost, and remaining money --/
def roast_cost (initial_money : ℤ) (vegetable_cost : ℤ) (remaining_money : ℤ) : ℤ :=
  initial_money - vegetable_cost - remaining_money

/-- Proves that the roast cost €17 given the problem conditions --/
theorem roast_cost_is_17 :
  roast_cost 100 11 72 = 17 := by
  sorry

end roast_cost_is_17_l2221_222167


namespace intersection_point_k_value_l2221_222103

theorem intersection_point_k_value (x y k : ℝ) : 
  x = -6.3 →
  3 * x + y = k →
  -0.75 * x + y = 25 →
  k = 1.375 := by
sorry

end intersection_point_k_value_l2221_222103


namespace equation_classification_l2221_222136

def equation (m : ℝ) (x : ℝ) : ℝ := (m^2 - 1) * x^2 + (m + 1) * x + (m - 2)

theorem equation_classification (m : ℝ) :
  (∀ x, equation m x = 0 → (m^2 - 1 ≠ 0 ↔ m ≠ 1 ∧ m ≠ -1)) ∧
  (∀ x, equation m x = 0 → (m^2 - 1 = 0 ∧ m + 1 ≠ 0 ↔ m = 1)) :=
by sorry

end equation_classification_l2221_222136


namespace club_size_after_four_years_l2221_222127

def club_size (initial_members : ℕ) (years : ℕ) : ℕ :=
  let active_members := initial_members - 3
  let growth_factor := 4
  (growth_factor ^ years) * active_members + 3

theorem club_size_after_four_years :
  club_size 21 4 = 4611 := by sorry

end club_size_after_four_years_l2221_222127


namespace max_value_sqrt_sum_l2221_222113

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end max_value_sqrt_sum_l2221_222113


namespace subtract_negative_three_l2221_222145

theorem subtract_negative_three : 0 - (-3) = 3 := by
  sorry

end subtract_negative_three_l2221_222145


namespace somu_age_problem_l2221_222128

/-- Proves that Somu was one-fifth of his father's age 8 years ago -/
theorem somu_age_problem (somu_age father_age years_ago : ℕ) : 
  somu_age = 16 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 8 := by
  sorry


end somu_age_problem_l2221_222128


namespace weight_sum_abby_damon_l2221_222182

/-- Given the weights of four people in pairs, prove that the sum of the weights of the first and fourth person is 300 pounds. -/
theorem weight_sum_abby_damon (a b c d : ℕ) : 
  a + b = 270 → 
  b + c = 250 → 
  c + d = 280 → 
  a + c = 300 → 
  a + d = 300 := by
sorry

end weight_sum_abby_damon_l2221_222182


namespace triangle_tangency_points_l2221_222139

theorem triangle_tangency_points (a b c : ℝ) : 
  (0 < a ∧ 0 < b ∧ 0 < c) →  -- positive sides
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- triangle inequality
  (∃ (M R : ℝ), 
    0 < M ∧ M < c ∧ 
    0 < R ∧ R < c ∧
    M ≠ R ∧
    M / c = 1 / 3 ∧ 
    (c - R) / c = 1 / 3 ∧
    (R - M) / c = 1 / 3) →  -- points divide c into three equal parts
  c = 3 * abs (a - b) ∧ 
  ((b < a ∧ a < 2 * b) ∨ (a < b ∧ b < 2 * a)) := by
sorry

end triangle_tangency_points_l2221_222139


namespace complex_subtraction_multiplication_l2221_222131

theorem complex_subtraction_multiplication (i : ℂ) : 
  (7 - 3*i) - 3*(2 - 5*i) = 1 + 12*i :=
by sorry

end complex_subtraction_multiplication_l2221_222131


namespace smallest_next_divisor_after_437_l2221_222121

theorem smallest_next_divisor_after_437 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : Even m) (h3 : m % 437 = 0) : 
  (∀ d : ℕ, d ∣ m → 437 < d → d ≥ 874) ∧ 874 ∣ m :=
sorry

end smallest_next_divisor_after_437_l2221_222121


namespace bakery_calculations_l2221_222142

-- Define the bakery's parameters
def cost_price : ℝ := 4
def selling_price : ℝ := 10
def clearance_price : ℝ := 2
def min_loaves : ℕ := 15
def max_loaves : ℕ := 30
def baked_loaves : ℕ := 21

-- Define the demand frequencies
def demand_freq : List (ℕ × ℕ) := [(15, 10), (18, 8), (21, 7), (24, 3), (27, 2)]

-- Calculate the probability of demand being at least 21 loaves
def prob_demand_ge_21 : ℚ := 2/5

-- Calculate the daily profit when demand is 15 loaves
def profit_demand_15 : ℝ := 78

-- Calculate the average daily profit over 30 days
def avg_daily_profit : ℝ := 103.6

theorem bakery_calculations :
  (prob_demand_ge_21 = 2/5) ∧
  (profit_demand_15 = 78) ∧
  (avg_daily_profit = 103.6) := by
  sorry

end bakery_calculations_l2221_222142


namespace integers_between_negative_two_and_three_l2221_222149

theorem integers_between_negative_two_and_three :
  {x : ℤ | x > -2 ∧ x ≤ 3} = {-1, 0, 1, 2, 3} := by sorry

end integers_between_negative_two_and_three_l2221_222149


namespace min_value_xy_l2221_222109

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) :
  x * y ≥ 64 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2/x₀ + 8/y₀ = 1 ∧ x₀ * y₀ = 64 :=
by sorry

end min_value_xy_l2221_222109


namespace water_missing_calculation_l2221_222197

/-- Calculates the amount of water missing from a tank's maximum capacity after a series of leaks and refilling. -/
def water_missing (initial_capacity : ℕ) (leak_rate1 leak_duration1 : ℕ) (leak_rate2 leak_duration2 : ℕ) (fill_rate fill_duration : ℕ) : ℕ :=
  let total_leak := leak_rate1 * leak_duration1 + leak_rate2 * leak_duration2
  let remaining_water := initial_capacity - total_leak
  let filled_water := fill_rate * fill_duration
  let final_water := remaining_water + filled_water
  initial_capacity - final_water

/-- Theorem stating that the amount of water missing from the tank's maximum capacity is 140,000 gallons. -/
theorem water_missing_calculation :
  water_missing 350000 32000 5 10000 10 40000 3 = 140000 := by
  sorry

end water_missing_calculation_l2221_222197


namespace chloe_earnings_l2221_222198

/-- Chloe's earnings over two weeks -/
theorem chloe_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 18 →
  hours_week2 = 26 →
  extra_earnings = 65.45 →
  ∃ (hourly_wage : ℚ),
    hourly_wage * (hours_week2 - hours_week1 : ℚ) = extra_earnings ∧
    hourly_wage * (hours_week1 + hours_week2 : ℚ) = 360 :=
by sorry

end chloe_earnings_l2221_222198


namespace hexagon_area_l2221_222180

/-- Right triangle with legs 3 and 4, hypotenuse 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  a_eq : a = 3
  b_eq : b = 4
  c_eq : c = 5

/-- Square with side length 3 -/
def square1_area : ℝ := 9

/-- Square with side length 4 -/
def square2_area : ℝ := 16

/-- Rectangle with sides 5 and 6 -/
def rectangle_area : ℝ := 30

/-- Area of the triangle formed by extending one side of the first square -/
def extended_triangle_area : ℝ := 4.5

/-- Theorem: The area of the hexagon DEFGHI is 52.5 -/
theorem hexagon_area (t : RightTriangle) : 
  square1_area + square2_area + rectangle_area + extended_triangle_area = 52.5 := by
  sorry

end hexagon_area_l2221_222180


namespace arithmetic_geometric_progression_l2221_222163

theorem arithmetic_geometric_progression (b c : ℝ) 
  (not_both_one : ¬(b = 1 ∧ c = 1))
  (arithmetic_prog : ∃ n : ℝ, b = 1 + n ∧ c = 1 + 2*n)
  (geometric_prog : c / 1 = b / c) :
  100 * (b - c) = 75 := by
sorry

end arithmetic_geometric_progression_l2221_222163


namespace complex_fraction_simplification_l2221_222134

theorem complex_fraction_simplification :
  (5 - 3*I) / (2 - 3*I) = -19/5 - 9/5*I :=
by sorry

end complex_fraction_simplification_l2221_222134


namespace helicopter_rental_theorem_l2221_222189

/-- Calculates the total cost of renting a helicopter given the daily rental hours, number of days, and hourly rate. -/
def helicopter_rental_cost (hours_per_day : ℕ) (days : ℕ) (rate_per_hour : ℕ) : ℕ :=
  hours_per_day * days * rate_per_hour

/-- Proves that renting a helicopter for 2 hours a day for 3 days at $75 per hour costs $450 in total. -/
theorem helicopter_rental_theorem : helicopter_rental_cost 2 3 75 = 450 := by
  sorry

end helicopter_rental_theorem_l2221_222189


namespace game_board_probability_l2221_222105

/-- Represents a triangle on the game board --/
structure GameTriangle :=
  (is_isosceles_right : Bool)
  (num_subdivisions : Nat)
  (num_shaded : Nat)

/-- Calculates the probability of landing in a shaded region --/
def probability_shaded (t : GameTriangle) : ℚ :=
  t.num_shaded / t.num_subdivisions

/-- The main theorem stating the probability for the specific game board configuration --/
theorem game_board_probability (t : GameTriangle) :
  t.is_isosceles_right = true →
  t.num_subdivisions = 6 →
  t.num_shaded = 2 →
  probability_shaded t = 1/3 := by
  sorry


end game_board_probability_l2221_222105


namespace degenerate_ellipse_max_y_coordinate_l2221_222155

theorem degenerate_ellipse_max_y_coordinate :
  ∀ x y : ℝ, (x^2 / 49 + (y - 3)^2 / 25 = 0) → y ≤ 3 :=
by sorry

end degenerate_ellipse_max_y_coordinate_l2221_222155


namespace lowest_fraction_job_l2221_222188

/-- Given three people who can individually complete a job in 4, 6, and 8 hours respectively,
    the lowest fraction of the job that can be done in 1 hour by 2 of the people working together is 7/24. -/
theorem lowest_fraction_job (person_a person_b person_c : ℝ) 
    (ha : person_a = 1 / 4) (hb : person_b = 1 / 6) (hc : person_c = 1 / 8) : 
    min (person_a + person_b) (min (person_a + person_c) (person_b + person_c)) = 7 / 24 := by
  sorry

end lowest_fraction_job_l2221_222188


namespace polynomial_multiplication_l2221_222115

theorem polynomial_multiplication (x : ℝ) :
  (3 * x^2 - 4 * x + 5) * (-2 * x^2 + 3 * x - 7) =
  -6 * x^4 + 17 * x^3 - 43 * x^2 + 43 * x - 35 := by
  sorry

end polynomial_multiplication_l2221_222115


namespace flight_cost_X_to_Y_l2221_222187

/-- The cost to fly between two cities given the distance and cost parameters. -/
def flight_cost (distance : ℝ) (cost_per_km : ℝ) (booking_fee : ℝ) : ℝ :=
  distance * cost_per_km + booking_fee

/-- Theorem stating that the flight cost from X to Y is $660. -/
theorem flight_cost_X_to_Y :
  flight_cost 4500 0.12 120 = 660 := by
  sorry

end flight_cost_X_to_Y_l2221_222187


namespace intersection_equality_implies_possible_a_l2221_222150

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a*x + 2 = 0}

-- Define the set of possible values for a
def possible_a : Set ℝ := {-1, 0, 2/3}

-- Theorem statement
theorem intersection_equality_implies_possible_a :
  ∀ a : ℝ, (M ∩ N a = N a) → a ∈ possible_a :=
by sorry

end intersection_equality_implies_possible_a_l2221_222150


namespace max_value_of_f_l2221_222126

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * x^2 + 2*a*x

theorem max_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  (∀ x ∈ Set.Icc 1 4, f a x ≥ -16/3) →
  (∃ x ∈ Set.Icc 1 4, f a x = -16/3) →
  (∃ x ∈ Set.Icc 1 4, f a x = 10/3) ∧
  (∀ x ∈ Set.Icc 1 4, f a x ≤ 10/3) :=
by sorry

end max_value_of_f_l2221_222126


namespace expression_evaluation_l2221_222172

theorem expression_evaluation : (4 * 5 * 6) * (1/4 + 1/5 - 1/10) = 42 := by
  sorry

end expression_evaluation_l2221_222172


namespace count_with_four_or_five_l2221_222140

/-- The number of integers from 1 to 343 (inclusive) in base 7 that do not contain 4 or 5 as a digit -/
def count_without_four_or_five : ℕ := 125

/-- The total number of integers from 1 to 343 in base 7 -/
def total_count : ℕ := 343

theorem count_with_four_or_five :
  total_count - count_without_four_or_five = 218 :=
sorry

end count_with_four_or_five_l2221_222140


namespace unique_solution_non_unique_solution_l2221_222185

-- Define the equation
def equation (x a b : ℝ) : Prop :=
  (x - a) / (x - 2) + (x - b) / (x - 3) = 2

-- Theorem for unique solution
theorem unique_solution (a b : ℝ) :
  (∃! x, equation x a b) ↔ (a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3) :=
sorry

-- Theorem for non-unique solution
theorem non_unique_solution (a b : ℝ) :
  (∃ x y, x ≠ y ∧ equation x a b ∧ equation y a b) ↔ (a = 2 ∧ b = 3) :=
sorry

end unique_solution_non_unique_solution_l2221_222185
