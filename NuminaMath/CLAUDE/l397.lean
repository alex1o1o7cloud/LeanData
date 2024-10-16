import Mathlib

namespace NUMINAMATH_CALUDE_composite_divisor_of_product_l397_39781

def product_up_to (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem composite_divisor_of_product (m : ℕ) :
  m > 1 →
  (m ∣ product_up_to m) ↔ (¬ Nat.Prime m ∧ m ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_composite_divisor_of_product_l397_39781


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_plus_constant_l397_39712

def repeat_digit (d : Nat) (n : Nat) : Nat :=
  d * (10^n - 1) / 9

def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_product_plus_constant : 
  sum_of_digits ((repeat_digit 9 47 * repeat_digit 4 47) + 100000) = 424 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_plus_constant_l397_39712


namespace NUMINAMATH_CALUDE_line_properties_l397_39705

/-- Given a line with equation 2x + y + 3 = 0, prove its slope and y-intercept -/
theorem line_properties :
  let line := {(x, y) : ℝ × ℝ | 2 * x + y + 3 = 0}
  ∃ (m b : ℝ), m = -2 ∧ b = -3 ∧ ∀ (x y : ℝ), (x, y) ∈ line ↔ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l397_39705


namespace NUMINAMATH_CALUDE_time_walking_away_l397_39718

-- Define the walking speed in miles per hour
def walking_speed : ℝ := 2

-- Define the total distance walked in miles
def total_distance : ℝ := 12

-- Define the theorem
theorem time_walking_away : 
  (total_distance / 2) / walking_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_walking_away_l397_39718


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l397_39770

def a : Fin 2 → ℝ := ![4, 4]
def b (m : ℝ) : Fin 2 → ℝ := ![5, m]
def c : Fin 2 → ℝ := ![1, 3]

theorem perpendicular_vectors (m : ℝ) :
  (∀ i : Fin 2, (a i - 2 * c i) * b m i = 0) ↔ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l397_39770


namespace NUMINAMATH_CALUDE_vector_equation_solution_l397_39783

def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

theorem vector_equation_solution (x y : ℝ) 
  (h : vector_sum (x, 1) (scalar_mult 2 (2, y)) = (5, -3)) : 
  x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l397_39783


namespace NUMINAMATH_CALUDE_john_used_16_bulbs_l397_39714

/-- The number of light bulbs John used -/
def bulbs_used : ℕ := sorry

/-- The initial number of light bulbs -/
def initial_bulbs : ℕ := 40

/-- The number of light bulbs John has left after giving some away -/
def remaining_bulbs : ℕ := 12

theorem john_used_16_bulbs : 
  bulbs_used = 16 ∧ 
  (initial_bulbs - bulbs_used) / 2 = remaining_bulbs :=
sorry

end NUMINAMATH_CALUDE_john_used_16_bulbs_l397_39714


namespace NUMINAMATH_CALUDE_second_to_fourth_l397_39711

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If A(a,b) is in the second quadrant, then B(b,a) is in the fourth quadrant -/
theorem second_to_fourth (a b : ℝ) :
  is_in_second_quadrant (Point.mk a b) →
  is_in_fourth_quadrant (Point.mk b a) := by
  sorry

end NUMINAMATH_CALUDE_second_to_fourth_l397_39711


namespace NUMINAMATH_CALUDE_remaining_value_proof_l397_39728

theorem remaining_value_proof (x : ℝ) (h : 0.36 * x = 2376) : 4500 - 0.7 * x = -120 := by
  sorry

end NUMINAMATH_CALUDE_remaining_value_proof_l397_39728


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l397_39729

/-- Represents the number of students in each grade --/
structure Students :=
  (ninth : ℕ)
  (seventh : ℕ)
  (fifth : ℕ)

/-- The ratio of 9th-graders to 7th-graders is 7:4 --/
def ratio_ninth_seventh (s : Students) : Prop :=
  7 * s.seventh = 4 * s.ninth

/-- The ratio of 7th-graders to 5th-graders is 6:5 --/
def ratio_seventh_fifth (s : Students) : Prop :=
  6 * s.fifth = 5 * s.seventh

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.ninth + s.seventh + s.fifth

/-- The theorem stating the smallest possible number of students --/
theorem smallest_number_of_students :
  ∃ (s : Students),
    ratio_ninth_seventh s ∧
    ratio_seventh_fifth s ∧
    (∀ (t : Students),
      ratio_ninth_seventh t ∧ ratio_seventh_fifth t →
      total_students s ≤ total_students t) ∧
    total_students s = 43 :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l397_39729


namespace NUMINAMATH_CALUDE_gcd_of_36_and_54_l397_39795

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_36_and_54_l397_39795


namespace NUMINAMATH_CALUDE_no_integer_solution_l397_39713

theorem no_integer_solution : ¬ ∃ (x y : ℤ), 21 * x - 35 * y = 59 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l397_39713


namespace NUMINAMATH_CALUDE_award_distribution_theorem_l397_39724

-- Define the number of awards and students
def num_awards : ℕ := 7
def num_students : ℕ := 4

-- Function to calculate the number of ways to distribute awards
def distribute_awards (awards : ℕ) (students : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem stating that the number of ways to distribute awards is 3920
theorem award_distribution_theorem :
  distribute_awards num_awards num_students = 3920 :=
by sorry

end NUMINAMATH_CALUDE_award_distribution_theorem_l397_39724


namespace NUMINAMATH_CALUDE_power_sum_l397_39791

theorem power_sum (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l397_39791


namespace NUMINAMATH_CALUDE_problem_solution_l397_39784

theorem problem_solution (a b : ℝ) 
  (h1 : a * b = 2 * (a + b) + 1) 
  (h2 : b - a = 4) : 
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l397_39784


namespace NUMINAMATH_CALUDE_square_product_closed_l397_39738

def P : Set ℕ := {n : ℕ | ∃ m : ℕ+, n = m ^ 2}

theorem square_product_closed (a b : ℕ) (ha : a ∈ P) (hb : b ∈ P) : 
  a * b ∈ P := by sorry

end NUMINAMATH_CALUDE_square_product_closed_l397_39738


namespace NUMINAMATH_CALUDE_max_flights_theorem_l397_39708

/-- Represents the number of cities with each airport type -/
structure AirportCounts where
  small : ℕ
  medium : ℕ
  big : ℕ

/-- Calculates the maximum number of flights given the airport counts -/
def max_flights (counts : AirportCounts) : ℕ :=
  counts.small * counts.medium +
  counts.small * counts.big +
  counts.medium * counts.big +
  25

/-- The theorem stating the maximum number of flights -/
theorem max_flights_theorem :
  ∃ (counts : AirportCounts),
    counts.small + counts.medium + counts.big = 28 ∧
    max_flights counts = 286 ∧
    ∀ (other_counts : AirportCounts),
      other_counts.small + other_counts.medium + other_counts.big = 28 →
      max_flights other_counts ≤ 286 :=
by
  sorry

#eval max_flights { small := 9, medium := 10, big := 9 }

end NUMINAMATH_CALUDE_max_flights_theorem_l397_39708


namespace NUMINAMATH_CALUDE_equation_solution_l397_39746

theorem equation_solution (x y : ℝ) 
  (hx0 : x ≠ 0) (hx3 : x ≠ 3) (hy0 : y ≠ 0) (hy4 : y ≠ 4)
  (h_eq : 3 / x + 2 / y = 5 / 6) :
  x = 18 * y / (5 * y - 12) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l397_39746


namespace NUMINAMATH_CALUDE_no_quadratic_composition_with_given_zeros_l397_39773

theorem no_quadratic_composition_with_given_zeros :
  ¬∃ (P Q : ℝ → ℝ),
    (∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) ∧
    (∃ d e f : ℝ, ∀ x, Q x = d * x^2 + e * x + f) ∧
    (∀ x, (P ∘ Q) x = 0 ↔ x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_composition_with_given_zeros_l397_39773


namespace NUMINAMATH_CALUDE_no_savings_on_group_purchase_l397_39719

def window_price : ℕ := 120

def free_windows (n : ℕ) : ℕ := (n / 10) * 2

def cost (n : ℕ) : ℕ := (n - free_windows n) * window_price

def alice_windows : ℕ := 9
def bob_windows : ℕ := 11
def celina_windows : ℕ := 10

theorem no_savings_on_group_purchase :
  cost (alice_windows + bob_windows + celina_windows) =
  cost alice_windows + cost bob_windows + cost celina_windows :=
by sorry

end NUMINAMATH_CALUDE_no_savings_on_group_purchase_l397_39719


namespace NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l397_39710

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) : 
  n = 2018^2 + 2^2018 → (n^2 + 2^n) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l397_39710


namespace NUMINAMATH_CALUDE_cookies_eaten_l397_39722

theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) 
  (h1 : initial = 93)
  (h2 : remaining = 78)
  (h3 : initial = remaining + eaten) :
  eaten = 15 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l397_39722


namespace NUMINAMATH_CALUDE_quadratic_single_solution_l397_39737

theorem quadratic_single_solution (a : ℝ) (ha : a ≠ 0) :
  (∃! x : ℝ, a * x^2 + 20 * x + 7 = 0) → 
  (∀ x : ℝ, a * x^2 + 20 * x + 7 = 0 → x = -7/10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_single_solution_l397_39737


namespace NUMINAMATH_CALUDE_bella_age_is_five_l397_39723

/-- Bella's age in years -/
def bella_age : ℕ := sorry

/-- Bella's brother's age in years -/
def brother_age : ℕ := sorry

/-- Theorem stating Bella's age given the conditions -/
theorem bella_age_is_five :
  (brother_age = bella_age + 9) →  -- Brother is 9 years older
  (bella_age + brother_age = 19) →  -- Ages add up to 19
  bella_age = 5 := by sorry

end NUMINAMATH_CALUDE_bella_age_is_five_l397_39723


namespace NUMINAMATH_CALUDE_cricket_players_count_l397_39742

theorem cricket_players_count (total players : ℕ) (hockey football softball : ℕ) :
  total = 55 →
  hockey = 12 →
  football = 13 →
  softball = 15 →
  players = total - (hockey + football + softball) →
  players = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_players_count_l397_39742


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l397_39785

theorem square_difference_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_eq : x - y = 8) : 
  x^2 - y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l397_39785


namespace NUMINAMATH_CALUDE_jonah_fish_exchange_l397_39715

/-- The number of new fish Jonah received in exchange -/
def exchange_fish (initial : ℕ) (added : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial + added - eaten)

/-- Theorem stating the number of new fish Jonah received -/
theorem jonah_fish_exchange :
  exchange_fish 14 2 6 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_jonah_fish_exchange_l397_39715


namespace NUMINAMATH_CALUDE_expected_ones_is_half_l397_39739

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1/6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ := 
  0 * (prob_not_one^num_dice) +
  1 * (num_dice.choose 1 * prob_one * prob_not_one^2) +
  2 * (num_dice.choose 2 * prob_one^2 * prob_not_one) +
  3 * prob_one^num_dice

theorem expected_ones_is_half : expected_ones = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_ones_is_half_l397_39739


namespace NUMINAMATH_CALUDE_arccos_one_half_l397_39755

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l397_39755


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l397_39748

theorem at_least_one_not_less_than_two (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l397_39748


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l397_39753

/-- Given a curve C with equation 4x^2 + 9y^2 = 36, 
    the maximum value of 3x + 4y for any point (x,y) on C is √145. -/
theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = Real.sqrt 145 ∧ 
  (∀ (x y : ℝ), 4 * x^2 + 9 * y^2 = 36 → 3 * x + 4 * y ≤ M) ∧
  (∃ (x y : ℝ), 4 * x^2 + 9 * y^2 = 36 ∧ 3 * x + 4 * y = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l397_39753


namespace NUMINAMATH_CALUDE_f_positive_implies_a_bounded_l397_39735

/-- The function f(x) defined as x^2 - ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 2

/-- Theorem stating that if f(x) > 0 for all x > 2, then a ≤ 3 -/
theorem f_positive_implies_a_bounded (a : ℝ) : 
  (∀ x > 2, f a x > 0) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_implies_a_bounded_l397_39735


namespace NUMINAMATH_CALUDE_trapezoid_sides_l397_39751

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithCircle where
  r : ℝ  -- radius of the inscribed circle
  a : ℝ  -- shorter base
  b : ℝ  -- longer base
  c : ℝ  -- left side
  d : ℝ  -- right side (hypotenuse)
  h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ r > 0  -- all lengths are positive
  ha : a = 4*r/3  -- shorter base condition
  hsum : a + b = c + d  -- sum of bases equals sum of non-parallel sides
  hright : c^2 + a^2 = d^2  -- right angle condition

/-- The sides of the trapezoid are 2r, 4r/3, 10r/3, and 4r -/
theorem trapezoid_sides (t : RightTrapezoidWithCircle) :
  t.c = 2*t.r ∧ t.a = 4*t.r/3 ∧ t.b = 10*t.r/3 ∧ t.d = 4*t.r :=
sorry

end NUMINAMATH_CALUDE_trapezoid_sides_l397_39751


namespace NUMINAMATH_CALUDE_distance_ratio_bound_l397_39780

/-- Manhattan distance between two points -/
def manhattan_distance (p q : ℝ × ℝ) : ℝ :=
  |p.1 - q.1| + |p.2 - q.2|

/-- The theorem to be proved -/
theorem distance_ratio_bound (points : Finset (ℝ × ℝ)) (h : points.card = 2023) :
  let distances := {d | ∃ (p q : ℝ × ℝ), p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ d = manhattan_distance p q}
  (⨆ d ∈ distances, d) / (⨅ d ∈ distances, d) ≥ 44 :=
sorry

end NUMINAMATH_CALUDE_distance_ratio_bound_l397_39780


namespace NUMINAMATH_CALUDE_second_shot_probability_l397_39767

theorem second_shot_probability 
  (p_first : ℝ) 
  (p_consecutive : ℝ) 
  (h1 : p_first = 0.75) 
  (h2 : p_consecutive = 0.6) : 
  p_consecutive / p_first = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_second_shot_probability_l397_39767


namespace NUMINAMATH_CALUDE_difference_of_percentages_l397_39736

theorem difference_of_percentages : 
  (75 / 100 * 480) - (3 / 5 * (20 / 100 * 2500)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_percentages_l397_39736


namespace NUMINAMATH_CALUDE_max_students_distribution_l397_39798

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1008) (h2 : pencils = 928) :
  (Nat.gcd pens pencils : ℕ) = 16 := by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l397_39798


namespace NUMINAMATH_CALUDE_cost_price_calculation_l397_39749

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 240)
  (h2 : profit_percentage = 0.25) : 
  ∃ (cost_price : ℝ), cost_price = 192 ∧ selling_price = cost_price * (1 + profit_percentage) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l397_39749


namespace NUMINAMATH_CALUDE_work_completion_smaller_group_l397_39759

/-- Given that 22 men complete a work in 55 days, and another group completes
    the same work in 121 days, prove that the number of men in the second group is 10. -/
theorem work_completion_smaller_group : 
  ∀ (work : ℕ) (group1_size group1_days group2_days : ℕ),
    group1_size = 22 →
    group1_days = 55 →
    group2_days = 121 →
    group1_size * group1_days = work →
    ∃ (group2_size : ℕ), 
      group2_size * group2_days = work ∧
      group2_size = 10 :=
by
  sorry

#check work_completion_smaller_group

end NUMINAMATH_CALUDE_work_completion_smaller_group_l397_39759


namespace NUMINAMATH_CALUDE_polynomial_factorization_l397_39741

theorem polynomial_factorization (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = 
  (a - b) * (b - c) * (c - a) * ((a + b + c)^3 - 3*a*b*c) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l397_39741


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l397_39709

def M : Set ℤ := {0}
def N : Set ℤ := {x | -1 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l397_39709


namespace NUMINAMATH_CALUDE_bridge_length_l397_39727

/-- Given a train crossing a bridge, this theorem calculates the length of the bridge. -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 170)
  (h2 : train_speed = 45 * 1000 / 3600)  -- Convert km/hr to m/s
  (h3 : crossing_time = 30) :
  train_speed * crossing_time - train_length = 205 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l397_39727


namespace NUMINAMATH_CALUDE_cans_display_rows_l397_39772

def triangular_display (n : ℕ) : ℕ := (3 * n * (n + 1)) / 2

theorem cans_display_rows :
  ∃ (n : ℕ), triangular_display n = 225 ∧ n = 11 := by
sorry

end NUMINAMATH_CALUDE_cans_display_rows_l397_39772


namespace NUMINAMATH_CALUDE_complex_fraction_power_l397_39790

theorem complex_fraction_power (i : ℂ) : i^2 = -1 → ((1 + i) / (1 - i))^2017 = i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l397_39790


namespace NUMINAMATH_CALUDE_bicycle_helmet_cost_l397_39702

theorem bicycle_helmet_cost (helmet_cost bicycle_cost total_cost : ℕ) : 
  helmet_cost = 40 →
  bicycle_cost = 5 * helmet_cost →
  total_cost = bicycle_cost + helmet_cost →
  total_cost = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_helmet_cost_l397_39702


namespace NUMINAMATH_CALUDE_candy_bar_cost_l397_39762

/-- Proves that the cost of each candy bar is $2, given the total spent and number of candy bars. -/
theorem candy_bar_cost (total_spent : ℚ) (num_candy_bars : ℕ) (h1 : total_spent = 4) (h2 : num_candy_bars = 2) :
  total_spent / num_candy_bars = 2 := by
  sorry

#check candy_bar_cost

end NUMINAMATH_CALUDE_candy_bar_cost_l397_39762


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l397_39747

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * x) := by sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l397_39747


namespace NUMINAMATH_CALUDE_problem_solution_l397_39745

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem problem_solution (a : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence (fun n => a (n + 1) - a n) →
  (fun n => a (n + 1) - a n) 1 = 2 →
  (∀ n : ℕ, (fun n => a (n + 1) - a n) (n + 1) - (fun n => a (n + 1) - a n) n = 2) →
  a 1 = 1 →
  43 < a m →
  a m < 73 →
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l397_39745


namespace NUMINAMATH_CALUDE_common_roots_of_polynomials_l397_39725

theorem common_roots_of_polynomials :
  let f (x : ℝ) := x^4 + 2*x^3 - x^2 - 2*x - 3
  let g (x : ℝ) := x^4 + 3*x^3 + x^2 - 4*x - 6
  let r₁ := (-1 + Real.sqrt 13) / 2
  let r₂ := (-1 - Real.sqrt 13) / 2
  (f r₁ = 0 ∧ f r₂ = 0) ∧ (g r₁ = 0 ∧ g r₂ = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_common_roots_of_polynomials_l397_39725


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_180_and_n_l397_39734

theorem greatest_common_divisor_of_180_and_n (n : ℕ) : 
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < d₂ ∧ d₂ < d₃ ∧ 
   {d : ℕ | d ∣ 180 ∧ d ∣ n} = {d₁, d₂, d₃}) →
  (Nat.gcd 180 n = 9) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_180_and_n_l397_39734


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l397_39758

theorem nested_fraction_evaluation : 
  2 + 3 / (4 + 5 / (6 + 7/8)) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l397_39758


namespace NUMINAMATH_CALUDE_polynomial_factorization_l397_39704

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 3) * (x^2 + 7*x + 10) + (x^2 + 7*x + 10) =
  (x^2 + 7*x + 20) * (x^2 + 7*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l397_39704


namespace NUMINAMATH_CALUDE_square_area_error_l397_39756

theorem square_area_error (edge : ℝ) (edge_error : ℝ) (area_error : ℝ) : 
  edge_error = 0.02 → 
  area_error = (((1 + edge_error) * edge)^2 - edge^2) / edge^2 * 100 → 
  area_error = 4.04 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l397_39756


namespace NUMINAMATH_CALUDE_sum_of_squares_of_G_digits_l397_39768

/-- Represents a fraction m/n -/
structure Fraction where
  m : ℕ+
  n : ℕ+
  m_lt_n : m < n
  lowest_terms : Nat.gcd m n = 1
  no_square_divisor : ∀ k > 1, ¬(k * k ∣ n)
  repeating_length_6 : ∃ k : ℕ+, m * 10^6 = k * n + m

/-- Count of valid fractions -/
def F : ℕ := 1109700

/-- Number of digits in F -/
def p : ℕ := 7

/-- G is defined as F + p -/
def G : ℕ := F + p

/-- Sum of squares of digits of a natural number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem to prove -/
theorem sum_of_squares_of_G_digits :
  sum_of_squares_of_digits G = 181 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_G_digits_l397_39768


namespace NUMINAMATH_CALUDE_bronson_leaf_collection_l397_39777

theorem bronson_leaf_collection (thursday_leaves : ℕ) (yellow_leaves : ℕ) 
  (h1 : thursday_leaves = 12)
  (h2 : yellow_leaves = 15)
  (h3 : yellow_leaves = (3 / 5 : ℚ) * (thursday_leaves + friday_leaves)) :
  friday_leaves = 13 := by
  sorry

end NUMINAMATH_CALUDE_bronson_leaf_collection_l397_39777


namespace NUMINAMATH_CALUDE_pipe_filling_time_l397_39775

theorem pipe_filling_time (p q r t : ℝ) (hp : p = 6) (hr : r = 24) (ht : t = 3.4285714285714284)
  (h_total : 1/p + 1/q + 1/r = 1/t) : q = 8 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l397_39775


namespace NUMINAMATH_CALUDE_f_of_2_equals_3_l397_39700

/-- Given a function f(x) = x^2 - 2x + 3, prove that f(2) = 3 -/
theorem f_of_2_equals_3 : let f : ℝ → ℝ := fun x ↦ x^2 - 2*x + 3
  f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_3_l397_39700


namespace NUMINAMATH_CALUDE_max_gcd_of_sum_1089_l397_39750

theorem max_gcd_of_sum_1089 (c d : ℕ+) (h : c + d = 1089) :
  (∃ (x y : ℕ+), x + y = 1089 ∧ Nat.gcd x y = 363) ∧
  (∀ (a b : ℕ+), a + b = 1089 → Nat.gcd a b ≤ 363) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_of_sum_1089_l397_39750


namespace NUMINAMATH_CALUDE_intersection_M_N_l397_39731

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {x : ℤ | (x + 1) * (x - 2) < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l397_39731


namespace NUMINAMATH_CALUDE_major_selection_theorem_l397_39765

-- Define the number of majors
def total_majors : ℕ := 7

-- Define the number of majors to be selected
def selected_majors : ℕ := 3

-- Define a function to calculate the number of ways to select and order majors
def ways_to_select_and_order (total : ℕ) (select : ℕ) (excluded : ℕ) : ℕ :=
  (Nat.choose total select - Nat.choose (total - excluded) (select - excluded)) * Nat.factorial select

-- Theorem statement
theorem major_selection_theorem : 
  ways_to_select_and_order total_majors selected_majors 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_major_selection_theorem_l397_39765


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l397_39779

theorem complex_arithmetic_equality : 9 - (8 + 7) * 6 + 5^2 - (4 * 3) + 2 - 1 = -67 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l397_39779


namespace NUMINAMATH_CALUDE_intersection_chord_length_l397_39721

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → ℝ → Prop

/-- Calculates the chord length of intersection between a circle and a line -/
def chordLength (c : PolarCircle) (l : PolarLine) : ℝ := sorry

/-- Main theorem: If a circle ρ = 4cosθ is intersected by a line ρsin(θ - φ) = a 
    with a chord length of 2, then a = 0 or a = -2 -/
theorem intersection_chord_length 
  (c : PolarCircle) 
  (l : PolarLine) 
  (h1 : c.equation = λ ρ θ => ρ = 4 * Real.cos θ)
  (h2 : l.equation = λ ρ θ φ => ρ * Real.sin (θ - φ) = a)
  (h3 : chordLength c l = 2) :
  a = 0 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l397_39721


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l397_39778

theorem fraction_sum_integer (n : ℕ) (hn : n > 0) 
  (h_sum : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) : 
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l397_39778


namespace NUMINAMATH_CALUDE_percentage_of_x_l397_39787

theorem percentage_of_x (x y : ℝ) (P : ℝ) : 
  (P / 100) * x = (20 / 100) * y →
  x / y = 2 →
  P = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_x_l397_39787


namespace NUMINAMATH_CALUDE_fifth_student_guess_l397_39764

def jellybean_guesses : ℕ → ℕ
  | 1 => 100
  | 2 => 8 * jellybean_guesses 1
  | 3 => jellybean_guesses 2 - 200
  | 4 => (jellybean_guesses 1 + jellybean_guesses 2 + jellybean_guesses 3) / 3 + 25
  | 5 => jellybean_guesses 4 + jellybean_guesses 4 / 5
  | _ => 0

theorem fifth_student_guess :
  jellybean_guesses 5 = 630 := by sorry

end NUMINAMATH_CALUDE_fifth_student_guess_l397_39764


namespace NUMINAMATH_CALUDE_olivia_showroom_spending_l397_39794

theorem olivia_showroom_spending (initial_amount supermarket_spending final_amount : ℕ) : 
  initial_amount = 106 →
  supermarket_spending = 31 →
  final_amount = 26 →
  initial_amount - supermarket_spending - final_amount = 49 := by
sorry

end NUMINAMATH_CALUDE_olivia_showroom_spending_l397_39794


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l397_39716

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l397_39716


namespace NUMINAMATH_CALUDE_not_necessary_nor_sufficient_l397_39757

theorem not_necessary_nor_sufficient : ∃ (x y : ℝ), 
  ((x / y > 1 ∧ x ≤ y) ∨ (x / y ≤ 1 ∧ x > y)) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_nor_sufficient_l397_39757


namespace NUMINAMATH_CALUDE_systematic_sample_correct_l397_39730

/-- Given a total number of students, sample size, and first drawn number,
    returns the list of remaining numbers in the systematic sampling sequence. -/
def systematicSample (totalStudents : Nat) (sampleSize : Nat) (firstNumber : Nat) : List Nat :=
  let interval := totalStudents / sampleSize
  List.range (sampleSize - 1) |>.map (fun i => (firstNumber + (i + 1) * interval) % totalStudents)

/-- Theorem stating that for the given conditions, the systematic sampling
    produces the expected sequence of numbers. -/
theorem systematic_sample_correct :
  systematicSample 60 5 4 = [16, 28, 40, 52] := by
  sorry

#eval systematicSample 60 5 4

end NUMINAMATH_CALUDE_systematic_sample_correct_l397_39730


namespace NUMINAMATH_CALUDE_raisin_distribution_l397_39733

/-- Given 5 boxes of raisins with a total of 437 raisins, where one box has 72 raisins,
    another has 74 raisins, and the remaining three boxes have an equal number of raisins,
    prove that each of the three equal boxes contains 97 raisins. -/
theorem raisin_distribution (total_raisins : ℕ) (total_boxes : ℕ) 
  (box1_raisins : ℕ) (box2_raisins : ℕ) (h1 : total_raisins = 437) 
  (h2 : total_boxes = 5) (h3 : box1_raisins = 72) (h4 : box2_raisins = 74) :
  ∃ (equal_box_raisins : ℕ), 
    equal_box_raisins * 3 + box1_raisins + box2_raisins = total_raisins ∧ 
    equal_box_raisins = 97 := by
  sorry

end NUMINAMATH_CALUDE_raisin_distribution_l397_39733


namespace NUMINAMATH_CALUDE_prob_red_pen_is_two_fifths_l397_39782

/-- The number of colored pens -/
def total_pens : ℕ := 5

/-- The number of pens to be selected -/
def selected_pens : ℕ := 2

/-- The number of ways to select 2 pens out of 5 -/
def total_selections : ℕ := Nat.choose total_pens selected_pens

/-- The number of ways to select a red pen and another different color -/
def red_selections : ℕ := total_pens - 1

/-- The probability of selecting a red pen when choosing 2 different colored pens out of 5 -/
def prob_red_pen : ℚ := red_selections / total_selections

theorem prob_red_pen_is_two_fifths : prob_red_pen = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_pen_is_two_fifths_l397_39782


namespace NUMINAMATH_CALUDE_mango_dishes_l397_39701

theorem mango_dishes (total_dishes : ℕ) (mango_salsa_dishes : ℕ) (mango_jelly_dishes : ℕ) (oliver_willing_dishes : ℕ) :
  total_dishes = 36 →
  mango_salsa_dishes = 3 →
  mango_jelly_dishes = 1 →
  oliver_willing_dishes = 28 →
  let fresh_mango_dishes : ℕ := total_dishes / 6
  let pickable_fresh_mango_dishes : ℕ := total_dishes - (oliver_willing_dishes + mango_salsa_dishes + mango_jelly_dishes)
  pickable_fresh_mango_dishes = 4 := by
sorry

end NUMINAMATH_CALUDE_mango_dishes_l397_39701


namespace NUMINAMATH_CALUDE_prob_odd_divisor_18_factorial_l397_39769

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of positive integer divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- The number of odd positive integer divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := sorry

/-- The probability of a randomly chosen positive integer divisor of n being odd -/
def probOddDivisor (n : ℕ) : ℚ := (numOddDivisors n : ℚ) / (numDivisors n : ℚ)

theorem prob_odd_divisor_18_factorial :
  probOddDivisor (factorial 18) = 1 / 17 := by sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_18_factorial_l397_39769


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l397_39766

def dividend (x : ℝ) : ℝ := 3 * x^5 - 2 * x^3 + 5 * x^2 - 9
def divisor (x : ℝ) : ℝ := x^2 - 2 * x + 1
def remainder (x : ℝ) : ℝ := 19 * x - 22

theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, dividend x = divisor x * q x + remainder x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l397_39766


namespace NUMINAMATH_CALUDE_miles_owns_seventeen_instruments_l397_39788

/-- Represents the number of musical instruments Miles owns --/
structure MilesInstruments where
  fingers : ℕ
  hands : ℕ
  heads : ℕ
  trumpets : ℕ
  guitars : ℕ
  trombones : ℕ
  frenchHorns : ℕ

/-- The total number of musical instruments Miles owns --/
def totalInstruments (m : MilesInstruments) : ℕ :=
  m.trumpets + m.guitars + m.trombones + m.frenchHorns

/-- Theorem stating that Miles owns 17 musical instruments --/
theorem miles_owns_seventeen_instruments (m : MilesInstruments)
  (h1 : m.fingers = 10)
  (h2 : m.hands = 2)
  (h3 : m.heads = 1)
  (h4 : m.trumpets = m.fingers - 3)
  (h5 : m.guitars = m.hands + 2)
  (h6 : m.trombones = m.heads + 2)
  (h7 : m.frenchHorns = m.guitars - 1) :
  totalInstruments m = 17 := by
  sorry

#check miles_owns_seventeen_instruments

end NUMINAMATH_CALUDE_miles_owns_seventeen_instruments_l397_39788


namespace NUMINAMATH_CALUDE_permutations_of_polarized_l397_39792

theorem permutations_of_polarized (n : ℕ) (h : n = 9) :
  Nat.factorial n = 362880 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_polarized_l397_39792


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l397_39743

/-- The number of grandchildren Mrs. Lee has -/
def n : ℕ := 12

/-- The probability of having a grandson (or granddaughter) -/
def p : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters -/
def unequal_probability : ℚ := 793/1024

theorem unequal_gender_probability :
  (1 - (n.choose (n/2)) * p^n) = unequal_probability :=
sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l397_39743


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l397_39763

/-- The ratio of muffin price to banana price in Martha's purchase -/
theorem muffin_banana_price_ratio :
  ∀ (m b : ℝ),
  m > 0 → b > 0 →
  3 * m + 4 * b = (1/3) * (5 * (1.2 * m) + 12 * (1.5 * b)) →
  (1.2 * m) / (1.5 * b) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l397_39763


namespace NUMINAMATH_CALUDE_largest_multiple_of_3_and_5_under_800_l397_39752

theorem largest_multiple_of_3_and_5_under_800 : 
  ∀ n : ℕ, n < 800 ∧ 3 ∣ n ∧ 5 ∣ n → n ≤ 795 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_3_and_5_under_800_l397_39752


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l397_39744

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + 6*y + m = 0 → y = x) ↔ 
  m = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l397_39744


namespace NUMINAMATH_CALUDE_extraneous_root_condition_l397_39717

/-- The equation has an extraneous root when m = -4 -/
theorem extraneous_root_condition (m : ℝ) : 
  (m = -4) → 
  (∃ (x : ℝ), x ≠ 2 ∧ 
    (m / (x - 2) - (2 * x) / (2 - x) = 1) ∧
    (m / (2 - 2) - (2 * 2) / (2 - 2) ≠ 1)) :=
by sorry


end NUMINAMATH_CALUDE_extraneous_root_condition_l397_39717


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l397_39754

theorem arithmetic_evaluation : 6 + 4 / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l397_39754


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l397_39720

theorem sum_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 10)
  (diff_eq : x - y = 8)
  (sq_diff_eq : x^2 - y^2 = 80) : 
  x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l397_39720


namespace NUMINAMATH_CALUDE_b_is_largest_l397_39774

/-- Represents a number with a finite or repeating decimal expansion -/
structure DecimalNumber where
  whole : ℕ
  finite : List ℕ
  repeating : List ℕ

/-- Converts a DecimalNumber to a real number -/
noncomputable def toReal (d : DecimalNumber) : ℝ :=
  sorry

/-- The five numbers we're comparing -/
def a : DecimalNumber := { whole := 8, finite := [1, 2, 3, 6, 6], repeating := [] }
def b : DecimalNumber := { whole := 8, finite := [1, 2, 3], repeating := [6] }
def c : DecimalNumber := { whole := 8, finite := [1, 2], repeating := [3, 6] }
def d : DecimalNumber := { whole := 8, finite := [1], repeating := [2, 3, 6] }
def e : DecimalNumber := { whole := 8, finite := [], repeating := [1, 2, 3, 6] }

/-- Theorem stating that b is the largest among the given numbers -/
theorem b_is_largest :
  (toReal b > toReal a) ∧
  (toReal b > toReal c) ∧
  (toReal b > toReal d) ∧
  (toReal b > toReal e) :=
by
  sorry

end NUMINAMATH_CALUDE_b_is_largest_l397_39774


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l397_39726

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 30 / 100 →
  germination_rate2 = 35 / 100 →
  (((seeds_plot1 : ℚ) * germination_rate1 + (seeds_plot2 : ℚ) * germination_rate2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) * 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l397_39726


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_power_l397_39732

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal -/
def symmetric_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

theorem symmetry_implies_sum_power (m n : ℝ) :
  symmetric_y_axis m 3 4 n → (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_power_l397_39732


namespace NUMINAMATH_CALUDE_max_value_theorem_l397_39793

def feasible_region (x y : ℝ) : Prop :=
  2 * x + y ≤ 4 ∧ x ≤ y ∧ x ≥ 1/2

def objective_function (x y : ℝ) : ℝ :=
  2 * x - y

theorem max_value_theorem :
  ∃ (x y : ℝ), feasible_region x y ∧
  ∀ (x' y' : ℝ), feasible_region x' y' →
  objective_function x y ≥ objective_function x' y' ∧
  objective_function x y = 4/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l397_39793


namespace NUMINAMATH_CALUDE_initial_persons_count_l397_39786

/-- The number of persons initially in the group -/
def n : ℕ := sorry

/-- The average weight increase when a new person replaces one person -/
def average_increase : ℚ := 5/2

/-- The weight difference between the new person and the replaced person -/
def weight_difference : ℕ := 20

/-- Theorem stating that the initial number of persons is 8 -/
theorem initial_persons_count : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_persons_count_l397_39786


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l397_39740

variable (n : ℕ) (hn : n ≥ 3) (hodd : Odd n)
variable (A B C : Polynomial ℝ)

theorem polynomial_equation_solution :
  A^n + B^n + C^n = 0 →
  ∃ (a b c : ℝ) (D : Polynomial ℝ),
    a^n + b^n + c^n = 0 ∧
    A = a • D ∧
    B = b • D ∧
    C = c • D :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l397_39740


namespace NUMINAMATH_CALUDE_sum_of_squares_nonzero_implies_one_nonzero_l397_39761

theorem sum_of_squares_nonzero_implies_one_nonzero (a b : ℝ) : 
  a^2 + b^2 ≠ 0 → a ≠ 0 ∨ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_nonzero_implies_one_nonzero_l397_39761


namespace NUMINAMATH_CALUDE_staircase_classroom_seats_l397_39776

/-- Represents the number of seats in a row of the staircase classroom. -/
def seats (n : ℕ) (a : ℕ) : ℕ := 12 + (n - 1) * a

theorem staircase_classroom_seats :
  ∃ a : ℕ,
  (seats 15 a = 2 * seats 5 a) ∧ 
  (seats 21 a = 52) := by
  sorry

end NUMINAMATH_CALUDE_staircase_classroom_seats_l397_39776


namespace NUMINAMATH_CALUDE_gumball_problem_l397_39797

theorem gumball_problem (alicia_gumballs : ℕ) : 
  alicia_gumballs = 20 →
  let pedro_gumballs := alicia_gumballs + (alicia_gumballs * 3 / 2)
  let maria_gumballs := pedro_gumballs / 2
  let alicia_eaten := alicia_gumballs / 3
  let pedro_eaten := pedro_gumballs / 3
  let maria_eaten := maria_gumballs / 3
  (alicia_gumballs - alicia_eaten) + (pedro_gumballs - pedro_eaten) + (maria_gumballs - maria_eaten) = 65 := by
sorry

end NUMINAMATH_CALUDE_gumball_problem_l397_39797


namespace NUMINAMATH_CALUDE_quadratic_solution_l397_39771

theorem quadratic_solution (x : ℝ) : 
  (x = (7 + Real.sqrt 57) / 4 ∨ x = (7 - Real.sqrt 57) / 4) ↔ 
  2 * x^2 - 7 * x - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l397_39771


namespace NUMINAMATH_CALUDE_smallest_factorial_with_1987_zeros_l397_39706

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The smallest natural number n such that n! ends with exactly 1987 zeros -/
def smallestFactorialWith1987Zeros : ℕ := 7960

theorem smallest_factorial_with_1987_zeros :
  (∀ m : ℕ, m < smallestFactorialWith1987Zeros → trailingZeros m < 1987) ∧
  trailingZeros smallestFactorialWith1987Zeros = 1987 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorial_with_1987_zeros_l397_39706


namespace NUMINAMATH_CALUDE_probability_no_adjacent_same_rolls_l397_39707

-- Define the number of people
def num_people : ℕ := 5

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Define the probability of adjacent people rolling different numbers
def prob_diff_adjacent : ℚ := (die_sides - 1) / die_sides

-- Define the probability of no two adjacent people rolling the same number
def prob_no_adjacent_same : ℚ := prob_diff_adjacent ^ (num_people - 1)

-- Theorem statement
theorem probability_no_adjacent_same_rolls :
  prob_no_adjacent_same = 625 / 1296 := by sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_same_rolls_l397_39707


namespace NUMINAMATH_CALUDE_total_pages_in_collection_l397_39789

/-- Represents a book in the reader's collection -/
structure Book where
  chapterPages : List Nat
  additionalPages : Nat

/-- The reader's book collection -/
def bookCollection : List Book := [
  { chapterPages := [22, 34, 18, 46, 30, 38], additionalPages := 14 },  -- Science
  { chapterPages := [24, 32, 40, 20], additionalPages := 13 },          -- History
  { chapterPages := [12, 28, 16, 22, 18, 26, 20], additionalPages := 8 }, -- Literature
  { chapterPages := [48, 52, 36, 62, 24], additionalPages := 18 },      -- Art
  { chapterPages := [16, 28, 44], additionalPages := 28 }               -- Mathematics
]

/-- Calculate the total pages in a book -/
def totalPagesInBook (book : Book) : Nat :=
  (book.chapterPages.sum) + book.additionalPages

/-- Calculate the total pages in the collection -/
def totalPagesInCollection (collection : List Book) : Nat :=
  collection.map totalPagesInBook |>.sum

/-- Theorem: The total number of pages in the reader's collection is 837 -/
theorem total_pages_in_collection :
  totalPagesInCollection bookCollection = 837 := by
  sorry


end NUMINAMATH_CALUDE_total_pages_in_collection_l397_39789


namespace NUMINAMATH_CALUDE_joan_video_game_spending_l397_39760

/-- The cost of the basketball game Joan purchased -/
def basketball_cost : ℚ := 5.2

/-- The cost of the racing game Joan purchased -/
def racing_cost : ℚ := 4.23

/-- The total amount Joan spent on video games -/
def total_spent : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total amount Joan spent on video games is $9.43 -/
theorem joan_video_game_spending : total_spent = 9.43 := by sorry

end NUMINAMATH_CALUDE_joan_video_game_spending_l397_39760


namespace NUMINAMATH_CALUDE_crayon_selection_combinations_l397_39796

theorem crayon_selection_combinations : Nat.choose 15 5 = 3003 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_combinations_l397_39796


namespace NUMINAMATH_CALUDE_rational_additive_function_is_linear_l397_39703

theorem rational_additive_function_is_linear 
  (f : ℚ → ℚ) 
  (h : ∀ (x y : ℚ), f (x + y) = f x + f y) : 
  ∃ (c : ℚ), ∀ (x : ℚ), f x = c * x := by
sorry

end NUMINAMATH_CALUDE_rational_additive_function_is_linear_l397_39703


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l397_39799

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles_width (carol_rect jordan_rect : Rectangle) 
  (h1 : carol_rect.length = 15)
  (h2 : carol_rect.width = 20)
  (h3 : jordan_rect.length = 6 * 12)
  (h4 : area carol_rect = area jordan_rect) :
  jordan_rect.width = 300 / (6 * 12) := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l397_39799
