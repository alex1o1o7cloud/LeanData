import Mathlib

namespace NUMINAMATH_CALUDE_min_value_abc_min_value_abc_attainable_l1730_173019

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + 2*b + 3*c = a*b*c) : 
  a*b*c ≥ 9*Real.sqrt 2 := by
sorry

theorem min_value_abc_attainable : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + 2*b + 3*c = a*b*c ∧ a*b*c = 9*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_abc_min_value_abc_attainable_l1730_173019


namespace NUMINAMATH_CALUDE_inequality_proof_ratio_proof_l1730_173049

-- Part I
theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by sorry

-- Part II
theorem ratio_proof (a b c x y z : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0)
  (h7 : a^2 + b^2 + c^2 = 10) (h8 : x^2 + y^2 + z^2 = 40) (h9 : a*x + b*y + c*z = 20) :
  (a + b + c) / (x + y + z) = 1/2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_ratio_proof_l1730_173049


namespace NUMINAMATH_CALUDE_theater_capacity_is_50_l1730_173066

/-- The maximum capacity of a movie theater -/
def theater_capacity (ticket_price : ℕ) (tickets_sold : ℕ) (loss_amount : ℕ) : ℕ :=
  tickets_sold + loss_amount / ticket_price

/-- Theorem: The maximum capacity of the movie theater is 50 people -/
theorem theater_capacity_is_50 :
  theater_capacity 8 24 208 = 50 := by
  sorry

end NUMINAMATH_CALUDE_theater_capacity_is_50_l1730_173066


namespace NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l1730_173024

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_last_two_digits (series : List ℕ) : ℕ :=
  (series.map (λ n => last_two_digits (factorial n))).sum

theorem fibonacci_factorial_last_two_digits_sum :
  sum_last_two_digits fibonacci_factorial_series = 5 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l1730_173024


namespace NUMINAMATH_CALUDE_dormitory_expenditure_l1730_173008

theorem dormitory_expenditure 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (cost_decrease : ℕ) 
  (expenditure_increase : ℕ) 
  (h1 : initial_students = 250)
  (h2 : new_students = 75)
  (h3 : cost_decrease = 20)
  (h4 : expenditure_increase = 10000) :
  (initial_students + new_students) * 
  ((initial_students + new_students) * expenditure_increase / initial_students - cost_decrease) = 65000 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_expenditure_l1730_173008


namespace NUMINAMATH_CALUDE_digit_sum_is_two_l1730_173018

/-- Given a four-digit number abcd and a three-digit number bcd, where a, b, c, d are distinct digits 
    and abcd - bcd is a two-digit number, the sum of a, b, c, and d is 2. -/
theorem digit_sum_is_two (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  1000 * a + 100 * b + 10 * c + d > 999 →
  1000 * a + 100 * b + 10 * c + d - (100 * b + 10 * c + d) < 100 →
  a + b + c + d = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_is_two_l1730_173018


namespace NUMINAMATH_CALUDE_circle_center_l1730_173003

/-- A circle passes through (0,1) and is tangent to y = (x-1)^2 at (3,4). Its center is (-2, 15/2). -/
theorem circle_center (c : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x - c.1)^2 + (y - c.2)^2 = (c.1 - 3)^2 + (c.2 - 4)^2 → 
    (x = 0 ∧ y = 1) ∨ (x = 3 ∧ y = 4)) →
  (∀ (x : ℝ), (x - 3)^2 + (((x - 1)^2 - 4)^2) / 16 = (c.1 - 3)^2 + (c.2 - 4)^2) →
  c = (-2, 15/2) := by
sorry

end NUMINAMATH_CALUDE_circle_center_l1730_173003


namespace NUMINAMATH_CALUDE_min_value_fraction_lower_bound_achievable_l1730_173087

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  (x + y) / (x * y * z) ≥ 16 := by
  sorry

theorem lower_bound_achievable : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ (x + y) / (x * y * z) = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_lower_bound_achievable_l1730_173087


namespace NUMINAMATH_CALUDE_set_a_values_l1730_173099

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem set_a_values (a : ℝ) : A ∪ B a = A ↔ a = -2 ∨ a ≥ 4 ∨ a < -4 := by
  sorry

end NUMINAMATH_CALUDE_set_a_values_l1730_173099


namespace NUMINAMATH_CALUDE_ryan_study_difference_l1730_173055

/-- Ryan's daily study hours for different languages -/
structure StudyHours where
  english : ℕ
  chinese : ℕ
  spanish : ℕ

/-- The difference in study hours between Chinese and Spanish -/
def chineseSpanishDifference (h : StudyHours) : ℤ :=
  h.chinese - h.spanish

/-- Theorem stating the difference in study hours between Chinese and Spanish -/
theorem ryan_study_difference :
  ∀ (h : StudyHours),
    h.english = 2 → h.chinese = 5 → h.spanish = 4 →
    chineseSpanishDifference h = 1 := by
  sorry

end NUMINAMATH_CALUDE_ryan_study_difference_l1730_173055


namespace NUMINAMATH_CALUDE_number_division_problem_l1730_173081

theorem number_division_problem (x : ℝ) : x / 0.3 = 7.3500000000000005 → x = 2.205 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1730_173081


namespace NUMINAMATH_CALUDE_xy_product_cardinality_l1730_173050

def X : Finset ℕ := {1, 2, 3, 4}
def Y : Finset ℕ := {5, 6, 7, 8}

theorem xy_product_cardinality :
  Finset.card ((X.product Y).image (λ (p : ℕ × ℕ) => p.1 * p.2)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_xy_product_cardinality_l1730_173050


namespace NUMINAMATH_CALUDE_sum_of_ages_after_20_years_l1730_173063

/-- Given the ages of Ann and her siblings and cousin, calculate the sum of their ages after 20 years -/
theorem sum_of_ages_after_20_years 
  (ann_age : ℕ)
  (tom_age : ℕ)
  (bill_age : ℕ)
  (cathy_age : ℕ)
  (emily_age : ℕ)
  (h1 : ann_age = 6)
  (h2 : tom_age = 2 * ann_age)
  (h3 : bill_age = tom_age - 3)
  (h4 : cathy_age = 2 * tom_age)
  (h5 : emily_age = cathy_age / 2)
  : ann_age + tom_age + bill_age + cathy_age + emily_age + 20 * 5 = 163 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_after_20_years_l1730_173063


namespace NUMINAMATH_CALUDE_class_size_l1730_173023

theorem class_size (top_scorers : Nat) (zero_scorers : Nat) (top_score : Nat) (rest_avg : Nat) (class_avg : Nat) :
  top_scorers = 3 →
  zero_scorers = 5 →
  top_score = 95 →
  rest_avg = 45 →
  class_avg = 42 →
  ∃ (N : Nat), N = 25 ∧ 
    (N * class_avg = top_scorers * top_score + zero_scorers * 0 + (N - top_scorers - zero_scorers) * rest_avg) :=
by sorry

end NUMINAMATH_CALUDE_class_size_l1730_173023


namespace NUMINAMATH_CALUDE_sqrt_one_plus_cos_alpha_l1730_173004

theorem sqrt_one_plus_cos_alpha (α : Real) (h : π < α ∧ α < 2*π) :
  Real.sqrt (1 + Real.cos α) = -Real.sqrt 2 * Real.cos (α/2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_plus_cos_alpha_l1730_173004


namespace NUMINAMATH_CALUDE_two_fifths_in_four_fifths_minus_one_tenth_l1730_173059

theorem two_fifths_in_four_fifths_minus_one_tenth : 
  (4/5 - 1/10) / (2/5) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_in_four_fifths_minus_one_tenth_l1730_173059


namespace NUMINAMATH_CALUDE_leftover_value_is_230_l1730_173056

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents a collection of coins --/
structure Coins where
  quarters : Nat
  dimes : Nat

def roll_size : RollSize := { quarters := 25, dimes := 40 }

def john_coins : Coins := { quarters := 47, dimes := 71 }
def mark_coins : Coins := { quarters := 78, dimes := 132 }

def combine_coins (c1 c2 : Coins) : Coins :=
  { quarters := c1.quarters + c2.quarters,
    dimes := c1.dimes + c2.dimes }

def leftover_coins (c : Coins) (r : RollSize) : Coins :=
  { quarters := c.quarters % r.quarters,
    dimes := c.dimes % r.dimes }

def coin_value (c : Coins) : Rat :=
  (c.quarters : Rat) * (1/4) + (c.dimes : Rat) * (1/10)

theorem leftover_value_is_230 :
  let combined := combine_coins john_coins mark_coins
  let leftover := leftover_coins combined roll_size
  coin_value leftover = 23/10 := by sorry

end NUMINAMATH_CALUDE_leftover_value_is_230_l1730_173056


namespace NUMINAMATH_CALUDE_train_length_calculation_l1730_173034

/-- The length of a train given crossing time and speeds --/
theorem train_length_calculation (crossing_time : ℝ) (man_speed : ℝ) (train_speed : ℝ) :
  crossing_time = 39.99680025597952 →
  man_speed = 2 →
  train_speed = 56 →
  ∃ (train_length : ℝ), abs (train_length - 599.95) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1730_173034


namespace NUMINAMATH_CALUDE_min_disks_is_ten_l1730_173057

/-- Represents the storage problem with given file sizes and disk capacity. -/
structure StorageProblem where
  total_files : Nat
  disk_capacity : Rat
  files_06MB : Nat
  files_10MB : Nat
  files_03MB : Nat

/-- Calculates the minimum number of disks needed for the given storage problem. -/
def min_disks_needed (problem : StorageProblem) : Nat :=
  sorry

/-- Theorem stating that the minimum number of disks needed is 10 for the given problem. -/
theorem min_disks_is_ten (problem : StorageProblem) 
  (h1 : problem.total_files = 25)
  (h2 : problem.disk_capacity = 2)
  (h3 : problem.files_06MB = 5)
  (h4 : problem.files_10MB = 10)
  (h5 : problem.files_03MB = 10) :
  min_disks_needed problem = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_disks_is_ten_l1730_173057


namespace NUMINAMATH_CALUDE_opponent_score_proof_l1730_173038

def championship_game (total_points : ℕ) (num_games : ℕ) (point_difference : ℕ) : Prop :=
  let avg_points : ℚ := (total_points : ℚ) / num_games
  let uf_championship_score : ℚ := avg_points / 2 - 2
  let opponent_score : ℚ := uf_championship_score + point_difference
  opponent_score = 15

theorem opponent_score_proof :
  championship_game 720 24 2 := by
  sorry

end NUMINAMATH_CALUDE_opponent_score_proof_l1730_173038


namespace NUMINAMATH_CALUDE_lemniscate_orthogonal_trajectories_l1730_173084

-- Define the lemniscate family
def lemniscate (a : ℝ) (ρ φ : ℝ) : Prop :=
  ρ^2 = a * Real.cos (2 * φ)

-- Define the orthogonal trajectory
def orthogonal_trajectory (C : ℝ) (ρ φ : ℝ) : Prop :=
  ρ^2 = C * Real.sin (2 * φ)

-- Theorem statement
theorem lemniscate_orthogonal_trajectories (a C : ℝ) (ρ φ : ℝ) :
  lemniscate a ρ φ → orthogonal_trajectory C ρ φ :=
by
  sorry

end NUMINAMATH_CALUDE_lemniscate_orthogonal_trajectories_l1730_173084


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l1730_173062

theorem triangle_cosine_inequality (A B C : Real) : 
  A > 0 → B > 0 → C > 0 → A + B + C = Real.pi → 
  Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2) ≤ 3 * Real.sqrt 3 / 2 ∧
  (Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2) = 3 * Real.sqrt 3 / 2 ↔ 
   A = Real.pi/3 ∧ B = Real.pi/3 ∧ C = Real.pi/3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l1730_173062


namespace NUMINAMATH_CALUDE_actual_average_height_l1730_173082

/-- The actual average height of boys in a class with measurement errors -/
theorem actual_average_height (n : ℕ) (initial_avg : ℝ) 
  (error1 : ℝ) (error2 : ℝ) : 
  n = 40 → 
  initial_avg = 184 → 
  error1 = 166 - 106 → 
  error2 = 190 - 180 → 
  (n * initial_avg - (error1 + error2)) / n = 182.25 := by
  sorry

end NUMINAMATH_CALUDE_actual_average_height_l1730_173082


namespace NUMINAMATH_CALUDE_ellipse_cos_angle_l1730_173083

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let a := 3
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  F₁ = (-c, 0) ∧ F₂ = (c, 0)

-- Define a point on the ellipse
def point_on_ellipse (M : ℝ × ℝ) : Prop :=
  ellipse M.1 M.2

-- Define perpendicularity condition
def perpendicular_condition (M F₁ F₂ : ℝ × ℝ) : Prop :=
  (M.1 - F₁.1) * (F₂.1 - F₁.1) + (M.2 - F₁.2) * (F₂.2 - F₁.2) = 0

-- Theorem statement
theorem ellipse_cos_angle (M F₁ F₂ : ℝ × ℝ) :
  foci F₁ F₂ →
  point_on_ellipse M →
  perpendicular_condition M F₁ F₂ →
  let MF₁ := Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2)
  let MF₂ := Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2)
  MF₁ / MF₂ = 2/7 :=
sorry

end NUMINAMATH_CALUDE_ellipse_cos_angle_l1730_173083


namespace NUMINAMATH_CALUDE_key_arrangement_count_l1730_173078

/-- The number of keys on the keychain -/
def total_keys : ℕ := 6

/-- The number of effective units to arrange (treating the adjacent pair as one unit) -/
def effective_units : ℕ := total_keys - 1

/-- The number of ways to arrange the adjacent pair -/
def adjacent_pair_arrangements : ℕ := 2

/-- The number of distinct circular arrangements of n objects -/
def circular_arrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The total number of distinct arrangements -/
def total_arrangements : ℕ := circular_arrangements effective_units * adjacent_pair_arrangements

theorem key_arrangement_count : total_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_key_arrangement_count_l1730_173078


namespace NUMINAMATH_CALUDE_candy_heating_rate_l1730_173086

/-- Candy heating problem -/
theorem candy_heating_rate
  (initial_temp : ℝ)
  (max_temp : ℝ)
  (final_temp : ℝ)
  (cooling_rate : ℝ)
  (total_time : ℝ)
  (h1 : initial_temp = 60)
  (h2 : max_temp = 240)
  (h3 : final_temp = 170)
  (h4 : cooling_rate = 7)
  (h5 : total_time = 46)
  : ∃ (heating_rate : ℝ), heating_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_heating_rate_l1730_173086


namespace NUMINAMATH_CALUDE_divisor_with_remainder_54_l1730_173001

theorem divisor_with_remainder_54 :
  ∃ (n : ℕ), n > 0 ∧ (55^55 + 55) % n = 54 ∧ n = 56 := by sorry

end NUMINAMATH_CALUDE_divisor_with_remainder_54_l1730_173001


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1730_173079

/-- Given a line L1 with equation x + 3y + 4 = 0, prove that the line L2 with equation 3x - y - 5 = 0
    passes through the point (2, 1) and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x + 3 * y + 4 = 0) →  -- Equation of line L1
  (3 * 2 - 1 - 5 = 0) →  -- L2 passes through (2, 1)
  (3 * (1 / 3) = -1) →   -- Slopes are negative reciprocals
  (3 * x - y - 5 = 0) -- Equation of line L2
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1730_173079


namespace NUMINAMATH_CALUDE_remainder_of_large_power_l1730_173037

theorem remainder_of_large_power (n : ℕ) : 
  4^(4^(4^4)) ≡ 656 [ZMOD 1000] :=
sorry

end NUMINAMATH_CALUDE_remainder_of_large_power_l1730_173037


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1730_173076

theorem quadratic_root_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 1 ∧ b = -6 ∧ c = 8 → |r₁ - r₂| = 2 :=
by
  sorry

#check quadratic_root_difference

end NUMINAMATH_CALUDE_quadratic_root_difference_l1730_173076


namespace NUMINAMATH_CALUDE_mother_double_age_in_18_years_l1730_173046

/-- Represents the number of years until Xiaoming's mother's age is twice Xiaoming's age -/
def years_until_double_age (xiaoming_age : ℕ) (mother_age : ℕ) : ℕ :=
  mother_age - 2 * xiaoming_age

theorem mother_double_age_in_18_years :
  let xiaoming_current_age : ℕ := 6
  let mother_current_age : ℕ := 30
  years_until_double_age xiaoming_current_age mother_current_age = 18 :=
by
  sorry

#check mother_double_age_in_18_years

end NUMINAMATH_CALUDE_mother_double_age_in_18_years_l1730_173046


namespace NUMINAMATH_CALUDE_side_to_base_ratio_l1730_173014

/-- Represents an isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  -- The length of one side of the isosceles triangle
  side : ℝ
  -- The length of the base of the isosceles triangle
  base : ℝ
  -- The distance from the vertex to the point of tangency on the side
  vertex_to_tangency : ℝ
  -- Ensure the triangle is isosceles
  isosceles : side > 0
  -- Ensure the point of tangency divides the side in 7:5 ratio
  tangency_ratio : vertex_to_tangency / (side - vertex_to_tangency) = 7 / 5

/-- 
Theorem: In an isosceles triangle with an inscribed circle, 
if the point of tangency on one side divides it in the ratio 7:5 (starting from the vertex), 
then the ratio of the side to the base is 6:5.
-/
theorem side_to_base_ratio 
  (triangle : IsoscelesTriangleWithInscribedCircle) : 
  triangle.side / triangle.base = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_side_to_base_ratio_l1730_173014


namespace NUMINAMATH_CALUDE_player_a_not_losing_probability_l1730_173093

theorem player_a_not_losing_probability
  (p_win : ℝ)
  (p_draw : ℝ)
  (h_win : p_win = 0.3)
  (h_draw : p_draw = 0.5) :
  p_win + p_draw = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_player_a_not_losing_probability_l1730_173093


namespace NUMINAMATH_CALUDE_arrangement_count_is_540_l1730_173085

/-- The number of ways to arrange teachers and students into groups and locations -/
def arrangement_count : ℕ :=
  (Nat.choose 6 2) * (Nat.choose 4 2) * (Nat.choose 2 2) * (Nat.factorial 3)

/-- Theorem stating that the number of arrangements is 540 -/
theorem arrangement_count_is_540 : arrangement_count = 540 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_540_l1730_173085


namespace NUMINAMATH_CALUDE_road_repair_hours_l1730_173069

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 39)
  (h2 : days1 = 12)
  (h3 : people2 = 30)
  (h4 : days2 = 26)
  (h5 : hours2 = 3)
  (h6 : people1 * days1 * (people1 * days1 * hours2 / (people2 * days2)) = people2 * days2 * hours2) :
  people1 * days1 * hours2 / (people2 * days2) = 5 := by
sorry

end NUMINAMATH_CALUDE_road_repair_hours_l1730_173069


namespace NUMINAMATH_CALUDE_grade_distribution_l1730_173006

theorem grade_distribution (n₂ n₃ n₄ n₅ : ℕ) : 
  n₂ + n₃ + n₄ + n₅ = 25 →
  n₄ = n₃ + 4 →
  2 * n₂ + 3 * n₃ + 4 * n₄ + 5 * n₅ = 121 →
  n₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_grade_distribution_l1730_173006


namespace NUMINAMATH_CALUDE_triangle_pieces_count_l1730_173074

/-- Sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Number of rods in the triangle -/
def totalRods : ℕ := arithmeticSum 5 5 10

/-- Number of connectors in the triangle -/
def totalConnectors : ℕ := arithmeticSum 3 3 11

/-- Total number of pieces in the triangle -/
def totalPieces : ℕ := totalRods + totalConnectors

theorem triangle_pieces_count : totalPieces = 473 := by
  sorry

end NUMINAMATH_CALUDE_triangle_pieces_count_l1730_173074


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_p_and_q_l1730_173054

theorem p_necessary_not_sufficient_for_p_and_q :
  (∃ p q : Prop, (p ∧ q → p) ∧ ¬(p → p ∧ q)) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_p_and_q_l1730_173054


namespace NUMINAMATH_CALUDE_product_not_equal_48_l1730_173044

theorem product_not_equal_48 : ∃! (a b : ℚ), (a, b) ∈ ({(-4, -12), (-3, -16), (1/2, -96), (1, 48), (4/3, 36)} : Set (ℚ × ℚ)) ∧ a * b ≠ 48 := by
  sorry

end NUMINAMATH_CALUDE_product_not_equal_48_l1730_173044


namespace NUMINAMATH_CALUDE_seagull_problem_l1730_173040

theorem seagull_problem (initial : ℕ) : 
  (initial : ℚ) * (3/4) * (2/3) = 18 → initial = 36 := by
  sorry

end NUMINAMATH_CALUDE_seagull_problem_l1730_173040


namespace NUMINAMATH_CALUDE_expansion_of_binomial_l1730_173047

theorem expansion_of_binomial (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x - 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₁ + a₂ + a₃ + a₄ = -80 ∧ (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 625) := by
  sorry

end NUMINAMATH_CALUDE_expansion_of_binomial_l1730_173047


namespace NUMINAMATH_CALUDE_max_power_of_two_divides_l1730_173015

/-- The highest power of 2 dividing a natural number -/
def v2 (n : ℕ) : ℕ := sorry

/-- The maximum power of 2 dividing (2019^n - 1) / 2018 for positive integer n -/
def max_power_of_two (n : ℕ+) : ℕ :=
  if n.val % 2 = 1 then 0 else v2 n.val + 1

/-- Theorem stating the maximum power of 2 dividing the given expression -/
theorem max_power_of_two_divides (n : ℕ+) :
  (2019^n.val - 1) / 2018 % 2^(max_power_of_two n) = 0 ∧
  ∀ k > max_power_of_two n, (2019^n.val - 1) / 2018 % 2^k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_max_power_of_two_divides_l1730_173015


namespace NUMINAMATH_CALUDE_addition_sequence_terms_l1730_173064

/-- Represents the nth term of the first sequence in the addition pattern -/
def a (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the nth term of the second sequence in the addition pattern -/
def b (n : ℕ) : ℕ := 5 * n - 1

/-- Proves the correctness of the 10th and 80th terms in the addition sequence -/
theorem addition_sequence_terms :
  (a 10 = 21 ∧ b 10 = 49) ∧ (a 80 = 161 ∧ b 80 = 399) := by
  sorry

#eval a 10  -- Expected: 21
#eval b 10  -- Expected: 49
#eval a 80  -- Expected: 161
#eval b 80  -- Expected: 399

end NUMINAMATH_CALUDE_addition_sequence_terms_l1730_173064


namespace NUMINAMATH_CALUDE_reciprocal_square_sum_l1730_173005

theorem reciprocal_square_sum : (((1 : ℚ) / 4 + 1 / 6) ^ 2)⁻¹ = 144 / 25 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_square_sum_l1730_173005


namespace NUMINAMATH_CALUDE_dance_attendance_l1730_173036

theorem dance_attendance (girls : ℕ) (boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end NUMINAMATH_CALUDE_dance_attendance_l1730_173036


namespace NUMINAMATH_CALUDE_division_problem_l1730_173071

theorem division_problem : (8900 / 6) / 4 = 1483 + 1/3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1730_173071


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1730_173080

/-- The equation of an asymptote of the hyperbola y²/8 - x²/6 = 1 -/
theorem hyperbola_asymptote :
  ∃ (x y : ℝ), (y^2 / 8 - x^2 / 6 = 1) →
  (2 * x - Real.sqrt 3 * y = 0 ∨ 2 * x + Real.sqrt 3 * y = 0) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1730_173080


namespace NUMINAMATH_CALUDE_silver_cube_side_length_l1730_173029

/-- Proves that a silver cube sold for $4455 at 110% of its silver value, 
    where a cubic inch of silver weighs 6 ounces and each ounce of silver 
    sells for $25, has a side length of 3 inches. -/
theorem silver_cube_side_length :
  let selling_price : ℝ := 4455
  let markup_percentage : ℝ := 1.10
  let weight_per_cubic_inch : ℝ := 6
  let price_per_ounce : ℝ := 25
  let side_length : ℝ := (selling_price / markup_percentage / price_per_ounce / weight_per_cubic_inch) ^ (1/3)
  side_length = 3 := by sorry

end NUMINAMATH_CALUDE_silver_cube_side_length_l1730_173029


namespace NUMINAMATH_CALUDE_train_distance_difference_l1730_173010

/-- Represents the distance traveled by a train given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the total distance between two points -/
def total_distance : ℝ := 900

/-- Represents the speed of the first train -/
def speed1 : ℝ := 50

/-- Represents the speed of the second train -/
def speed2 : ℝ := 40

/-- Theorem stating the difference in distance traveled by two trains -/
theorem train_distance_difference :
  ∃ (time : ℝ), 
    time > 0 ∧
    distance speed1 time + distance speed2 time = total_distance ∧
    distance speed1 time - distance speed2 time = 100 :=
sorry

end NUMINAMATH_CALUDE_train_distance_difference_l1730_173010


namespace NUMINAMATH_CALUDE_parabola_properties_l1730_173098

/-- Parabola with symmetric axis at x = -2 passing through (1, -2) and c > 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  symmetric_axis : a * (-2) + b = 0
  passes_through : a * 1^2 + b * 1 + c = -2
  c_positive : c > 0

theorem parabola_properties (p : Parabola) :
  p.a < 0 ∧ 16 * p.a + p.c > 4 * p.b := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1730_173098


namespace NUMINAMATH_CALUDE_stock_price_decrease_l1730_173053

/-- The percentage decrease required for a stock to return to its original price after a 40% increase -/
theorem stock_price_decrease (initial_price : ℝ) (h : initial_price > 0) :
  let increased_price := 1.4 * initial_price
  let decrease_percent := (increased_price - initial_price) / increased_price
  decrease_percent = 0.2857142857142857 := by
sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l1730_173053


namespace NUMINAMATH_CALUDE_line_inclination_45_degrees_l1730_173000

theorem line_inclination_45_degrees (a : ℝ) : 
  (∃ (x y : ℝ), ax + (2*a - 3)*y = 0) →   -- Line equation
  (Real.arctan (-a / (2*a - 3)) = π/4) →  -- 45° inclination
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_line_inclination_45_degrees_l1730_173000


namespace NUMINAMATH_CALUDE_sum_equals_300_l1730_173020

theorem sum_equals_300 : 192 + 58 + 42 + 8 = 300 := by sorry

end NUMINAMATH_CALUDE_sum_equals_300_l1730_173020


namespace NUMINAMATH_CALUDE_tyler_meal_choices_l1730_173041

-- Define the number of options for each food category
def num_meats : ℕ := 3
def num_vegetables : ℕ := 5
def num_desserts : ℕ := 4

-- Define the number of vegetables to be chosen
def vegetables_to_choose : ℕ := 2

-- Theorem statement
theorem tyler_meal_choices :
  (num_meats * (Nat.choose num_vegetables vegetables_to_choose) * num_desserts) = 120 := by
  sorry

end NUMINAMATH_CALUDE_tyler_meal_choices_l1730_173041


namespace NUMINAMATH_CALUDE_cary_shoe_savings_l1730_173045

def cost_of_shoes : ℕ := 120
def amount_saved : ℕ := 30
def earnings_per_lawn : ℕ := 5
def lawns_per_weekend : ℕ := 3

def weekends_needed : ℕ :=
  (cost_of_shoes - amount_saved) / (earnings_per_lawn * lawns_per_weekend)

theorem cary_shoe_savings : weekends_needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_cary_shoe_savings_l1730_173045


namespace NUMINAMATH_CALUDE_prob_green_is_one_eighth_l1730_173009

-- Define the number of cubes for each color
def pink_cubes : ℕ := 36
def blue_cubes : ℕ := 18
def green_cubes : ℕ := 9
def red_cubes : ℕ := 6
def purple_cubes : ℕ := 3

-- Define the total number of cubes
def total_cubes : ℕ := pink_cubes + blue_cubes + green_cubes + red_cubes + purple_cubes

-- Define the probability of selecting a green cube
def prob_green : ℚ := green_cubes / total_cubes

-- Theorem statement
theorem prob_green_is_one_eighth : prob_green = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_is_one_eighth_l1730_173009


namespace NUMINAMATH_CALUDE_star_two_three_l1730_173089

-- Define the star operation
def star (c d : ℝ) : ℝ := c^3 + 3*c^2*d + 3*c*d^2 + d^3

-- State the theorem
theorem star_two_three : star 2 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_star_two_three_l1730_173089


namespace NUMINAMATH_CALUDE_inequality_proof_l1730_173048

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1730_173048


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l1730_173088

theorem least_positive_integer_with_given_remainders : ∃! x : ℕ, 
  x > 0 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 7 = 6 ∧
  x % 9 = 8 ∧
  ∀ y : ℕ, y > 0 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 7 = 6 ∧ y % 9 = 8 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l1730_173088


namespace NUMINAMATH_CALUDE_steps_per_floor_l1730_173092

/-- Proves that the number of steps across each floor is 30 --/
theorem steps_per_floor (
  num_floors : ℕ) 
  (steps_per_second : ℕ)
  (total_time : ℕ)
  (h1 : num_floors = 9)
  (h2 : steps_per_second = 3)
  (h3 : total_time = 90)
  : (steps_per_second * total_time) / num_floors = 30 := by
  sorry

end NUMINAMATH_CALUDE_steps_per_floor_l1730_173092


namespace NUMINAMATH_CALUDE_division_increase_by_digit_swap_l1730_173042

theorem division_increase_by_digit_swap (n : Nat) (d : Nat) :
  n = 952473 →
  d = 18 →
  (954273 / d) - (n / d) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_division_increase_by_digit_swap_l1730_173042


namespace NUMINAMATH_CALUDE_pizza_delivery_gas_remaining_l1730_173091

theorem pizza_delivery_gas_remaining (start_amount used_amount : ℚ) 
  (h1 : start_amount = 0.5)
  (h2 : used_amount = 0.33) : 
  start_amount - used_amount = 0.17 := by
sorry

end NUMINAMATH_CALUDE_pizza_delivery_gas_remaining_l1730_173091


namespace NUMINAMATH_CALUDE_correct_number_of_pupils_l1730_173030

/-- The number of pupils in a class where an error in one pupil's marks
    caused the class average to increase by half a mark. -/
def number_of_pupils : ℕ :=
  -- We define this as 20, which is the value we want to prove
  20

/-- The increase in one pupil's marks due to the error -/
def mark_increase : ℕ := 10

/-- The increase in the class average due to the error -/
def average_increase : ℚ := 1/2

theorem correct_number_of_pupils :
  mark_increase = (number_of_pupils : ℚ) * average_increase :=
sorry

end NUMINAMATH_CALUDE_correct_number_of_pupils_l1730_173030


namespace NUMINAMATH_CALUDE_cube_root_simplification_l1730_173077

theorem cube_root_simplification :
  Real.rpow (20^3 + 30^3 + 40^3) (1/3) = 10 * Real.rpow 99 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l1730_173077


namespace NUMINAMATH_CALUDE_remainder_proof_l1730_173021

theorem remainder_proof (R1 : ℕ) : 
  (129 = Nat.gcd (1428 - R1) (2206 - 13)) → 
  (2206 % 129 = 13) → 
  (1428 % 129 = 19) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l1730_173021


namespace NUMINAMATH_CALUDE_school_gender_ratio_l1730_173031

theorem school_gender_ratio (num_girls : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) :
  num_girls = 1200 →
  ratio_boys = 5 →
  ratio_girls = 4 →
  (ratio_boys : ℚ) / ratio_girls * num_girls = 1500 :=
by sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l1730_173031


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1730_173007

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24 → 
  (x : ℤ) + y ≤ (a : ℤ) + b → (x : ℤ) + y = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1730_173007


namespace NUMINAMATH_CALUDE_negation_equivalence_l1730_173026

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1730_173026


namespace NUMINAMATH_CALUDE_y_is_twenty_percent_of_x_l1730_173017

/-- Given two equations involving x, y, and z, prove that y is 20% of x -/
theorem y_is_twenty_percent_of_x (x y z : ℝ) 
  (eq1 : 0.3 * (x - y) = 0.2 * (x + y))
  (eq2 : 0.4 * (x + z) = 0.1 * (y - z)) :
  y = 0.2 * x := by
  sorry

end NUMINAMATH_CALUDE_y_is_twenty_percent_of_x_l1730_173017


namespace NUMINAMATH_CALUDE_sin_75_times_sin_15_l1730_173094

theorem sin_75_times_sin_15 : Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_times_sin_15_l1730_173094


namespace NUMINAMATH_CALUDE_class_size_l1730_173043

theorem class_size (initial_avg : ℝ) (misread_weight : ℝ) (correct_weight : ℝ) (final_avg : ℝ) :
  initial_avg = 58.4 →
  misread_weight = 56 →
  correct_weight = 66 →
  final_avg = 58.9 →
  ∃ n : ℕ, n > 0 ∧ n * initial_avg + (correct_weight - misread_weight) = n * final_avg ∧ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l1730_173043


namespace NUMINAMATH_CALUDE_smallest_valid_n_l1730_173052

def is_valid (n : ℕ) : Prop :=
  ∃ k₁ k₂ : ℕ, 1 ≤ k₁ ∧ k₁ ≤ n ∧ 1 ≤ k₂ ∧ k₂ ≤ n ∧
  (n^2 + n) % k₁ = 0 ∧ (n^2 + n) % k₂ ≠ 0

theorem smallest_valid_n :
  is_valid 4 ∧ ∀ m : ℕ, 0 < m ∧ m < 4 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l1730_173052


namespace NUMINAMATH_CALUDE_percentage_euros_to_dollars_l1730_173035

/-- Converts a percentage of Euros to US Dollars -/
theorem percentage_euros_to_dollars
  (X : ℝ) -- Unknown amount in Euros
  (Y : ℝ) -- Exchange rate (1 Euro = Y US Dollars)
  (h : Y > 0) -- Y is positive
  : (25 / 100 : ℝ) * X * Y = 0.25 * X * Y := by
  sorry

end NUMINAMATH_CALUDE_percentage_euros_to_dollars_l1730_173035


namespace NUMINAMATH_CALUDE_rachel_painting_time_l1730_173039

/-- Prove that Rachel's painting time is 13 hours -/
theorem rachel_painting_time : ℝ → ℝ → ℝ → Prop :=
  fun matt_time patty_time rachel_time =>
    matt_time = 12 ∧
    patty_time = matt_time / 3 ∧
    rachel_time = 2 * patty_time + 5 →
    rachel_time = 13

/-- Proof of the theorem -/
lemma rachel_painting_time_proof : rachel_painting_time 12 4 13 := by
  sorry


end NUMINAMATH_CALUDE_rachel_painting_time_l1730_173039


namespace NUMINAMATH_CALUDE_max_difference_on_board_l1730_173022

/-- A type representing a 10x10 board with numbers from 1 to 100 -/
def Board := Fin 10 → Fin 10 → Fin 100

/-- A predicate that checks if a board is valid (each number appears exactly once) -/
def is_valid_board (b : Board) : Prop :=
  ∀ n : Fin 100, ∃! (i j : Fin 10), b i j = n

/-- The main theorem statement -/
theorem max_difference_on_board :
  ∀ b : Board, is_valid_board b →
    ∃ (i j k : Fin 10), 
      (i = k ∨ j = k) ∧ 
      ((b i j : ℕ) ≥ (b k j : ℕ) + 54 ∨ (b k j : ℕ) ≥ (b i j : ℕ) + 54) :=
by sorry

end NUMINAMATH_CALUDE_max_difference_on_board_l1730_173022


namespace NUMINAMATH_CALUDE_distribution_count_l1730_173002

/-- Represents the number of ways to distribute items between two people. -/
def distribute (pencils notebooks pens : Nat) : Nat :=
  let pencil_distributions := 3  -- (1,3), (2,2), (3,1)
  let notebook_distributions := 1  -- (1,1)
  let pen_distributions := 2  -- (1,2), (2,1)
  pencil_distributions * notebook_distributions * pen_distributions

/-- Theorem stating that the number of ways to distribute the given items is 6. -/
theorem distribution_count :
  ∀ (erasers : Nat), erasers > 0 → distribute 4 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_l1730_173002


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l1730_173073

theorem sum_remainder_mod_nine : (8357 + 8358 + 8359 + 8360 + 8361) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l1730_173073


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1730_173011

theorem fraction_equivalence : ∃ x : ℚ, (4 + x) / (7 + x) = 3 / 4 := by
  use 5
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1730_173011


namespace NUMINAMATH_CALUDE_johnny_tables_l1730_173012

/-- The number of tables that can be built given a total number of planks and planks required per table -/
def tables_built (total_planks : ℕ) (planks_per_table : ℕ) : ℕ :=
  total_planks / planks_per_table

/-- Theorem: Given 45 planks of wood and 9 planks required per table, 5 tables can be built -/
theorem johnny_tables : tables_built 45 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_johnny_tables_l1730_173012


namespace NUMINAMATH_CALUDE_average_time_per_km_l1730_173061

-- Define the race distance in kilometers
def race_distance : ℝ := 10

-- Define the time for the first half of the race in minutes
def first_half_time : ℝ := 20

-- Define the time for the second half of the race in minutes
def second_half_time : ℝ := 30

-- Theorem statement
theorem average_time_per_km (total_time : ℝ) (avg_time_per_km : ℝ) :
  total_time = first_half_time + second_half_time →
  avg_time_per_km = total_time / race_distance →
  avg_time_per_km = 5 := by
  sorry


end NUMINAMATH_CALUDE_average_time_per_km_l1730_173061


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l1730_173097

/-- Proves that if the cost price of 50 articles equals the selling price of 40 articles, 
    then the gain percent is 25%. -/
theorem gain_percent_calculation (C S : ℝ) 
  (h : 50 * C = 40 * S) : (S - C) / C * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l1730_173097


namespace NUMINAMATH_CALUDE_shopkeeper_cloth_sale_l1730_173027

/-- Proves the total amount a shopkeeper receives for selling cloth at a loss. -/
theorem shopkeeper_cloth_sale (total_metres : ℕ) (cost_price_per_metre : ℕ) (loss_per_metre : ℕ) : 
  total_metres = 600 →
  cost_price_per_metre = 70 →
  loss_per_metre = 10 →
  (total_metres * (cost_price_per_metre - loss_per_metre) : ℕ) = 36000 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_cloth_sale_l1730_173027


namespace NUMINAMATH_CALUDE_equation_proof_l1730_173032

theorem equation_proof : (36 / 18) * (36 / 72) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1730_173032


namespace NUMINAMATH_CALUDE_min_distance_to_circle_l1730_173072

/-- Line l in polar form -/
def line_l (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.cos θ + ρ * Real.sin θ + 4 = 0

/-- Circle C in Cartesian form -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*y = 0

/-- Distance between a point (ρ, θ) and its tangent to circle C -/
noncomputable def distance_to_tangent (ρ θ : ℝ) : ℝ :=
  sorry

/-- Theorem stating the minimum distance and its occurrence -/
theorem min_distance_to_circle (ρ θ : ℝ) :
  line_l ρ θ →
  distance_to_tangent ρ θ ≥ 2 ∧
  (distance_to_tangent ρ θ = 2 ↔ ρ = 2 ∧ θ = Real.pi) :=
  sorry

end NUMINAMATH_CALUDE_min_distance_to_circle_l1730_173072


namespace NUMINAMATH_CALUDE_rational_numbers_product_sum_negative_l1730_173028

theorem rational_numbers_product_sum_negative (x y : ℚ) 
  (h_product : x * y < 0) 
  (h_sum : x + y < 0) : 
  (abs x > abs y ∧ x < 0 ∧ y > 0) ∨ (abs y > abs x ∧ y < 0 ∧ x > 0) := by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_product_sum_negative_l1730_173028


namespace NUMINAMATH_CALUDE_problem_solution_l1730_173013

theorem problem_solution : 
  (Real.sqrt 48 - Real.sqrt 27 + Real.sqrt (1/3) = (4 * Real.sqrt 3) / 3) ∧
  ((Real.sqrt 5 - Real.sqrt 2) * (Real.sqrt 5 + Real.sqrt 2) - (Real.sqrt 3 - 1)^2 = 2 * Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1730_173013


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l1730_173075

theorem smallest_number_with_remainder_two (n : ℕ) : 
  (n % 3 = 2 ∧ n % 4 = 2 ∧ n % 6 = 2 ∧ n % 8 = 2) → n ≥ 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l1730_173075


namespace NUMINAMATH_CALUDE_equation_solution_l1730_173067

theorem equation_solution : ∃ x : ℚ, 3 * x + 6 = |(-19 + 5)| ∧ x = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1730_173067


namespace NUMINAMATH_CALUDE_product_of_real_parts_l1730_173051

theorem product_of_real_parts (x : ℂ) : 
  x^2 + 4*x = -1 + Complex.I → 
  (∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ = -1 + Complex.I ∧ x₂^2 + 4*x₂ = -1 + Complex.I ∧ 
    (x₁.re * x₂.re = (1 + 3 * Real.sqrt 10) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_l1730_173051


namespace NUMINAMATH_CALUDE_gcd_a4_3a2_1_a3_2a_eq_one_l1730_173058

theorem gcd_a4_3a2_1_a3_2a_eq_one (a : ℕ) : 
  Nat.gcd (a^4 + 3*a^2 + 1) (a^3 + 2*a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_a4_3a2_1_a3_2a_eq_one_l1730_173058


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l1730_173090

/-- The set of all numbers that can be represented as the sum of four consecutive positive integers -/
def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

/-- The greatest common divisor of all numbers in set B is 2 -/
theorem gcd_of_B_is_two : 
  ∃ (d : ℕ), d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l1730_173090


namespace NUMINAMATH_CALUDE_miss_adamson_class_size_l1730_173016

theorem miss_adamson_class_size :
  let num_classes : ℕ := 4
  let sheets_per_student : ℕ := 5
  let total_sheets : ℕ := 400
  let total_students : ℕ := total_sheets / sheets_per_student
  let students_per_class : ℕ := total_students / num_classes
  students_per_class = 20 := by
  sorry

end NUMINAMATH_CALUDE_miss_adamson_class_size_l1730_173016


namespace NUMINAMATH_CALUDE_weighted_average_is_70_55_l1730_173065

def mathematics_score : ℝ := 76
def science_score : ℝ := 65
def social_studies_score : ℝ := 82
def english_score : ℝ := 67
def biology_score : ℝ := 55
def computer_science_score : ℝ := 89
def history_score : ℝ := 74
def geography_score : ℝ := 63
def physics_score : ℝ := 78
def chemistry_score : ℝ := 71

def mathematics_weight : ℝ := 0.20
def science_weight : ℝ := 0.15
def social_studies_weight : ℝ := 0.10
def english_weight : ℝ := 0.15
def biology_weight : ℝ := 0.10
def computer_science_weight : ℝ := 0.05
def history_weight : ℝ := 0.05
def geography_weight : ℝ := 0.10
def physics_weight : ℝ := 0.05
def chemistry_weight : ℝ := 0.05

def weighted_average : ℝ :=
  mathematics_score * mathematics_weight +
  science_score * science_weight +
  social_studies_score * social_studies_weight +
  english_score * english_weight +
  biology_score * biology_weight +
  computer_science_score * computer_science_weight +
  history_score * history_weight +
  geography_score * geography_weight +
  physics_score * physics_weight +
  chemistry_score * chemistry_weight

theorem weighted_average_is_70_55 : weighted_average = 70.55 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_is_70_55_l1730_173065


namespace NUMINAMATH_CALUDE_matrix_commute_equality_l1730_173068

theorem matrix_commute_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) :
  A + B = A * B →
  A * B = ![![1, 2], ![3, 4]] →
  (A * B = B * A) →
  B * A = ![![1, 2], ![3, 4]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_commute_equality_l1730_173068


namespace NUMINAMATH_CALUDE_power_equation_solution_l1730_173070

theorem power_equation_solution (x : ℝ) : (1 / 8 : ℝ) * 2^36 = 4^x → x = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1730_173070


namespace NUMINAMATH_CALUDE_stadium_length_conversion_l1730_173025

/-- Converts yards to feet given the number of yards and the conversion factor. -/
def yards_to_feet (yards : ℕ) (conversion_factor : ℕ) : ℕ :=
  yards * conversion_factor

/-- Proves that 62 yards is equal to 186 feet when converted. -/
theorem stadium_length_conversion :
  let stadium_length_yards : ℕ := 62
  let yards_to_feet_conversion : ℕ := 3
  yards_to_feet stadium_length_yards yards_to_feet_conversion = 186 := by
  sorry

#check stadium_length_conversion

end NUMINAMATH_CALUDE_stadium_length_conversion_l1730_173025


namespace NUMINAMATH_CALUDE_multiplier_problem_l1730_173033

theorem multiplier_problem (x m : ℝ) (h1 : x = -10) (h2 : m * x - 8 = -12) : m = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l1730_173033


namespace NUMINAMATH_CALUDE_square_has_four_axes_of_symmetry_l1730_173060

-- Define the shapes
inductive Shape
  | Square
  | Rhombus
  | Rectangle
  | IsoscelesTrapezoid

-- Define a function to count axes of symmetry
def axesOfSymmetry (s : Shape) : Nat :=
  match s with
  | Shape.Square => 4
  | Shape.Rhombus => 2
  | Shape.Rectangle => 2
  | Shape.IsoscelesTrapezoid => 1

-- Theorem statement
theorem square_has_four_axes_of_symmetry :
  ∀ s : Shape, axesOfSymmetry s = 4 → s = Shape.Square := by
  sorry

#check square_has_four_axes_of_symmetry

end NUMINAMATH_CALUDE_square_has_four_axes_of_symmetry_l1730_173060


namespace NUMINAMATH_CALUDE_no_solution_and_inequality_solution_l1730_173095

theorem no_solution_and_inequality_solution :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → (x + 1) / (x - 1) + 4 / (1 - x^2) ≠ 1) ∧
  (∀ x : ℝ, 2 * (x - 1) ≥ x + 1 ∧ x - 2 > (2 * x - 1) / 3 ↔ x > 5) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_and_inequality_solution_l1730_173095


namespace NUMINAMATH_CALUDE_line_inclination_l1730_173096

theorem line_inclination (a : ℝ) : 
  (((2 - (-3)) / (1 - a) = Real.tan (135 * π / 180)) → a = 6) := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_l1730_173096
