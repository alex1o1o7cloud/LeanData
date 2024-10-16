import Mathlib

namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_7_l177_17720

def is_divisible_by_7 (n : ℕ) : Prop := ∃ k, n = 7 * k

def number_with_digit (x : ℕ) : ℕ := 5200 + 10 * x + 4

theorem smallest_digit_divisible_by_7 :
  (∃ x : ℕ, x ≤ 9 ∧ is_divisible_by_7 (number_with_digit x)) ∧
  (∀ y : ℕ, y < 2 → ¬is_divisible_by_7 (number_with_digit y)) ∧
  is_divisible_by_7 (number_with_digit 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_7_l177_17720


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l177_17705

theorem problems_left_to_grade (problems_per_worksheet : ℕ) (total_worksheets : ℕ) (graded_worksheets : ℕ) : 
  problems_per_worksheet = 4 →
  total_worksheets = 16 →
  graded_worksheets = 8 →
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 32 := by
  sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l177_17705


namespace NUMINAMATH_CALUDE_max_x_value_l177_17729

theorem max_x_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x - 2 * Real.sqrt y = Real.sqrt (2 * x - y)) : 
  (∀ z : ℝ, z > 0 ∧ ∃ w : ℝ, w > 0 ∧ z - 2 * Real.sqrt w = Real.sqrt (2 * z - w) → z ≤ 10) :=
sorry

end NUMINAMATH_CALUDE_max_x_value_l177_17729


namespace NUMINAMATH_CALUDE_power_function_through_point_l177_17715

/-- Given a power function that passes through the point (2, 8), prove its equation is x^3 -/
theorem power_function_through_point (n : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = x^n) → f 2 = 8 → (∀ x, f x = x^3) := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l177_17715


namespace NUMINAMATH_CALUDE_complex_equation_solution_l177_17780

theorem complex_equation_solution (b : ℝ) : (1 + b * Complex.I) * Complex.I = 1 + Complex.I → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l177_17780


namespace NUMINAMATH_CALUDE_sequence_a_formula_sequence_a_first_term_sequence_a_second_term_sequence_a_third_term_sequence_a_fourth_term_l177_17756

def sequence_a (n : ℕ) : ℚ := (18 * n - 9) / (7 * (10^n - 1))

theorem sequence_a_formula (n : ℕ) : 
  sequence_a n = (18 * n - 9) / (7 * (10^n - 1)) :=
by sorry

theorem sequence_a_first_term : sequence_a 1 = 1 / 7 :=
by sorry

theorem sequence_a_second_term : sequence_a 2 = 3 / 77 :=
by sorry

theorem sequence_a_third_term : sequence_a 3 = 5 / 777 :=
by sorry

theorem sequence_a_fourth_term : sequence_a 4 = 7 / 7777 :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_formula_sequence_a_first_term_sequence_a_second_term_sequence_a_third_term_sequence_a_fourth_term_l177_17756


namespace NUMINAMATH_CALUDE_cookies_percentage_increase_l177_17786

def cookies_problem (monday tuesday wednesday : ℕ) : Prop :=
  monday = 5 ∧
  tuesday = 2 * monday ∧
  wednesday > tuesday ∧
  monday + tuesday + wednesday = 29

theorem cookies_percentage_increase :
  ∀ monday tuesday wednesday : ℕ,
  cookies_problem monday tuesday wednesday →
  (wednesday - tuesday : ℚ) / tuesday * 100 = 40 :=
by sorry

end NUMINAMATH_CALUDE_cookies_percentage_increase_l177_17786


namespace NUMINAMATH_CALUDE_chess_players_count_l177_17787

theorem chess_players_count : ℕ :=
  let total_players : ℕ := 40
  let never_lost_fraction : ℚ := 1/4
  let lost_at_least_once : ℕ := 30
  have h1 : (1 - never_lost_fraction) * total_players = lost_at_least_once := by sorry
  have h2 : never_lost_fraction * total_players + lost_at_least_once = total_players := by sorry
  total_players

end NUMINAMATH_CALUDE_chess_players_count_l177_17787


namespace NUMINAMATH_CALUDE_parallelogram_reflection_l177_17718

-- Define the reflection across x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Define the reflection across x = 3
def reflect_x3 (p : ℝ × ℝ) : ℝ × ℝ := (6 - p.1, p.2)

-- Define the composition of both reflections
def double_reflect (p : ℝ × ℝ) : ℝ × ℝ := reflect_x3 (reflect_x p)

theorem parallelogram_reflection :
  double_reflect (4, 1) = (2, -1) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_reflection_l177_17718


namespace NUMINAMATH_CALUDE_ratio_of_sums_l177_17714

theorem ratio_of_sums (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l177_17714


namespace NUMINAMATH_CALUDE_third_pile_balls_l177_17790

theorem third_pile_balls (a b c : ℕ) (x : ℕ) :
  a + b + c = 2012 →
  b - x = 17 →
  a - x = 2 * (c - x) →
  c = 665 := by
sorry

end NUMINAMATH_CALUDE_third_pile_balls_l177_17790


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l177_17736

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: A point with negative x-coordinate and positive y-coordinate is in the second quadrant -/
theorem point_in_second_quadrant (p : Point) (hx : p.x < 0) (hy : p.y > 0) :
  SecondQuadrant p := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l177_17736


namespace NUMINAMATH_CALUDE_largest_root_divisibility_l177_17761

theorem largest_root_divisibility (a : ℝ) : 
  (a^3 - 3*a^2 + 1 = 0) →
  (∀ x : ℝ, x^3 - 3*x^2 + 1 = 0 → x ≤ a) →
  (17 ∣ ⌊a^1788⌋) ∧ (17 ∣ ⌊a^1988⌋) := by
sorry

end NUMINAMATH_CALUDE_largest_root_divisibility_l177_17761


namespace NUMINAMATH_CALUDE_fish_population_estimate_l177_17706

theorem fish_population_estimate 
  (tagged_june : ℕ) 
  (caught_october : ℕ) 
  (tagged_october : ℕ) 
  (death_migration_rate : ℚ) 
  (new_fish_rate : ℚ) 
  (h1 : tagged_june = 50) 
  (h2 : caught_october = 80) 
  (h3 : tagged_october = 4) 
  (h4 : death_migration_rate = 30/100) 
  (h5 : new_fish_rate = 35/100) : 
  ℕ := by
  sorry

#check fish_population_estimate

end NUMINAMATH_CALUDE_fish_population_estimate_l177_17706


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l177_17743

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The total amount Sasha has in dollars -/
def total_amount : ℚ := 48 / 10

theorem max_quarters_sasha (q : ℕ) : 
  q * quarter_value + (2 * q) * dime_value ≤ total_amount → 
  q ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l177_17743


namespace NUMINAMATH_CALUDE_fraction_ordering_l177_17722

theorem fraction_ordering : (6 : ℚ) / 29 < 8 / 25 ∧ 8 / 25 < 11 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l177_17722


namespace NUMINAMATH_CALUDE_no_eighteen_consecutive_good_numbers_l177_17778

/-- A natural number is good if it has exactly two prime divisors. -/
def IsGood (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ (∀ r : ℕ, Prime r → r ∣ n → r = p ∨ r = q)

/-- Theorem: It is impossible for 18 consecutive natural numbers to all be good. -/
theorem no_eighteen_consecutive_good_numbers :
  ¬∃ start : ℕ, ∀ i : ℕ, i < 18 → IsGood (start + i) := by
  sorry

end NUMINAMATH_CALUDE_no_eighteen_consecutive_good_numbers_l177_17778


namespace NUMINAMATH_CALUDE_circle_equation_l177_17739

/-- Given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 4 = 0

/-- Tangent line -/
def tangent_line (y : ℝ) : Prop :=
  y = 0

/-- Possible equations of the sought circle -/
def sought_circle (x y : ℝ) : Prop :=
  ((x - 2 - 2*Real.sqrt 10)^2 + (y - 4)^2 = 16) ∨
  ((x - 2 + 2*Real.sqrt 10)^2 + (y - 4)^2 = 16) ∨
  ((x - 2 - 2*Real.sqrt 6)^2 + (y + 4)^2 = 16) ∨
  ((x - 2 + 2*Real.sqrt 6)^2 + (y + 4)^2 = 16)

/-- Theorem stating the properties of the sought circle -/
theorem circle_equation :
  ∃ (a b : ℝ), 
    (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = 16) ∧
    (∃ (x y : ℝ), given_circle x y ∧ (x - a)^2 + (y - b)^2 = 36) ∧
    (∃ y : ℝ, tangent_line y ∧ (a - a)^2 + (y - b)^2 = 16) →
    sought_circle a b :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l177_17739


namespace NUMINAMATH_CALUDE_smallest_k_for_negative_three_in_range_l177_17702

-- Define the function g
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + k

-- State the theorem
theorem smallest_k_for_negative_three_in_range :
  (∃ k₀ : ℝ, (∀ k : ℝ, (∃ x : ℝ, g k x = -3) → k ≥ k₀) ∧
             (∃ x : ℝ, g k₀ x = -3) ∧
             k₀ = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_negative_three_in_range_l177_17702


namespace NUMINAMATH_CALUDE_gcf_of_12_and_16_l177_17748

theorem gcf_of_12_and_16 (n : ℕ) : 
  n = 12 → Nat.lcm n 16 = 48 → Nat.gcd n 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_12_and_16_l177_17748


namespace NUMINAMATH_CALUDE_inequality_proof_l177_17707

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (x / (x + y*z)) + (y / (y + z*x)) + (z / (z + x*y)) ≤ 2 / (1 - 3*x*y*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l177_17707


namespace NUMINAMATH_CALUDE_minimal_sum_of_squares_l177_17782

theorem minimal_sum_of_squares (a b c : ℕ+) : 
  a ≠ b → b ≠ c → a ≠ c →
  ∃ p q r : ℕ+, (a + b : ℕ) = p^2 ∧ (b + c : ℕ) = q^2 ∧ (a + c : ℕ) = r^2 →
  (a : ℕ) + b + c ≥ 55 :=
sorry

end NUMINAMATH_CALUDE_minimal_sum_of_squares_l177_17782


namespace NUMINAMATH_CALUDE_simplify_expression_l177_17754

theorem simplify_expression (a b : ℝ) (hb : b ≠ 0) (ha : a ≠ b^(1/3)) :
  (a^3 - b^3) / (a * b) - (a * b - b^2) / (a * b - a^3) = (a^2 + a * b + b^2) / b :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l177_17754


namespace NUMINAMATH_CALUDE_percentage_problem_l177_17755

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 400) : 1.2 * x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l177_17755


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l177_17701

-- 1
theorem problem_1 : 4 - (-28) + (-2) = 30 := by sorry

-- 2
theorem problem_2 : (-3) * ((-2/5) / (-1/4)) = -24/5 := by sorry

-- 3
theorem problem_3 : (-42) / (-7) - (-6) * 4 = 30 := by sorry

-- 4
theorem problem_4 : -3^2 / (-3)^2 + 3 * (-2) + |(-4)| = -3 := by sorry

-- 5
theorem problem_5 : (-24) * (3/4 - 5/6 + 7/12) = -12 := by sorry

-- 6
theorem problem_6 : -1^4 - (1 - 0.5) / (5/2) * (1/5) = -26/25 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l177_17701


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l177_17737

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l177_17737


namespace NUMINAMATH_CALUDE_james_stickers_l177_17741

theorem james_stickers (initial_stickers new_stickers total_stickers : ℕ) 
  (h1 : new_stickers = 22)
  (h2 : total_stickers = 61)
  (h3 : total_stickers = initial_stickers + new_stickers) : 
  initial_stickers = 39 := by
  sorry

end NUMINAMATH_CALUDE_james_stickers_l177_17741


namespace NUMINAMATH_CALUDE_no_consistent_solution_l177_17795

-- Define the types for teams and match results
inductive Team : Type
| Spartak | Dynamo | Zenit | Lokomotiv

structure MatchResult :=
(winner : Team)
(loser : Team)

-- Define the problem setup
def problem_setup (match1 match2 : MatchResult) (fan_count : Team → ℕ) : Prop :=
  match1.winner ≠ match1.loser ∧ 
  match2.winner ≠ match2.loser ∧
  match1.winner ≠ match2.winner ∧
  (fan_count Team.Spartak + fan_count match1.loser + fan_count match2.loser = 200) ∧
  (fan_count Team.Dynamo + fan_count match1.loser + fan_count match2.loser = 300) ∧
  (fan_count Team.Zenit = 500) ∧
  (fan_count Team.Lokomotiv = 600)

-- Theorem statement
theorem no_consistent_solution :
  ∀ (match1 match2 : MatchResult) (fan_count : Team → ℕ),
  problem_setup match1 match2 fan_count → False :=
sorry

end NUMINAMATH_CALUDE_no_consistent_solution_l177_17795


namespace NUMINAMATH_CALUDE_rex_driving_lessons_l177_17779

theorem rex_driving_lessons 
  (total_hours : ℕ) 
  (hours_per_week : ℕ) 
  (remaining_weeks : ℕ) 
  (h1 : total_hours = 40)
  (h2 : hours_per_week = 4)
  (h3 : remaining_weeks = 4) :
  total_hours - (remaining_weeks * hours_per_week) = 6 * hours_per_week :=
by sorry

end NUMINAMATH_CALUDE_rex_driving_lessons_l177_17779


namespace NUMINAMATH_CALUDE_oil_barrels_problem_l177_17773

/-- The minimum number of barrels needed to contain a given amount of oil -/
def min_barrels (total_oil : ℕ) (barrel_capacity : ℕ) : ℕ :=
  (total_oil + barrel_capacity - 1) / barrel_capacity

/-- Proof that at least 7 barrels are needed for 250 kg of oil with 40 kg capacity barrels -/
theorem oil_barrels_problem :
  min_barrels 250 40 = 7 := by
  sorry

end NUMINAMATH_CALUDE_oil_barrels_problem_l177_17773


namespace NUMINAMATH_CALUDE_total_bad_produce_l177_17772

-- Define the number of carrots and tomatoes picked by each person
def vanessa_carrots : ℕ := 17
def vanessa_tomatoes : ℕ := 12
def mom_carrots : ℕ := 14
def mom_tomatoes : ℕ := 22
def brother_carrots : ℕ := 6
def brother_tomatoes : ℕ := 8

-- Define the number of good carrots and tomatoes
def good_carrots : ℕ := 28
def good_tomatoes : ℕ := 35

-- Define the total number of carrots and tomatoes picked
def total_carrots : ℕ := vanessa_carrots + mom_carrots + brother_carrots
def total_tomatoes : ℕ := vanessa_tomatoes + mom_tomatoes + brother_tomatoes

-- Define the number of bad carrots and tomatoes
def bad_carrots : ℕ := total_carrots - good_carrots
def bad_tomatoes : ℕ := total_tomatoes - good_tomatoes

-- Theorem to prove
theorem total_bad_produce : bad_carrots + bad_tomatoes = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_bad_produce_l177_17772


namespace NUMINAMATH_CALUDE_janet_rent_advance_l177_17750

/-- Given Janet's apartment rental situation, prove the number of months' rent required in advance. -/
theorem janet_rent_advance (savings : ℕ) (monthly_rent : ℕ) (deposit : ℕ) (additional_needed : ℕ)
  (h1 : savings = 2225)
  (h2 : monthly_rent = 1250)
  (h3 : deposit = 500)
  (h4 : additional_needed = 775) :
  (savings + additional_needed - deposit) / monthly_rent = 2 := by
  sorry

end NUMINAMATH_CALUDE_janet_rent_advance_l177_17750


namespace NUMINAMATH_CALUDE_game_result_l177_17703

theorem game_result (a : ℝ) : ((2 * a + 6) / 2) - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l177_17703


namespace NUMINAMATH_CALUDE_K_travel_time_l177_17725

/-- K's travel time for 50 miles given the conditions -/
theorem K_travel_time (x : ℝ) (h1 : x > 0) : 
  ∃ (y : ℝ), y > 0 ∧ 
  50 / (x - 1/2) - 50 / x = 3/4 ∧ 
  50 / x = y :=
sorry

end NUMINAMATH_CALUDE_K_travel_time_l177_17725


namespace NUMINAMATH_CALUDE_alex_paper_distribution_l177_17767

/-- The number of ways to distribute n distinct items to m recipients,
    where each recipient can receive multiple items. -/
def distribution_ways (n m : ℕ) : ℕ := m^n

/-- The problem statement -/
theorem alex_paper_distribution :
  distribution_ways 5 10 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_alex_paper_distribution_l177_17767


namespace NUMINAMATH_CALUDE_imaginary_unit_multiplication_l177_17775

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_unit_multiplication :
  i * (1 - i) = 1 + i := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_multiplication_l177_17775


namespace NUMINAMATH_CALUDE_determinant_transformation_l177_17791

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 9 →
  Matrix.det !![2*p, 5*p + 4*q; 2*r, 5*r + 4*s] = 72 := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l177_17791


namespace NUMINAMATH_CALUDE_percentage_of_fresh_peaches_l177_17764

def total_peaches : ℕ := 250
def thrown_away : ℕ := 15
def peaches_left : ℕ := 135

def fresh_peaches : ℕ := total_peaches - (thrown_away + (total_peaches - peaches_left))

theorem percentage_of_fresh_peaches :
  (fresh_peaches : ℚ) / total_peaches * 100 = 48 := by sorry

end NUMINAMATH_CALUDE_percentage_of_fresh_peaches_l177_17764


namespace NUMINAMATH_CALUDE_particle_position_after_3045_minutes_l177_17713

/-- Represents the position of a particle -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Calculates the time taken for n rectangles -/
def timeForNRectangles (n : ℕ) : ℕ :=
  (n + 1)^2 - 1

/-- Calculates the position after n complete rectangles -/
def positionAfterNRectangles (n : ℕ) : Position :=
  if n % 2 = 0 then
    ⟨0, n⟩
  else
    ⟨0, n⟩

/-- Calculates the final position after given time -/
def finalPosition (time : ℕ) : Position :=
  let n := (Nat.sqrt (time + 1) : ℕ) - 1
  let remainingTime := time - timeForNRectangles n
  let basePosition := positionAfterNRectangles n
  if n % 2 = 0 then
    ⟨basePosition.x + remainingTime, basePosition.y⟩
  else
    ⟨basePosition.x + remainingTime, basePosition.y⟩

theorem particle_position_after_3045_minutes :
  finalPosition 3045 = ⟨21, 54⟩ := by
  sorry


end NUMINAMATH_CALUDE_particle_position_after_3045_minutes_l177_17713


namespace NUMINAMATH_CALUDE_square_root_product_squared_l177_17749

theorem square_root_product_squared : (Real.sqrt 900 * Real.sqrt 784)^2 = 705600 := by
  sorry

end NUMINAMATH_CALUDE_square_root_product_squared_l177_17749


namespace NUMINAMATH_CALUDE_fibonacci_mod_127_l177_17717

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_mod_127 :
  (∀ m : ℕ, m < 256 → (fib m % 127 ≠ 0 ∨ fib (m + 1) % 127 ≠ 1)) ∧
  fib 256 % 127 = 0 ∧ fib 257 % 127 = 1 :=
sorry

end NUMINAMATH_CALUDE_fibonacci_mod_127_l177_17717


namespace NUMINAMATH_CALUDE_equation_solution_l177_17716

theorem equation_solution (x y : ℝ) : 
  (4 * x + y = 9) → (y = 9 - 4 * x) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l177_17716


namespace NUMINAMATH_CALUDE_problem_solution_l177_17719

def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {-a, a^2 + 3}

theorem problem_solution (a : ℝ) : A ∪ B a = {1, 2, 4} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l177_17719


namespace NUMINAMATH_CALUDE_sandys_initial_money_l177_17752

/-- Sandy's shopping problem -/
theorem sandys_initial_money 
  (shirt_cost : ℝ) 
  (jacket_cost : ℝ) 
  (pocket_money : ℝ) 
  (h1 : shirt_cost = 12.14)
  (h2 : jacket_cost = 9.28)
  (h3 : pocket_money = 7.43) :
  shirt_cost + jacket_cost + pocket_money = 28.85 := by
sorry

end NUMINAMATH_CALUDE_sandys_initial_money_l177_17752


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l177_17732

-- Define the conditions p and q
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, not_p x → not_q x) ∧ 
  ¬(∀ x, not_q x → not_p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l177_17732


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l177_17777

theorem gcd_digits_bound (a b : ℕ) (ha : 1000000 ≤ a ∧ a < 10000000) (hb : 1000000 ≤ b ∧ b < 10000000)
  (hlcm : Nat.lcm a b < 100000000000) : Nat.gcd a b < 1000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l177_17777


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l177_17738

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define a point on the hyperbola
def on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (h1 : on_hyperbola A) 
  (h2 : on_hyperbola B) 
  (h3 : A.1 < 0 ∧ B.1 < 0)  -- A and B are on the left branch
  (h4 : ∃ t : ℝ, A.1 = (1 - t) * left_focus.1 + t * B.1 ∧ 
               A.2 = (1 - t) * left_focus.2 + t * B.2)  -- AB passes through left focus
  (h5 : distance A B = 5) :
  distance A right_focus + distance B right_focus + distance A B = 26 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l177_17738


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l177_17734

/-- A random variable following a normal distribution -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability function for the normal distribution -/
noncomputable def prob (X : NormalDistribution) (a b : ℝ) : ℝ :=
  sorry

theorem normal_distribution_probability (X : NormalDistribution) 
  (h1 : X.μ = 4)
  (h2 : X.σ = 1)
  (h3 : prob X (X.μ - 2*X.σ) (X.μ + 2*X.σ) = 0.9544)
  (h4 : prob X (X.μ - X.σ) (X.μ + X.σ) = 0.6826) :
  prob X 5 6 = 0.1359 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l177_17734


namespace NUMINAMATH_CALUDE_knicks_win_probability_l177_17760

/-- The probability of the Bulls winning a single game -/
def p : ℚ := 3/4

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The maximum number of games in the series -/
def max_games : ℕ := 2 * games_to_win - 1

/-- The probability of the Knicks winning the series in exactly 7 games -/
def knicks_win_in_seven : ℚ := 135/4096

theorem knicks_win_probability :
  knicks_win_in_seven = (Nat.choose 6 3 : ℚ) * (1 - p)^3 * p^3 * (1 - p) :=
sorry

end NUMINAMATH_CALUDE_knicks_win_probability_l177_17760


namespace NUMINAMATH_CALUDE_total_cds_on_shelf_l177_17796

/-- The number of CDs that a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- The number of racks that can fit on a shelf -/
def racks_per_shelf : ℕ := 4

/-- Theorem: The total number of CDs that can fit on a shelf is 32 -/
theorem total_cds_on_shelf : cds_per_rack * racks_per_shelf = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_cds_on_shelf_l177_17796


namespace NUMINAMATH_CALUDE_f_properties_l177_17793

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (2*a + 1)*x + a * log x

theorem f_properties (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((2*x - (2*a + 1) + a/x) : ℝ) x) ∧
  (HasDerivAt (f a) 0 1 ↔ a = 1) ∧
  (∀ x > 1, f a x > 0 ↔ a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l177_17793


namespace NUMINAMATH_CALUDE_barrel_capacity_l177_17785

/-- Represents a barrel with two taps -/
structure Barrel :=
  (capacity : ℝ)
  (midwayTapRate : ℝ) -- Liters per minute
  (bottomTapRate : ℝ) -- Liters per minute

/-- Represents the scenario of drawing beer from the barrel -/
def drawBeer (barrel : Barrel) (earlyUseTime : ℝ) (assistantUseTime : ℝ) : Prop :=
  -- The capacity is twice the amount drawn early plus the amount drawn by the assistant
  barrel.capacity = 2 * (earlyUseTime * barrel.midwayTapRate + assistantUseTime * barrel.bottomTapRate)

/-- The main theorem stating the capacity of the barrel -/
theorem barrel_capacity : ∃ (b : Barrel), 
  b.midwayTapRate = 1 / 6 ∧ 
  b.bottomTapRate = 1 / 4 ∧ 
  drawBeer b 24 16 ∧ 
  b.capacity = 16 := by
  sorry

end NUMINAMATH_CALUDE_barrel_capacity_l177_17785


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_min_value_f_l177_17730

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7 ∨ x > 5/3} := by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_min_value_f_l177_17730


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_sqrt5_l177_17726

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_magnitude_equals_sqrt5 : 
  Complex.abs (2 + i^2 + 2*i^3) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_sqrt5_l177_17726


namespace NUMINAMATH_CALUDE_infinite_series_sum_l177_17794

/-- The sum of the infinite series ∑(k=1 to ∞) [6^k / ((3^k - 2^k)(3^(k+1) - 2^(k+1)))] is equal to 2. -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (6:ℝ)^k / ((3:ℝ)^k - (2:ℝ)^k * ((3:ℝ)^(k+1) - (2:ℝ)^(k+1)))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l177_17794


namespace NUMINAMATH_CALUDE_outfit_combinations_l177_17762

def num_shirts : ℕ := 5
def num_pants : ℕ := 6
def restricted_combinations : ℕ := 2

theorem outfit_combinations :
  let total_combinations := num_shirts * num_pants
  let restricted_shirt_combinations := num_pants - restricted_combinations
  let unrestricted_combinations := (num_shirts - 1) * num_pants
  unrestricted_combinations + restricted_shirt_combinations = 28 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l177_17762


namespace NUMINAMATH_CALUDE_circle_of_students_l177_17768

theorem circle_of_students (n : ℕ) (h : n > 0) :
  (∃ (a b : ℕ), a < n ∧ b < n ∧ a = 6 ∧ b = 16 ∧ (b - a) * 2 + 2 = n) →
  n = 22 :=
by sorry

end NUMINAMATH_CALUDE_circle_of_students_l177_17768


namespace NUMINAMATH_CALUDE_rent_increase_new_mean_l177_17792

theorem rent_increase_new_mean 
  (num_friends : ℕ) 
  (initial_average : ℝ) 
  (increased_rent : ℝ) 
  (increase_percentage : ℝ) : 
  num_friends = 4 → 
  initial_average = 800 → 
  increased_rent = 800 → 
  increase_percentage = 0.25 → 
  (num_friends * initial_average + increased_rent * increase_percentage) / num_friends = 850 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_new_mean_l177_17792


namespace NUMINAMATH_CALUDE_election_votes_l177_17776

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 : ℚ) / 100 * total_votes - (38 : ℚ) / 100 * total_votes = 408) :
  (62 : ℚ) / 100 * total_votes = 1054 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l177_17776


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l177_17721

theorem triangle_abc_problem (a b c A B C : ℝ) 
  (h1 : a * Real.sin A = 4 * b * Real.sin B)
  (h2 : a * c = Real.sqrt 5 * (a^2 - b^2 - c^2)) :
  Real.cos A = -(Real.sqrt 5) / 5 ∧ 
  Real.sin (2 * B - A) = -2 * (Real.sqrt 5) / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_problem_l177_17721


namespace NUMINAMATH_CALUDE_subtract_to_one_l177_17742

theorem subtract_to_one : ∃ x : ℤ, (-5) - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_subtract_to_one_l177_17742


namespace NUMINAMATH_CALUDE_exists_odd_white_square_l177_17757

/-- Represents a cell color in the grid -/
inductive CellColor
| Black
| White

/-- Represents the 200×200 grid -/
def Grid := Fin 200 → Fin 200 → CellColor

/-- Counts the number of cells with the given color in the grid -/
def countCells (g : Grid) (c : CellColor) : ℕ := sorry

/-- Checks if a 2×2 square at the given position has an odd number of white cells -/
def hasOddWhiteSquare (g : Grid) (row col : Fin 199) : Prop := sorry

/-- The main theorem to prove -/
theorem exists_odd_white_square (g : Grid) 
  (h : countCells g CellColor.Black = countCells g CellColor.White + 404) :
  ∃ (row col : Fin 199), hasOddWhiteSquare g row col := sorry

end NUMINAMATH_CALUDE_exists_odd_white_square_l177_17757


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l177_17789

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 2 * Nat.factorial 5 = 36120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l177_17789


namespace NUMINAMATH_CALUDE_equal_volume_implies_equal_breadth_l177_17765

/-- Represents the volume of earth dug in a project -/
structure EarthVolume where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of earth dug -/
def calculateVolume (v : EarthVolume) : ℝ :=
  v.depth * v.length * v.breadth

theorem equal_volume_implies_equal_breadth 
  (project1 : EarthVolume)
  (project2 : EarthVolume)
  (h1 : project1.depth = 100)
  (h2 : project1.length = 25)
  (h3 : project1.breadth = 30)
  (h4 : project2.depth = 75)
  (h5 : project2.length = 20)
  (h6 : calculateVolume project1 = calculateVolume project2) :
  project2.breadth = 50 := by
sorry

end NUMINAMATH_CALUDE_equal_volume_implies_equal_breadth_l177_17765


namespace NUMINAMATH_CALUDE_union_complement_equality_l177_17783

def U : Finset Nat := {1, 2, 3, 4, 5}
def M : Finset Nat := {1, 4}
def N : Finset Nat := {2, 5}

theorem union_complement_equality : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l177_17783


namespace NUMINAMATH_CALUDE_second_week_cut_percentage_sculpture_problem_l177_17709

/-- Calculates the percentage of marble cut away in the second week of sculpting -/
theorem second_week_cut_percentage (initial_weight : ℝ) (first_week_cut : ℝ) 
  (third_week_cut : ℝ) (final_weight : ℝ) : ℝ :=
  let remaining_after_first := initial_weight * (1 - first_week_cut / 100)
  let second_week_cut := 100 * (1 - (final_weight / (remaining_after_first * (1 - third_week_cut / 100))))
  second_week_cut

/-- The percentage of marble cut away in the second week is 30% -/
theorem sculpture_problem :
  second_week_cut_percentage 300 30 15 124.95 = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_week_cut_percentage_sculpture_problem_l177_17709


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_one_l177_17771

theorem sum_of_roots_eq_one : 
  ∃ (x₁ x₂ : ℝ), (x₁ + 3) * (x₁ - 4) = 22 ∧ 
                 (x₂ + 3) * (x₂ - 4) = 22 ∧ 
                 x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_one_l177_17771


namespace NUMINAMATH_CALUDE_assembly_line_average_output_l177_17798

/-- Represents the production data for an assembly line phase -/
structure ProductionPhase where
  cogs_produced : ℕ
  production_rate : ℕ

/-- Calculates the time taken for a production phase in hours -/
def time_taken (phase : ProductionPhase) : ℚ :=
  phase.cogs_produced / phase.production_rate

/-- Calculates the overall average output for two production phases -/
def overall_average_output (phase1 phase2 : ProductionPhase) : ℚ :=
  (phase1.cogs_produced + phase2.cogs_produced) / (time_taken phase1 + time_taken phase2)

/-- Theorem stating that the overall average output is 30 cogs per hour -/
theorem assembly_line_average_output :
  let phase1 : ProductionPhase := { cogs_produced := 60, production_rate := 20 }
  let phase2 : ProductionPhase := { cogs_produced := 60, production_rate := 60 }
  overall_average_output phase1 phase2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_assembly_line_average_output_l177_17798


namespace NUMINAMATH_CALUDE_root_product_equals_eight_l177_17769

theorem root_product_equals_eight :
  (32 : ℝ) ^ (1/5 : ℝ) * (8 : ℝ) ^ (1/3 : ℝ) * (4 : ℝ) ^ (1/2 : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_eight_l177_17769


namespace NUMINAMATH_CALUDE_or_not_implies_other_l177_17711

theorem or_not_implies_other (p q : Prop) : (p ∨ q) → ¬p → q := by sorry

end NUMINAMATH_CALUDE_or_not_implies_other_l177_17711


namespace NUMINAMATH_CALUDE_stratified_sample_grade12_l177_17745

/-- Calculates the number of grade 12 students to be selected in a stratified sample -/
theorem stratified_sample_grade12 (total : ℕ) (grade10 : ℕ) (grade11 : ℕ) (sample_size : ℕ) :
  total = 1500 →
  grade10 = 550 →
  grade11 = 450 →
  sample_size = 300 →
  (sample_size * (total - grade10 - grade11)) / total = 100 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_grade12_l177_17745


namespace NUMINAMATH_CALUDE_parallel_linear_functions_min_value_l177_17710

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

/-- The minimum value of a quadratic function -/
def quadratic_min (h : ℝ → ℝ) : ℝ := sorry

theorem parallel_linear_functions_min_value 
  (funcs : ParallelLinearFunctions) 
  (h_min : quadratic_min (λ x => (funcs.f x)^2 + 2 * funcs.g x) = 5) :
  quadratic_min (λ x => (funcs.g x)^2 + 2 * funcs.f x) = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_linear_functions_min_value_l177_17710


namespace NUMINAMATH_CALUDE_min_value_f_plus_f_deriv_l177_17724

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

-- Theorem statement
theorem min_value_f_plus_f_deriv (a : ℝ) :
  (∃ m : ℝ, f_deriv a m = 0 ∧ m = 1) →
  (∃ min_val : ℝ, 
    ∀ m n : ℝ, m ∈ Set.Icc (-1 : ℝ) 1 → n ∈ Set.Icc (-1 : ℝ) 1 → 
    f a m + f_deriv a n ≥ min_val ∧
    (∃ m' n' : ℝ, m' ∈ Set.Icc (-1 : ℝ) 1 ∧ n' ∈ Set.Icc (-1 : ℝ) 1 ∧
      f a m' + f_deriv a n' = min_val)) ∧
  min_val = -13 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_plus_f_deriv_l177_17724


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l177_17704

/-- A quadratic function satisfying certain conditions -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The function g defined in terms of f and m -/
def g (a b c m : ℝ) : ℝ → ℝ := fun x ↦ f a b c x + 2 * (1 - m) * x

/-- The theorem statement -/
theorem quadratic_function_theorem (a b c : ℝ) :
  (∀ x, f a b c x ≥ 0) →
  f a b c 0 = 1 →
  f a b c 1 = 0 →
  (∃ m, (∀ x ∈ Set.Icc (-2 : ℝ) 5, g a b c m x ≤ 13) ∧
        (∃ x ∈ Set.Icc (-2 : ℝ) 5, g a b c m x = 13)) →
  ((a = 1 ∧ b = -2 ∧ c = 1) ∧ (m = 13/10 ∨ m = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l177_17704


namespace NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l177_17770

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 890 ways to distribute 7 distinguishable balls into 4 indistinguishable boxes -/
theorem distribute_seven_balls_four_boxes : distribute_balls 7 4 = 890 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l177_17770


namespace NUMINAMATH_CALUDE_lg_6_equals_a_plus_b_l177_17700

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_6_equals_a_plus_b (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) : lg 6 = a + b := by
  sorry

end NUMINAMATH_CALUDE_lg_6_equals_a_plus_b_l177_17700


namespace NUMINAMATH_CALUDE_intersection_condition_l177_17784

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l177_17784


namespace NUMINAMATH_CALUDE_perpendicular_slope_correct_l177_17747

-- Define the slope of the given line
def given_line_slope : ℚ := 3 / 4

-- Define the slope of the perpendicular line
def perpendicular_slope : ℚ := -4 / 3

-- Theorem stating that the perpendicular slope is correct
theorem perpendicular_slope_correct :
  perpendicular_slope = -1 / given_line_slope :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_correct_l177_17747


namespace NUMINAMATH_CALUDE_gilbert_basil_bushes_l177_17781

/-- The number of basil bushes Gilbert planted initially -/
def initial_basil_bushes : ℕ := 1

/-- The total number of herb plants at the end of spring -/
def total_plants : ℕ := 5

/-- The number of mint types (which were eaten) -/
def mint_types : ℕ := 2

/-- The number of parsley plants -/
def parsley_plants : ℕ := 1

/-- The number of extra basil plants that grew during spring -/
def extra_basil : ℕ := 1

theorem gilbert_basil_bushes :
  initial_basil_bushes = total_plants - mint_types - parsley_plants - extra_basil :=
by sorry

end NUMINAMATH_CALUDE_gilbert_basil_bushes_l177_17781


namespace NUMINAMATH_CALUDE_total_houses_l177_17731

theorem total_houses (garage : ℕ) (pool : ℕ) (both : ℕ) 
  (h1 : garage = 50) 
  (h2 : pool = 40) 
  (h3 : both = 35) : 
  garage + pool - both = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_houses_l177_17731


namespace NUMINAMATH_CALUDE_basketball_league_female_fraction_l177_17758

theorem basketball_league_female_fraction :
  -- Define variables
  let male_last_year : ℕ := 30
  let total_increase_rate : ℚ := 115 / 100
  let male_increase_rate : ℚ := 110 / 100
  let female_increase_rate : ℚ := 125 / 100

  -- Calculate values
  let male_this_year : ℚ := male_last_year * male_increase_rate
  let female_last_year : ℚ := (total_increase_rate * (male_last_year : ℚ) - male_this_year) / (female_increase_rate - total_increase_rate)
  let female_this_year : ℚ := female_last_year * female_increase_rate
  let total_this_year : ℚ := male_this_year + female_this_year

  -- Prove the fraction
  female_this_year / total_this_year = 25 / 69 := by sorry

end NUMINAMATH_CALUDE_basketball_league_female_fraction_l177_17758


namespace NUMINAMATH_CALUDE_best_route_is_D_l177_17763

-- Define the structure for a route
structure Route where
  name : String
  baseTime : ℕ
  numLights : ℕ
  redLightTime : ℕ
  trafficDensity : String
  weatherCondition : String
  roadCondition : String

-- Define the routes
def routeA : Route := {
  name := "A",
  baseTime := 10,
  numLights := 3,
  redLightTime := 3,
  trafficDensity := "moderate",
  weatherCondition := "light rain",
  roadCondition := "good"
}

def routeB : Route := {
  name := "B",
  baseTime := 12,
  numLights := 4,
  redLightTime := 2,
  trafficDensity := "high",
  weatherCondition := "clear",
  roadCondition := "pothole"
}

def routeC : Route := {
  name := "C",
  baseTime := 11,
  numLights := 2,
  redLightTime := 4,
  trafficDensity := "low",
  weatherCondition := "clear",
  roadCondition := "construction"
}

def routeD : Route := {
  name := "D",
  baseTime := 14,
  numLights := 0,
  redLightTime := 0,
  trafficDensity := "medium",
  weatherCondition := "potential fog",
  roadCondition := "unknown"
}

-- Define the list of all routes
def allRoutes : List Route := [routeA, routeB, routeC, routeD]

-- Calculate the worst-case travel time for a route
def worstCaseTime (r : Route) : ℕ := r.baseTime + r.numLights * r.redLightTime

-- Define the theorem
theorem best_route_is_D :
  ∀ r ∈ allRoutes, worstCaseTime routeD ≤ worstCaseTime r :=
sorry

end NUMINAMATH_CALUDE_best_route_is_D_l177_17763


namespace NUMINAMATH_CALUDE_marie_gift_boxes_l177_17753

/-- Represents the number of gift boxes Marie used to pack chocolate eggs. -/
def num_gift_boxes (total_eggs : ℕ) (egg_weight : ℕ) (remaining_weight : ℕ) : ℕ :=
  let total_weight := total_eggs * egg_weight
  let melted_weight := total_weight - remaining_weight
  let eggs_per_box := melted_weight / egg_weight
  total_eggs / eggs_per_box

/-- Proves that Marie packed the chocolate eggs in 4 gift boxes. -/
theorem marie_gift_boxes :
  num_gift_boxes 12 10 90 = 4 := by
  sorry

end NUMINAMATH_CALUDE_marie_gift_boxes_l177_17753


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l177_17723

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (containedIn : Line → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLineToPlane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (l : Line) (α β : Plane)
  (h1 : parallelPlanes α β)
  (h2 : containedIn l α) :
  parallelLineToPlane l β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l177_17723


namespace NUMINAMATH_CALUDE_problem_solution_l177_17728

theorem problem_solution (m n : ℝ) 
  (h1 : m * n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 44 + n^4)
  (h4 : m^5 + 5 = 11) :
  m^9 + n = 38 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l177_17728


namespace NUMINAMATH_CALUDE_expression_can_be_any_real_l177_17712

theorem expression_can_be_any_real (x : ℝ) : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  a + b + c = 1 ∧ 
  (a^4 + b^4 + c^4) / (a*b + b*c + c*a) = x :=
sorry

end NUMINAMATH_CALUDE_expression_can_be_any_real_l177_17712


namespace NUMINAMATH_CALUDE_roots_equal_condition_l177_17759

theorem roots_equal_condition (m : ℝ) : 
  (∃! x : ℝ, (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m) ↔ m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_roots_equal_condition_l177_17759


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l177_17797

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 5*x - 14 > 0) ∧
  (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 5*x - 14)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l177_17797


namespace NUMINAMATH_CALUDE_parabola_equation_l177_17751

def is_valid_parabola (p : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ),
    (2 * x₁ + 1)^2 = 2 * p * x₁ ∧
    (2 * x₂ + 1)^2 = 2 * p * x₂ ∧
    (x₁ - x₂)^2 * 5 = 15

theorem parabola_equation :
  ∀ p : ℝ, is_valid_parabola p → (p = -2 ∨ p = 6) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l177_17751


namespace NUMINAMATH_CALUDE_coin_toss_sequences_count_l177_17788

/-- The number of ways to distribute n indistinguishable balls into k distinguishable urns -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of different sequences of 18 coin tosses with specific subsequence counts -/
def coinTossSequences : ℕ :=
  let numTosses := 18
  let numHH := 3
  let numHT := 4
  let numTH := 5
  let numTT := 6
  let numTGaps := numTH + 1
  let numHGaps := numHT + 1
  let tDistributions := starsAndBars numTT numTGaps
  let hDistributions := starsAndBars numHH numHGaps
  tDistributions * hDistributions

theorem coin_toss_sequences_count : coinTossSequences = 4200 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_sequences_count_l177_17788


namespace NUMINAMATH_CALUDE_pens_per_student_l177_17744

theorem pens_per_student (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) 
  (h1 : total_pens = 1001)
  (h2 : total_pencils = 910)
  (h3 : max_students = 91) :
  total_pens / max_students = 11 := by
sorry

end NUMINAMATH_CALUDE_pens_per_student_l177_17744


namespace NUMINAMATH_CALUDE_percentage_of_big_bottles_sold_l177_17708

theorem percentage_of_big_bottles_sold
  (initial_small : ℕ)
  (initial_big : ℕ)
  (small_sold_percentage : ℚ)
  (total_remaining : ℕ)
  (h1 : initial_small = 6000)
  (h2 : initial_big = 15000)
  (h3 : small_sold_percentage = 11/100)
  (h4 : total_remaining = 18540)
  : ∃ (big_sold_percentage : ℚ),
    big_sold_percentage = 12/100 ∧
    total_remaining = initial_small - (small_sold_percentage * initial_small).floor +
                      initial_big - (big_sold_percentage * initial_big).floor :=
sorry

end NUMINAMATH_CALUDE_percentage_of_big_bottles_sold_l177_17708


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l177_17735

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (a^2 + 2*a - 3) + (a + 3)*Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l177_17735


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l177_17740

/-- The area of a stripe wrapped around a cylinder -/
theorem stripe_area_on_cylinder 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℝ) 
  (h1 : diameter = 20) 
  (h2 : stripe_width = 2) 
  (h3 : revolutions = 3) : 
  stripe_width * revolutions * (π * diameter) = 240 * π := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l177_17740


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l177_17733

/-- Represents the amount of pizza eaten by each sibling -/
structure PizzaConsumption where
  alex : Rat
  beth : Rat
  cyril : Rat
  dan : Rat
  eliza : Rat

/-- Checks if a list of rationals is in decreasing order -/
def isDecreasing (l : List Rat) : Prop :=
  ∀ i j, i < j → j < l.length → l[i]! ≥ l[j]!

/-- The main theorem stating the correct order of pizza consumption -/
theorem pizza_consumption_order (p : PizzaConsumption) 
  (h1 : p.alex = 1/6)
  (h2 : p.beth = 1/4)
  (h3 : p.cyril = 1/3)
  (h4 : p.dan = 0)
  (h5 : p.eliza = 1 - (p.alex + p.beth + p.cyril + p.dan)) :
  isDecreasing [p.cyril, p.beth, p.eliza, p.alex, p.dan] := by
  sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l177_17733


namespace NUMINAMATH_CALUDE_f_period_l177_17774

open Real

noncomputable def f (x : ℝ) : ℝ := 
  (sin (2 * x) + sin (2 * x + π / 3)) / (cos (2 * x) + cos (2 * x + π / 3))

theorem f_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ 
  T = π / 2 :=
sorry

end NUMINAMATH_CALUDE_f_period_l177_17774


namespace NUMINAMATH_CALUDE_ship_passengers_l177_17746

theorem ship_passengers :
  ∀ (P : ℕ),
  (P / 20 : ℚ) + (P / 15 : ℚ) + (P / 10 : ℚ) + (P / 12 : ℚ) + (P / 30 : ℚ) + 60 = P →
  P = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l177_17746


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l177_17727

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/y + 1 = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 2/b + 1 = 2 → 2*x + y ≤ 2*a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l177_17727


namespace NUMINAMATH_CALUDE_math_team_combinations_l177_17766

/-- The number of girls in the math club -/
def num_girls : ℕ := 4

/-- The number of boys in the math club -/
def num_boys : ℕ := 6

/-- The number of girls to be selected for the team -/
def girls_in_team : ℕ := 3

/-- The number of boys to be selected for the team -/
def boys_in_team : ℕ := 2

/-- The total number of possible team combinations -/
def total_combinations : ℕ := 60

theorem math_team_combinations :
  (Nat.choose num_girls girls_in_team) * (Nat.choose num_boys boys_in_team) = total_combinations :=
sorry

end NUMINAMATH_CALUDE_math_team_combinations_l177_17766


namespace NUMINAMATH_CALUDE_c_most_suitable_l177_17799

-- Define the structure for an athlete
structure Athlete where
  name : String
  average : ℝ
  variance : ℝ

-- Define the list of athletes
def athletes : List Athlete := [
  ⟨"A", 169, 6.0⟩,
  ⟨"B", 168, 17.3⟩,
  ⟨"C", 169, 5.0⟩,
  ⟨"D", 168, 19.5⟩
]

-- Function to determine if an athlete is suitable
def isSuitable (a : Athlete) : Prop :=
  ∀ b ∈ athletes, 
    a.average ≥ b.average ∧ 
    (a.average = b.average → a.variance ≤ b.variance)

-- Theorem stating that C is the most suitable candidate
theorem c_most_suitable : 
  ∃ c ∈ athletes, c.name = "C" ∧ isSuitable c :=
sorry

end NUMINAMATH_CALUDE_c_most_suitable_l177_17799
