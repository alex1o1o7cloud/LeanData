import Mathlib

namespace NUMINAMATH_CALUDE_theatre_fraction_l2780_278000

-- Define the total number of students
variable (T : ℕ)

-- Define the number of students in each elective
def P : ℕ := T / 2
def Th : ℕ := T / 6
def M : ℕ := T / 3

-- Theorem statement
theorem theatre_fraction (T : ℕ) (h : T > 0) :
  let remaining_PE := (2 : ℚ) / 3 * P T
  let remaining_Th := (3 : ℚ) / 4 * Th T
  let remaining_M := M T
  (remaining_PE + remaining_M) / T = (2 : ℚ) / 3 →
  (Th T : ℚ) / T = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_theatre_fraction_l2780_278000


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2780_278088

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 5 * x^2 + b₁ * x + 10 * x + 15 = 0 → (b₁ + 10)^2 = 300) ∧
  (∀ x, 5 * x^2 + b₂ * x + 10 * x + 15 = 0 → (b₂ + 10)^2 = 300) ∧
  (∀ b, (∀ x, 5 * x^2 + b * x + 10 * x + 15 = 0 → (b + 10)^2 = 300) → b = b₁ ∨ b = b₂) →
  b₁ + b₂ = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2780_278088


namespace NUMINAMATH_CALUDE_plumbing_equal_charge_time_l2780_278010

def pauls_visit_fee : ℚ := 55
def pauls_hourly_rate : ℚ := 35
def reliable_visit_fee : ℚ := 75
def reliable_hourly_rate : ℚ := 30

theorem plumbing_equal_charge_time : 
  ∃ h : ℚ, h > 0 ∧ (pauls_visit_fee + pauls_hourly_rate * h = reliable_visit_fee + reliable_hourly_rate * h) ∧ h = 4 := by
  sorry

end NUMINAMATH_CALUDE_plumbing_equal_charge_time_l2780_278010


namespace NUMINAMATH_CALUDE_green_peaches_count_l2780_278032

/-- The number of green peaches in a basket, given the total number of peaches and the number of red peaches. -/
def num_green_peaches (total : ℕ) (red : ℕ) : ℕ :=
  total - red

/-- Theorem stating that there are 3 green peaches in the basket. -/
theorem green_peaches_count :
  let total := 16
  let red := 13
  num_green_peaches total red = 3 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2780_278032


namespace NUMINAMATH_CALUDE_power_mean_inequality_l2780_278001

theorem power_mean_inequality (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2)^n :=
by sorry

end NUMINAMATH_CALUDE_power_mean_inequality_l2780_278001


namespace NUMINAMATH_CALUDE_mini_train_length_l2780_278092

/-- The length of a mini-train given its speed and time to cross a pole -/
theorem mini_train_length (speed_kmph : ℝ) (time_seconds : ℝ) : 
  speed_kmph = 75 → time_seconds = 3 → 
  (speed_kmph * 1000 / 3600) * time_seconds = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_mini_train_length_l2780_278092


namespace NUMINAMATH_CALUDE_power_function_through_point_l2780_278016

/-- A power function that passes through the point (2,8) -/
def f (x : ℝ) : ℝ := x^3

theorem power_function_through_point (x : ℝ) :
  f 2 = 8 ∧ f 3 = 27 := by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2780_278016


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2780_278009

noncomputable section

variables (a b c : ℝ) (f : ℝ → ℝ)

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_properties
  (h1 : quadratic_function f)
  (h2 : f 0 = 2)
  (h3 : ∀ x, f (x + 1) - f x = 2 * x - 1) :
  (∀ x, f x = x^2 - 2*x + 2) ∧
  (∀ x, x > 1 → (deriv f) x > 0) ∧
  (∀ x, x < 1 → (deriv f) x < 0) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 5) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_quadratic_function_properties_l2780_278009


namespace NUMINAMATH_CALUDE_exchange_theorem_l2780_278080

/-- Represents the number of exchanges between Xiao Zhang and Xiao Li -/
def num_exchanges : ℕ := 4

/-- Initial number of pencils Xiao Zhang has -/
def initial_pencils : ℕ := 200

/-- Initial number of fountain pens Xiao Li has -/
def initial_pens : ℕ := 20

/-- Number of pencils exchanged per transaction -/
def pencils_per_exchange : ℕ := 6

/-- Number of pens exchanged per transaction -/
def pens_per_exchange : ℕ := 1

/-- Ratio of pencils to pens after exchanges -/
def final_ratio : ℕ := 11

theorem exchange_theorem : 
  initial_pencils - num_exchanges * pencils_per_exchange = 
  final_ratio * (initial_pens - num_exchanges * pens_per_exchange) :=
by sorry


end NUMINAMATH_CALUDE_exchange_theorem_l2780_278080


namespace NUMINAMATH_CALUDE_trigonometric_ratio_equals_three_fourths_trigonometric_expression_equals_negative_four_l2780_278036

def α : Real := sorry
def n : ℤ := sorry

-- Part 1
theorem trigonometric_ratio_equals_three_fourths 
  (h1 : Real.cos α = -4/5) 
  (h2 : Real.sin α = 3/5) : 
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = 3/4 := by sorry

-- Part 2
theorem trigonometric_expression_equals_negative_four
  (h1 : Real.cos (π + α) = -1/2)
  (h2 : α > 3*π/2 ∧ α < 2*π) :
  (Real.sin (α + (2*n + 1)*π) + Real.sin (α - (2*n + 1)*π)) / 
  (Real.sin (α + 2*n*π) * Real.cos (α - 2*n*π)) = -4 := by sorry

end NUMINAMATH_CALUDE_trigonometric_ratio_equals_three_fourths_trigonometric_expression_equals_negative_four_l2780_278036


namespace NUMINAMATH_CALUDE_snake_diet_decade_l2780_278073

/-- The number of mice a snake eats in a decade -/
def mice_eaten_in_decade (weeks_per_mouse : ℕ) (weeks_per_year : ℕ) (years_per_decade : ℕ) : ℕ :=
  (weeks_per_year / weeks_per_mouse) * years_per_decade

/-- Theorem: A snake eating one mouse every 4 weeks will eat 130 mice in a decade -/
theorem snake_diet_decade : 
  mice_eaten_in_decade 4 52 10 = 130 := by
  sorry

#eval mice_eaten_in_decade 4 52 10

end NUMINAMATH_CALUDE_snake_diet_decade_l2780_278073


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_minus_3i_squared_l2780_278052

theorem imaginary_part_of_i_minus_3i_squared (i : ℂ) : 
  i * i = -1 →
  Complex.im (i * (1 - 3 * i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_minus_3i_squared_l2780_278052


namespace NUMINAMATH_CALUDE_students_left_proof_l2780_278058

/-- The number of students who showed up initially -/
def initial_students : ℕ := 16

/-- The number of students who were checked out early -/
def checked_out_students : ℕ := 7

/-- The number of students left at the end of the day -/
def remaining_students : ℕ := initial_students - checked_out_students

theorem students_left_proof : remaining_students = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_left_proof_l2780_278058


namespace NUMINAMATH_CALUDE_floor_times_x_equals_88_l2780_278096

theorem floor_times_x_equals_88 (x : ℝ) (h1 : x > 0) (h2 : ⌊x⌋ * x = 88) : x = 88 / 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_x_equals_88_l2780_278096


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l2780_278054

/-- The parabola equation y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line equation y = k(x-2) + 1 passing through P(2,1) -/
def line (k x y : ℝ) : Prop := y = k*(x-2) + 1

/-- The number of common points between the line and the parabola -/
inductive CommonPoints
  | one
  | two
  | none

/-- Theorem stating the conditions for the number of common points -/
theorem line_parabola_intersection (k : ℝ) :
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2) ↔ k = 0 ∧
  ¬(∃ p q : ℝ × ℝ, p ≠ q ∧ parabola p.1 p.2 ∧ parabola q.1 q.2 ∧ line k p.1 p.2 ∧ line k q.1 q.2) ∧
  ¬(∀ p : ℝ × ℝ, parabola p.1 p.2 → ¬line k p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l2780_278054


namespace NUMINAMATH_CALUDE_squats_on_fourth_day_l2780_278048

/-- Calculates the number of squats on a given day, given the initial number of squats and daily increase. -/
def squats_on_day (initial_squats : ℕ) (daily_increase : ℕ) (day : ℕ) : ℕ :=
  initial_squats + (day - 1) * daily_increase

/-- Theorem stating that given an initial number of 30 squats and a daily increase of 5,
    the number of squats on the fourth day will be 45. -/
theorem squats_on_fourth_day :
  squats_on_day 30 5 4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_squats_on_fourth_day_l2780_278048


namespace NUMINAMATH_CALUDE_square_root_x_minus_y_l2780_278031

theorem square_root_x_minus_y (x y : ℝ) (h : Real.sqrt (x - 2) + (y + 1)^2 = 0) : 
  (Real.sqrt (x - y))^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_x_minus_y_l2780_278031


namespace NUMINAMATH_CALUDE_roots_independent_of_k_l2780_278019

/-- The polynomial function with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^4 - (k+3)*x^3 - (k-11)*x^2 + (k+3)*x + (k-12)

/-- Theorem stating that 1 and -1 are roots of the polynomial for all real k -/
theorem roots_independent_of_k :
  ∀ k : ℝ, f k 1 = 0 ∧ f k (-1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_independent_of_k_l2780_278019


namespace NUMINAMATH_CALUDE_function_difference_l2780_278075

theorem function_difference (m : ℚ) : 
  let f (x : ℚ) := 4 * x^2 - 3 * x + 5
  let g (x : ℚ) := x^2 - m * x - 8
  (f 5 - g 5 = 20) → m = -53/5 := by
sorry

end NUMINAMATH_CALUDE_function_difference_l2780_278075


namespace NUMINAMATH_CALUDE_half_diamond_four_thirds_l2780_278038

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := sorry

-- Axioms
axiom diamond_def {a b : ℝ} (ha : 0 < a) (hb : 0 < b) : 
  diamond (a * b) b = a * (diamond b b)

axiom diamond_identity {a : ℝ} (ha : 0 < a) : 
  diamond (diamond a 1) a = diamond a 1

axiom diamond_one : diamond 1 1 = 1

-- Theorem to prove
theorem half_diamond_four_thirds : 
  diamond (1/2) (4/3) = 2/3 := by sorry

end NUMINAMATH_CALUDE_half_diamond_four_thirds_l2780_278038


namespace NUMINAMATH_CALUDE_product_and_sum_relations_l2780_278055

/-- Given positive integers p, q, r satisfying the specified conditions, prove that p - r = -430 --/
theorem product_and_sum_relations (p q r : ℕ+) 
  (h_product : p * q * r = Nat.factorial 10)
  (h_sum1 : p * q + p + q = 2450)
  (h_sum2 : q * r + q + r = 1012)
  (h_sum3 : r * p + r + p = 2020) :
  (p : ℤ) - (r : ℤ) = -430 := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_relations_l2780_278055


namespace NUMINAMATH_CALUDE_collinear_probability_l2780_278017

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots we are selecting -/
def selectedDots : ℕ := 5

/-- The number of possible collinear sets of 5 dots in the grid -/
def collinearSets : ℕ := 2 * gridSize + 2

/-- The probability of selecting 5 collinear dots from a 5x5 grid -/
theorem collinear_probability : 
  (collinearSets : ℚ) / Nat.choose totalDots selectedDots = 2 / 8855 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_l2780_278017


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2780_278063

/-- Given a circle with circumference 36 cm, its area is 324/π square centimeters. -/
theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 2 * π * r = 36 → π * r^2 = 324 / π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2780_278063


namespace NUMINAMATH_CALUDE_root_difference_range_l2780_278030

/-- Given a quadratic function f(x) = ax² + (b-a)x + (c-b) where a > b > c and a + b + c = 0,
    the absolute difference between its roots |x₁ - x₂| lies in the open interval (3/2, 2√3). -/
theorem root_difference_range (a b c : ℝ) (ha : a > b) (hb : b > c) (hsum : a + b + c = 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + (b - a) * x + (c - b)
  let x₁ := (-(b - a) + Real.sqrt ((b - a)^2 - 4 * a * (c - b))) / (2 * a)
  let x₂ := (-(b - a) - Real.sqrt ((b - a)^2 - 4 * a * (c - b))) / (2 * a)
  3/2 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_root_difference_range_l2780_278030


namespace NUMINAMATH_CALUDE_correct_product_after_reversal_error_l2780_278003

-- Define a function to reverse digits of a two-digit number
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

-- Define the theorem
theorem correct_product_after_reversal_error (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (reverseDigits a * b = 221) →  -- erroneous product is 221
  (a * b = 923) :=  -- correct product is 923
by sorry

end NUMINAMATH_CALUDE_correct_product_after_reversal_error_l2780_278003


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2780_278061

theorem quadratic_equation_properties (a b : ℝ) (ha : a > 0) (hab : a^2 = 4*b) :
  (a^2 - b^2 ≤ 4) ∧
  (a^2 + 1/b ≥ 4) ∧
  (∀ c : ℝ, ∀ x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x + b < c ↔ x₁ < x ∧ x < x₂) → |x₁ - x₂| = 4 → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2780_278061


namespace NUMINAMATH_CALUDE_topsoil_cost_theorem_l2780_278099

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def volume_in_cubic_yards : ℝ := 7

/-- The cost of topsoil for a given volume in cubic yards -/
def topsoil_cost (volume : ℝ) : ℝ :=
  volume * cubic_yards_to_cubic_feet * cost_per_cubic_foot

theorem topsoil_cost_theorem :
  topsoil_cost volume_in_cubic_yards = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_theorem_l2780_278099


namespace NUMINAMATH_CALUDE_f_shifted_is_even_f_has_three_zeros_l2780_278034

/-- A function that satisfies the given conditions -/
def f (x : ℝ) : ℝ := (x - 1)^2 - |x - 1|

/-- Theorem stating that f(x+1) is an even function on ℝ -/
theorem f_shifted_is_even : ∀ x : ℝ, f (x + 1) = f (-x + 1) := by sorry

/-- Theorem stating that f(x) has exactly three zeros on ℝ -/
theorem f_has_three_zeros : ∃! (a b c : ℝ), a < b ∧ b < c ∧ 
  (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) := by sorry

end NUMINAMATH_CALUDE_f_shifted_is_even_f_has_three_zeros_l2780_278034


namespace NUMINAMATH_CALUDE_f_value_at_2_l2780_278037

-- Define the function f
def f (x a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : 
  (f (-2) a b = 3) → (f 2 a b = -19) := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l2780_278037


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l2780_278062

def base_6_to_10 (n : List Nat) : Nat :=
  List.foldl (fun acc d => acc * 6 + d) 0 n.reverse

theorem pirate_loot_sum :
  let silver := base_6_to_10 [4, 5, 3, 2]
  let pearls := base_6_to_10 [1, 2, 5, 4]
  let spices := base_6_to_10 [6, 5, 4]
  silver + pearls + spices = 1636 := by
sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l2780_278062


namespace NUMINAMATH_CALUDE_vector_equality_sufficient_not_necessary_l2780_278020

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def parallel (a b : E) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_equality_sufficient_not_necessary 
  (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a = b → (‖a‖ = ‖b‖ ∧ parallel a b)) ∧ 
  ∃ (c d : E), ‖c‖ = ‖d‖ ∧ parallel c d ∧ c ≠ d := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_sufficient_not_necessary_l2780_278020


namespace NUMINAMATH_CALUDE_smallest_factor_of_32_not_8_l2780_278070

theorem smallest_factor_of_32_not_8 : ∃ n : ℕ, n = 16 ∧ 
  (32 % n = 0) ∧ (8 % n ≠ 0) ∧ 
  (∀ m : ℕ, m < n → (32 % m = 0 → 8 % m = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_of_32_not_8_l2780_278070


namespace NUMINAMATH_CALUDE_total_dogs_l2780_278094

/-- The number of dogs that can fetch -/
def fetch : ℕ := 55

/-- The number of dogs that can roll over -/
def roll : ℕ := 32

/-- The number of dogs that can play dead -/
def play : ℕ := 40

/-- The number of dogs that can fetch and roll over -/
def fetch_roll : ℕ := 20

/-- The number of dogs that can fetch and play dead -/
def fetch_play : ℕ := 18

/-- The number of dogs that can roll over and play dead -/
def roll_play : ℕ := 15

/-- The number of dogs that can do all three tricks -/
def all_tricks : ℕ := 12

/-- The number of dogs that can do no tricks -/
def no_tricks : ℕ := 14

/-- Theorem stating the total number of dogs in the center -/
theorem total_dogs : 
  fetch + roll + play - fetch_roll - fetch_play - roll_play + all_tricks + no_tricks = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_l2780_278094


namespace NUMINAMATH_CALUDE_root_sum_squared_l2780_278006

theorem root_sum_squared (a b : ℝ) : 
  (a^2 + 2*a - 2016 = 0) → 
  (b^2 + 2*b - 2016 = 0) → 
  (a + b = -2) → 
  (a^2 + 3*a + b = 2014) := by
sorry

end NUMINAMATH_CALUDE_root_sum_squared_l2780_278006


namespace NUMINAMATH_CALUDE_denise_age_l2780_278085

theorem denise_age (amanda beth carlos denise : ℕ) 
  (h1 : amanda = carlos - 4)
  (h2 : carlos = beth + 5)
  (h3 : denise = beth + 2)
  (h4 : amanda = 16) : 
  denise = 17 := by
  sorry

end NUMINAMATH_CALUDE_denise_age_l2780_278085


namespace NUMINAMATH_CALUDE_tan_ratio_given_sin_equality_l2780_278026

theorem tan_ratio_given_sin_equality (α : ℝ) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * (π / 180))) : 
  Real.tan (α + π / 180) / Real.tan (α - π / 180) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_given_sin_equality_l2780_278026


namespace NUMINAMATH_CALUDE_whole_number_between_36_and_40_l2780_278067

theorem whole_number_between_36_and_40 (M : ℕ) : 
  (9 < (M : ℚ) / 4) ∧ ((M : ℚ) / 4 < 10) → M = 37 ∨ M = 38 ∨ M = 39 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_36_and_40_l2780_278067


namespace NUMINAMATH_CALUDE_fraction_equality_l2780_278077

theorem fraction_equality (x y : ℚ) (a b : ℤ) (h1 : y = 40) (h2 : x + 35 = 4 * y) (h3 : 1/5 * x = a/b * y) (h4 : b ≠ 0) : a/b = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2780_278077


namespace NUMINAMATH_CALUDE_distance_for_specific_cube_l2780_278018

/-- Represents a cube suspended above a plane -/
structure SuspendedCube where
  side_length : ℝ
  adjacent_heights : Fin 3 → ℝ

/-- The distance from the closest vertex to the plane for a suspended cube -/
def distance_to_plane (cube : SuspendedCube) : ℝ :=
  sorry

/-- Theorem stating the distance for the given cube configuration -/
theorem distance_for_specific_cube :
  let cube : SuspendedCube :=
    { side_length := 8
      adjacent_heights := ![8, 10, 9] }
  distance_to_plane cube = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_for_specific_cube_l2780_278018


namespace NUMINAMATH_CALUDE_truck_rental_theorem_l2780_278012

-- Define the capacities of small and large trucks
def small_truck_capacity : ℕ := 300
def large_truck_capacity : ℕ := 400

-- Define the conditions from the problem
axiom condition1 : 2 * small_truck_capacity + 3 * large_truck_capacity = 1800
axiom condition2 : 3 * small_truck_capacity + 4 * large_truck_capacity = 2500

-- Define the total items to be transported
def total_items : ℕ := 3100

-- Define a rental plan as a pair of natural numbers (small trucks, large trucks)
def RentalPlan := ℕ × ℕ

-- Define a function to check if a rental plan is valid
def is_valid_plan (plan : RentalPlan) : Prop :=
  plan.1 * small_truck_capacity + plan.2 * large_truck_capacity = total_items

-- Define the set of all valid rental plans
def valid_plans : Set RentalPlan :=
  {plan | is_valid_plan plan}

-- Theorem stating the main result
theorem truck_rental_theorem :
  (small_truck_capacity = 300 ∧ large_truck_capacity = 400) ∧
  (valid_plans = {(9, 1), (5, 4), (1, 7)}) := by
  sorry


end NUMINAMATH_CALUDE_truck_rental_theorem_l2780_278012


namespace NUMINAMATH_CALUDE_largest_digit_sum_is_8_l2780_278040

/-- Represents a three-digit decimal as a fraction 1/y where y is an integer between 1 and 16 -/
def IsValidFraction (a b c : ℕ) (y : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  1 < y ∧ y ≤ 16 ∧
  (100 * a + 10 * b + c : ℚ) / 1000 = 1 / y

/-- The sum of digits a, b, and c is at most 8 given the conditions -/
theorem largest_digit_sum_is_8 :
  ∀ a b c y : ℕ, IsValidFraction a b c y → a + b + c ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_sum_is_8_l2780_278040


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2780_278002

/-- If three lines ax + y + 1 = 0, y = 3x, and x + y = 4 intersect at one point, then a = -4 -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃! p : ℝ × ℝ, a * p.1 + p.2 + 1 = 0 ∧ p.2 = 3 * p.1 ∧ p.1 + p.2 = 4) →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2780_278002


namespace NUMINAMATH_CALUDE_min_sum_squares_consecutive_integers_l2780_278057

theorem min_sum_squares_consecutive_integers (y : ℤ) : 
  (∃ x : ℤ, y^2 = (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + 
              (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2) →
  y^2 ≥ 121 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_consecutive_integers_l2780_278057


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2780_278076

/-- A geometric sequence with the given property -/
structure GeometricSequence where
  a : ℕ → ℝ
  has_identical_roots : ∃ x : ℝ, a 1 * x^2 - a 3 * x + a 2 = 0 ∧ 
    ∀ y : ℝ, a 1 * y^2 - a 3 * y + a 2 = 0 → y = x

/-- Sum of the first n terms of a geometric sequence -/
def sum (seq : GeometricSequence) (n : ℕ) : ℝ := sorry

/-- The main theorem -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : 
  sum seq 9 / sum seq 3 = 21 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2780_278076


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2780_278082

theorem hemisphere_surface_area (r : Real) : 
  π * r^2 = 3 → 3 * π * r^2 = 9 := by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2780_278082


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2780_278084

/-- Given a principal amount and an interest rate, if the simple interest for 2 years is 660
    and the compound interest for 2 years is 696.30, then the interest rate is 11%. -/
theorem interest_rate_calculation (P R : ℝ) : 
  P * R * 2 / 100 = 660 →
  P * ((1 + R / 100)^2 - 1) = 696.30 →
  R = 11 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2780_278084


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2780_278069

theorem inequality_equivalence (x : ℝ) :
  (3 * x - 5 ≥ 9 - 2 * x) ↔ (x ≥ 14 / 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2780_278069


namespace NUMINAMATH_CALUDE_restaurant_bill_rounding_l2780_278029

theorem restaurant_bill_rounding (people : ℕ) (individual_payment : ℚ) (total_payment : ℚ) :
  people = 9 →
  individual_payment = 3491/100 →
  total_payment = 31419/100 →
  ∃ (original_bill : ℚ), 
    original_bill = 31418/100 ∧
    original_bill * people ≤ total_payment ∧
    total_payment - original_bill * people < people * (1/100) :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_rounding_l2780_278029


namespace NUMINAMATH_CALUDE_roots_of_equation_l2780_278024

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 5*x + 6)*(x - 3)*(x + 2)
  {x : ℝ | f x = 0} = {2, 3, -2} := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2780_278024


namespace NUMINAMATH_CALUDE_board_problem_l2780_278074

def board_operation (a b c : ℤ) : ℤ × ℤ × ℤ :=
  (a, b, a + b - c)

def is_arithmetic_sequence (a b c : ℤ) : Prop :=
  b - a = c - b

def can_reach_sequence (start_a start_b start_c target_a target_b target_c : ℤ) : Prop :=
  ∃ (n : ℕ), ∃ (seq : ℕ → ℤ × ℤ × ℤ),
    seq 0 = (start_a, start_b, start_c) ∧
    (∀ i, i < n → 
      let (a, b, c) := seq i
      seq (i + 1) = board_operation a b c ∨ 
      seq (i + 1) = board_operation a c b ∨ 
      seq (i + 1) = board_operation b c a) ∧
    seq n = (target_a, target_b, target_c)

theorem board_problem :
  can_reach_sequence 3 9 15 2013 2019 2025 ∧
  is_arithmetic_sequence 2013 2019 2025 :=
sorry

end NUMINAMATH_CALUDE_board_problem_l2780_278074


namespace NUMINAMATH_CALUDE_xiaoming_age_l2780_278007

def is_valid_age (birth_year : ℕ) (current_year : ℕ) : Prop :=
  current_year - birth_year = (birth_year / 1000) + ((birth_year / 100) % 10) + ((birth_year / 10) % 10) + (birth_year % 10)

theorem xiaoming_age :
  ∃ (age : ℕ), (age = 22 ∨ age = 4) ∧
  ∃ (birth_year : ℕ),
    birth_year ≥ 1900 ∧
    birth_year < 2015 ∧
    is_valid_age birth_year 2015 ∧
    age = 2015 - birth_year :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_age_l2780_278007


namespace NUMINAMATH_CALUDE_unique_solution_l2780_278027

/-- The inequality condition for positive real numbers a, b, c, d, and real number x -/
def inequality_condition (a b c d x : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  ((a^3 / (a^3 + 15*b*c*d))^(1/2) : ℝ) ≥ (a^x / (a^x + b^x + c^x + d^x) : ℝ)

/-- The theorem stating that 15/8 is the only solution -/
theorem unique_solution :
  ∀ x : ℝ, (∀ a b c d : ℝ, inequality_condition a b c d x) ↔ x = 15/8 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2780_278027


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_theorem_l2780_278014

/-- The distance between the center of a sphere and the plane of a right triangle tangent to the sphere. -/
def sphere_triangle_distance (sphere_radius : ℝ) (triangle_side1 triangle_side2 triangle_side3 : ℝ) : ℝ :=
  sorry

/-- Theorem stating the distance between the center of a sphere and the plane of a right triangle tangent to the sphere. -/
theorem sphere_triangle_distance_theorem :
  sphere_triangle_distance 10 8 15 17 = Real.sqrt 91 := by
  sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_theorem_l2780_278014


namespace NUMINAMATH_CALUDE_expected_value_is_five_l2780_278071

/-- Represents the outcome of rolling a fair 8-sided die -/
inductive DieRoll
  | one | two | three | four | five | six | seven | eight

/-- The probability of each outcome of the die roll -/
def prob : DieRoll → ℚ
  | _ => 1/8

/-- The winnings for each outcome of the die roll -/
def winnings : DieRoll → ℚ
  | DieRoll.two => 4
  | DieRoll.four => 8
  | DieRoll.six => 12
  | DieRoll.eight => 16
  | _ => 0

/-- The expected value of the winnings -/
def expected_value : ℚ :=
  (prob DieRoll.two * winnings DieRoll.two) +
  (prob DieRoll.four * winnings DieRoll.four) +
  (prob DieRoll.six * winnings DieRoll.six) +
  (prob DieRoll.eight * winnings DieRoll.eight)

theorem expected_value_is_five :
  expected_value = 5 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_five_l2780_278071


namespace NUMINAMATH_CALUDE_ratio_problem_l2780_278025

theorem ratio_problem (second_part : ℝ) (ratio_percent : ℝ) : 
  second_part = 4 → ratio_percent = 125 → (ratio_percent / 100) * second_part = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2780_278025


namespace NUMINAMATH_CALUDE_x_axis_intersection_correct_y_coord_correct_l2780_278053

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 5 * y - 2 * x = 10

/-- The point where the line intersects the x-axis -/
def x_axis_intersection : ℝ × ℝ := (-5, 0)

/-- The y-coordinate when x = -5 -/
def y_coord_at_neg_five : ℝ := 0

/-- Theorem stating that x_axis_intersection is on the line and has y-coordinate 0 -/
theorem x_axis_intersection_correct :
  line_equation x_axis_intersection.1 x_axis_intersection.2 ∧ x_axis_intersection.2 = 0 := by sorry

/-- Theorem stating that when x = -5, the y-coordinate is y_coord_at_neg_five -/
theorem y_coord_correct : line_equation (-5) y_coord_at_neg_five := by sorry

end NUMINAMATH_CALUDE_x_axis_intersection_correct_y_coord_correct_l2780_278053


namespace NUMINAMATH_CALUDE_same_range_implies_b_constraint_l2780_278093

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 2

-- Define the function g
def g (b : ℝ) (x : ℝ) : ℝ := f b (f b x)

-- State the theorem
theorem same_range_implies_b_constraint (b : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f b x = y) ↔ (∀ y : ℝ, ∃ x : ℝ, g b x = y) →
  b ≥ 4 ∨ b ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_same_range_implies_b_constraint_l2780_278093


namespace NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l2780_278097

theorem sin_x_squared_not_periodic : ¬∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), Real.sin ((x + p)^2) = Real.sin (x^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l2780_278097


namespace NUMINAMATH_CALUDE_ones_digit_of_first_prime_in_sequence_l2780_278023

/-- A function that returns the ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Definition of an arithmetic sequence of five primes -/
def is_arithmetic_prime_sequence (p q r s t : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ Nat.Prime t ∧
  q = p + 10 ∧ r = q + 10 ∧ s = r + 10 ∧ t = s + 10

theorem ones_digit_of_first_prime_in_sequence (p q r s t : ℕ) :
  is_arithmetic_prime_sequence p q r s t → p > 5 → ones_digit p = 1 :=
by sorry

end NUMINAMATH_CALUDE_ones_digit_of_first_prime_in_sequence_l2780_278023


namespace NUMINAMATH_CALUDE_f_of_5_equals_15_l2780_278046

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem f_of_5_equals_15 : f 5 = 15 := by sorry

end NUMINAMATH_CALUDE_f_of_5_equals_15_l2780_278046


namespace NUMINAMATH_CALUDE_triangle_side_length_l2780_278090

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    if a = 4√5, b = 5, and cos A = 3/5, then c = 11. -/
theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 4 * Real.sqrt 5 →
  b = 5 →
  Real.cos A = 3 / 5 →
  c = 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2780_278090


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_l2780_278087

noncomputable def inscribed_sphere_radius : ℝ := Real.sqrt 6 - 1

theorem circumscribed_sphere_radius
  (inscribed_radius : ℝ)
  (h_inscribed_radius : inscribed_radius = inscribed_sphere_radius)
  (h_touching : inscribed_radius > 0) :
  ∃ (circumscribed_radius : ℝ),
    circumscribed_radius = 5 * (Real.sqrt 2 + 1) * inscribed_radius :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_l2780_278087


namespace NUMINAMATH_CALUDE_apartment_number_l2780_278066

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def swap_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem apartment_number : 
  ∃! n : ℕ, is_three_digit n ∧ is_perfect_cube n ∧ Nat.Prime (swap_digits n) ∧ n = 125 := by
  sorry

end NUMINAMATH_CALUDE_apartment_number_l2780_278066


namespace NUMINAMATH_CALUDE_count_multiples_6_or_8_not_both_l2780_278015

theorem count_multiples_6_or_8_not_both : Nat := by
  -- Define the set of positive integers less than 201
  let S := Finset.range 201

  -- Define the subsets of multiples of 6 and 8
  let mult6 := S.filter (λ n => n % 6 = 0)
  let mult8 := S.filter (λ n => n % 8 = 0)

  -- Define the set of numbers that are multiples of either 6 or 8, but not both
  let result := (mult6 ∪ mult8) \ (mult6 ∩ mult8)

  -- The theorem statement
  have : result.card = 42 := by sorry

  -- Return the result
  exact result.card

end NUMINAMATH_CALUDE_count_multiples_6_or_8_not_both_l2780_278015


namespace NUMINAMATH_CALUDE_madeline_utilities_l2780_278083

/-- Calculates the amount left for utilities given expenses and income --/
def amount_for_utilities (rent groceries medical emergency hourly_wage hours : ℕ) : ℕ :=
  hourly_wage * hours - (rent + groceries + medical + emergency)

/-- Proves that Madeline's amount left for utilities is $70 --/
theorem madeline_utilities : amount_for_utilities 1200 400 200 200 15 138 = 70 := by
  sorry

end NUMINAMATH_CALUDE_madeline_utilities_l2780_278083


namespace NUMINAMATH_CALUDE_pretzels_theorem_l2780_278060

def pretzels_problem (initial_pretzels john_pretzels marcus_pretzels : ℕ) : Prop :=
  let alan_pretzels := initial_pretzels - john_pretzels - marcus_pretzels
  john_pretzels - alan_pretzels = 1

theorem pretzels_theorem :
  ∀ (initial_pretzels john_pretzels marcus_pretzels : ℕ),
    initial_pretzels = 95 →
    john_pretzels = 28 →
    marcus_pretzels = 40 →
    marcus_pretzels = john_pretzels + 12 →
    pretzels_problem initial_pretzels john_pretzels marcus_pretzels :=
by
  sorry

#check pretzels_theorem

end NUMINAMATH_CALUDE_pretzels_theorem_l2780_278060


namespace NUMINAMATH_CALUDE_cupcakes_frosted_proof_l2780_278013

/-- Cagney's frosting rate in cupcakes per second -/
def cagney_rate : ℚ := 1 / 25

/-- Lacey's frosting rate in cupcakes per second -/
def lacey_rate : ℚ := 1 / 35

/-- Total working time in seconds -/
def total_time : ℕ := 600

/-- The number of cupcakes frosted when working together -/
def cupcakes_frosted : ℕ := 41

theorem cupcakes_frosted_proof :
  ⌊(cagney_rate + lacey_rate) * total_time⌋ = cupcakes_frosted := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_frosted_proof_l2780_278013


namespace NUMINAMATH_CALUDE_find_number_l2780_278033

theorem find_number : ∃ x : ℤ, x + 5 = 9 ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_find_number_l2780_278033


namespace NUMINAMATH_CALUDE_multiply_72517_and_9999_l2780_278089

theorem multiply_72517_and_9999 : 72517 * 9999 = 725097483 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72517_and_9999_l2780_278089


namespace NUMINAMATH_CALUDE_total_stuffed_animals_l2780_278011

def stuffed_animals (mckenna kenley tenly : ℕ) : Prop :=
  (kenley = 2 * mckenna) ∧ 
  (tenly = kenley + 5) ∧ 
  (mckenna + kenley + tenly = 175)

theorem total_stuffed_animals :
  ∃ (mckenna kenley tenly : ℕ), 
    mckenna = 34 ∧ 
    stuffed_animals mckenna kenley tenly :=
by
  sorry

end NUMINAMATH_CALUDE_total_stuffed_animals_l2780_278011


namespace NUMINAMATH_CALUDE_sine_sum_of_roots_l2780_278004

theorem sine_sum_of_roots (a b c : ℝ) (α β : ℝ) : 
  a^2 + b^2 ≠ 0 → 
  0 ≤ α → α ≤ π →
  0 ≤ β → β ≤ π →
  α ≠ β →
  (a * Real.cos α + b * Real.sin α + c = 0) →
  (a * Real.cos β + b * Real.sin β + c = 0) →
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_sine_sum_of_roots_l2780_278004


namespace NUMINAMATH_CALUDE_only_valid_number_l2780_278078

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ 
  n = (n / 10)^2 + (n % 10)^2 + 13

theorem only_valid_number : ∀ n : ℕ, is_valid_number n ↔ n = 54 := by sorry

end NUMINAMATH_CALUDE_only_valid_number_l2780_278078


namespace NUMINAMATH_CALUDE_collectors_edition_dolls_combined_l2780_278050

theorem collectors_edition_dolls_combined (dina_dolls : ℕ) (ivy_dolls : ℕ) (luna_dolls : ℕ) :
  dina_dolls = 60 →
  dina_dolls = 2 * ivy_dolls →
  ivy_dolls = luna_dolls + 10 →
  (2 : ℕ) * (ivy_dolls * 2) = 3 * ivy_dolls →
  2 * luna_dolls = luna_dolls →
  (2 : ℕ) * (ivy_dolls * 2) / 3 + luna_dolls / 2 = 30 := by
sorry

end NUMINAMATH_CALUDE_collectors_edition_dolls_combined_l2780_278050


namespace NUMINAMATH_CALUDE_identity_proof_l2780_278072

theorem identity_proof (A B C A₁ B₁ C₁ : ℝ) :
  (A^2 + B^2 + C^2) * (A₁^2 + B₁^2 + C₁^2) - (A*A₁ + B*B₁ + C*C₁)^2 =
  (A*B₁ + A₁*B)^2 + (A*C₁ + A₁*C)^2 + (B*C₁ + B₁*C)^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2780_278072


namespace NUMINAMATH_CALUDE_value_of_b_l2780_278041

theorem value_of_b (a b t : ℝ) 
  (eq1 : a - t / 6 * b = 20)
  (eq2 : a - t / 5 * b = -10)
  (t_val : t = 60) : b = 15 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2780_278041


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_iff_m_eq_one_l2780_278039

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem complex_purely_imaginary_iff_m_eq_one (m : ℝ) :
  is_purely_imaginary (m^2 - m + m * Complex.I) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_iff_m_eq_one_l2780_278039


namespace NUMINAMATH_CALUDE_parabola_directrix_l2780_278065

/-- The equation of the directrix of the parabola y = 4x^2 is y = -1/16 -/
theorem parabola_directrix (x y : ℝ) : 
  y = 4 * x^2 → (∃ (k : ℝ), k = -1/16 ∧ y = k) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2780_278065


namespace NUMINAMATH_CALUDE_square_field_area_l2780_278035

/-- The area of a square field with side length 20 meters is 400 square meters. -/
theorem square_field_area (side_length : ℝ) (h : side_length = 20) : 
  side_length * side_length = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l2780_278035


namespace NUMINAMATH_CALUDE_cans_per_cat_package_is_9_l2780_278005

/-- The number of cans in each package of cat food -/
def cans_per_cat_package : ℕ := sorry

/-- The number of packages of cat food bought -/
def cat_packages : ℕ := 6

/-- The number of packages of dog food bought -/
def dog_packages : ℕ := 2

/-- The number of cans in each package of dog food -/
def cans_per_dog_package : ℕ := 3

/-- The difference between total cat food cans and total dog food cans -/
def can_difference : ℕ := 48

theorem cans_per_cat_package_is_9 : 
  cans_per_cat_package = 9 := by sorry

end NUMINAMATH_CALUDE_cans_per_cat_package_is_9_l2780_278005


namespace NUMINAMATH_CALUDE_solution_set1_correct_solution_set2_correct_l2780_278081

-- Define the solution set for the first inequality
def solutionSet1 : Set ℝ := {x | x ≥ 1 ∨ x < 0}

-- Define the solution set for the second inequality based on the value of a
def solutionSet2 (a : ℝ) : Set ℝ :=
  if a = 0 then
    {x | x > -1}
  else if a > 0 then
    {x | -1 < x ∧ x < 1/a}
  else if a < -1 then
    {x | x < -1 ∨ x > 1/a}
  else if a = -1 then
    {x | x ≠ -1}
  else
    {x | x < 1/a ∨ x > -1}

-- Theorem for the first inequality
theorem solution_set1_correct :
  ∀ x : ℝ, x ∈ solutionSet1 ↔ (x - 1) / x ≥ 0 ∧ x ≠ 0 :=
sorry

-- Theorem for the second inequality
theorem solution_set2_correct :
  ∀ a x : ℝ, x ∈ solutionSet2 a ↔ a * x^2 + (a - 1) * x - 1 < 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set1_correct_solution_set2_correct_l2780_278081


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l2780_278022

theorem eighth_term_of_sequence (x : ℝ) : 
  let nth_term (n : ℕ) := (-1)^(n+1) * ((n^2 + 1) : ℝ) * x^n
  nth_term 8 = -65 * x^8 := by sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l2780_278022


namespace NUMINAMATH_CALUDE_product_evaluation_l2780_278098

theorem product_evaluation : (6 - 5) * (6 - 4) * (6 - 3) * (6 - 2) * (6 - 1) * 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2780_278098


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l2780_278056

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) →
  (3 * q^2 + 9 * q - 21 = 0) →
  (3*p - 4) * (6*q - 8) = 122 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l2780_278056


namespace NUMINAMATH_CALUDE_max_a_value_l2780_278064

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- Define the property that f(x) ≤ 6 for all x in (0,2]
def property (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x ≤ 2 → f a x ≤ 6

-- State the theorem
theorem max_a_value :
  (∃ a : ℝ, property a) →
  (∃ a_max : ℝ, property a_max ∧ ∀ a : ℝ, property a → a ≤ a_max) →
  (∀ a_max : ℝ, (property a_max ∧ ∀ a : ℝ, property a → a ≤ a_max) → a_max = -1) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l2780_278064


namespace NUMINAMATH_CALUDE_loser_received_35_percent_l2780_278079

/-- Given a total number of votes and the difference between winner and loser,
    calculate the percentage of votes received by the losing candidate. -/
def loser_vote_percentage (total_votes : ℕ) (vote_difference : ℕ) : ℚ :=
  (total_votes - vote_difference) / (2 * total_votes) * 100

/-- Theorem stating that given 4500 total votes and a 1350 vote difference,
    the losing candidate received 35% of the votes. -/
theorem loser_received_35_percent :
  loser_vote_percentage 4500 1350 = 35 := by
  sorry

end NUMINAMATH_CALUDE_loser_received_35_percent_l2780_278079


namespace NUMINAMATH_CALUDE_inequality_solution_set_k_value_range_l2780_278028

-- Problem 1
theorem inequality_solution_set (x : ℝ) : 
  -2 * x^2 - x + 6 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3/2 :=
sorry

-- Problem 2
theorem k_value_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) ↔ (k > Real.sqrt 2 ∨ k < -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_k_value_range_l2780_278028


namespace NUMINAMATH_CALUDE_balance_after_first_year_l2780_278051

/-- Represents the account balance after one year -/
def balance_after_one_year (initial_deposit : ℝ) (interest : ℝ) : ℝ :=
  initial_deposit + interest

/-- The theorem states that given an initial deposit of $1000 and an interest of $100,
    the balance after one year is $1100 -/
theorem balance_after_first_year :
  balance_after_one_year 1000 100 = 1100 := by
  sorry

end NUMINAMATH_CALUDE_balance_after_first_year_l2780_278051


namespace NUMINAMATH_CALUDE_mothers_day_discount_l2780_278045

def original_price : ℝ := 125
def mother_discount : ℝ := 0.1
def additional_discount : ℝ := 0.04
def children_count : ℕ := 4

theorem mothers_day_discount (price : ℝ) (md : ℝ) (ad : ℝ) (cc : ℕ) :
  price > 0 →
  md > 0 →
  ad > 0 →
  cc ≥ 3 →
  price * (1 - md) * (1 - ad) = 108 := by
sorry

end NUMINAMATH_CALUDE_mothers_day_discount_l2780_278045


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l2780_278008

theorem triangle_side_lengths 
  (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = c + 2) 
  (h3 : Real.sin (Real.arcsin (Real.sqrt 3 / 2)) = Real.sqrt 3 / 2) : 
  a = 7 ∧ b = 5 ∧ c = 3 := by
  sorry

#check triangle_side_lengths

end NUMINAMATH_CALUDE_triangle_side_lengths_l2780_278008


namespace NUMINAMATH_CALUDE_min_value_of_f_l2780_278047

/-- The function f(x) = 2x³ - 6x² + m, where m is a constant -/
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

/-- Theorem: Given f(x) = 2x³ - 6x² + m, where m is a constant,
    and f(x) reaches a maximum value of 2 within the interval [-2, 2],
    the minimum value of f(x) within [-2, 2] is -6. -/
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 2) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -6 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, -6 ≤ f y m :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_f_l2780_278047


namespace NUMINAMATH_CALUDE_degrees_minutes_to_decimal_l2780_278059

-- Define the conversion factor from minutes to degrees
def minutes_to_degrees (m : ℚ) : ℚ := m / 60

-- Define the problem
theorem degrees_minutes_to_decimal (d : ℚ) (m : ℚ) :
  d + minutes_to_degrees m = 18.4 → d = 18 ∧ m = 24 :=
by sorry

end NUMINAMATH_CALUDE_degrees_minutes_to_decimal_l2780_278059


namespace NUMINAMATH_CALUDE_divisibility_implies_inequality_l2780_278044

theorem divisibility_implies_inequality (a k : ℕ+) :
  (a^2 + k : ℕ) ∣ ((a - 1) * a * (a + 1) : ℕ) → k ≥ a :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_inequality_l2780_278044


namespace NUMINAMATH_CALUDE_mr_blue_bean_yield_l2780_278091

/-- Calculates the expected bean yield for a rectangular terrain --/
def expected_bean_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (length_steps : ℝ) * step_length * (width_steps : ℝ) * step_length * yield_per_sqft

/-- Proves that the expected bean yield for Mr. Blue's terrain is 5906.25 pounds --/
theorem mr_blue_bean_yield :
  expected_bean_yield 25 35 3 0.75 = 5906.25 := by
  sorry

#eval expected_bean_yield 25 35 3 0.75

end NUMINAMATH_CALUDE_mr_blue_bean_yield_l2780_278091


namespace NUMINAMATH_CALUDE_hyperbola_I_equation_equilateral_hyperbola_equation_l2780_278043

-- Part I
def hyperbola_I (x y : ℝ) : Prop :=
  y^2 / 36 - x^2 / 28 = 1

theorem hyperbola_I_equation
  (foci_on_y_axis : True)
  (focal_distance : ℝ)
  (h_focal_distance : focal_distance = 16)
  (eccentricity : ℝ)
  (h_eccentricity : eccentricity = 4/3) :
  ∃ (x y : ℝ), hyperbola_I x y :=
sorry

-- Part II
def equilateral_hyperbola (x y : ℝ) : Prop :=
  x^2 / 18 - y^2 / 18 = 1

theorem equilateral_hyperbola_equation
  (is_equilateral : True)
  (focus : ℝ × ℝ)
  (h_focus : focus = (-6, 0)) :
  ∃ (x y : ℝ), equilateral_hyperbola x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_I_equation_equilateral_hyperbola_equation_l2780_278043


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l2780_278086

theorem absolute_value_equals_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l2780_278086


namespace NUMINAMATH_CALUDE_books_remaining_after_sale_l2780_278095

-- Define the initial number of books
def initial_books : Nat := 136

-- Define the number of books sold
def books_sold : Nat := 109

-- Theorem to prove
theorem books_remaining_after_sale : 
  initial_books - books_sold = 27 := by sorry

end NUMINAMATH_CALUDE_books_remaining_after_sale_l2780_278095


namespace NUMINAMATH_CALUDE_tyler_meal_combinations_l2780_278068

/-- The number of meat options available -/
def num_meats : ℕ := 4

/-- The number of vegetable options available -/
def num_vegetables : ℕ := 4

/-- The number of dessert options available -/
def num_desserts : ℕ := 5

/-- The number of bread options available -/
def num_breads : ℕ := 3

/-- The number of vegetables Tyler must choose -/
def vegetables_to_choose : ℕ := 2

/-- The number of breads Tyler must choose -/
def breads_to_choose : ℕ := 2

/-- Calculates the number of ways to choose k items from n items without replacement and without order -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of ways to choose k items from n items without replacement but with order -/
def permute (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The total number of meal combinations Tyler can choose -/
def total_combinations : ℕ := 
  num_meats * choose num_vegetables vegetables_to_choose * num_desserts * permute num_breads breads_to_choose

theorem tyler_meal_combinations : total_combinations = 720 := by
  sorry

end NUMINAMATH_CALUDE_tyler_meal_combinations_l2780_278068


namespace NUMINAMATH_CALUDE_probability_no_repetition_l2780_278049

def three_digit_numbers : ℕ := 3^3

def numbers_without_repetition : ℕ := 6

theorem probability_no_repetition :
  (numbers_without_repetition : ℚ) / three_digit_numbers = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_repetition_l2780_278049


namespace NUMINAMATH_CALUDE_salary_change_percentage_l2780_278042

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * 1.5
  let final_salary := increased_salary * 0.9
  (final_salary - initial_salary) / initial_salary * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l2780_278042


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2780_278021

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + x + 1

/-- The derivative of the parabola function -/
def f' (x : ℝ) : ℝ := 2*x + 1

/-- The point through which the tangent line passes -/
def P : ℝ × ℝ := (-1, 0)

/-- Theorem: The tangent line to y = x^2 + x + 1 passing through (-1, 0) is x - y + 1 = 0 -/
theorem tangent_line_equation :
  ∃ (x₀ : ℝ), 
    let y₀ := f x₀
    let m := f' x₀
    (P.1 - x₀) * m = P.2 - y₀ ∧
    ∀ (x y : ℝ), y = m * (x - x₀) + y₀ ↔ x - y + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2780_278021
