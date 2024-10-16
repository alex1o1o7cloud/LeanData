import Mathlib

namespace NUMINAMATH_CALUDE_min_value_quadratic_ratio_l916_91620

theorem min_value_quadratic_ratio (a b : ℝ) (h1 : b > 0) 
  (h2 : b^2 - 4*a = 0) : 
  (∀ x : ℝ, (a*x^2 + b*x + 1) / b ≥ 2) ∧ 
  (∃ x : ℝ, (a*x^2 + b*x + 1) / b = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_ratio_l916_91620


namespace NUMINAMATH_CALUDE_no_absolute_winner_probability_l916_91634

/-- Represents a player in the mini-tournament -/
inductive Player : Type
| Alyosha : Player
| Borya : Player
| Vasya : Player

/-- Represents the result of a match between two players -/
def MatchResult := Player → Player → ℝ

/-- The probability that there is no absolute winner in the mini-tournament -/
def noAbsoluteWinnerProbability (matchResult : MatchResult) : ℝ :=
  let p_AB := matchResult Player.Alyosha Player.Borya
  let p_BV := matchResult Player.Borya Player.Vasya
  0.24 * (1 - p_AB) * (1 - p_BV) + 0.36 * p_AB * (1 - p_BV)

/-- The main theorem stating that the probability of no absolute winner is 0.36 -/
theorem no_absolute_winner_probability (matchResult : MatchResult) 
  (h1 : matchResult Player.Alyosha Player.Borya = 0.6)
  (h2 : matchResult Player.Borya Player.Vasya = 0.4) :
  noAbsoluteWinnerProbability matchResult = 0.36 := by
  sorry


end NUMINAMATH_CALUDE_no_absolute_winner_probability_l916_91634


namespace NUMINAMATH_CALUDE_bakers_sales_l916_91699

/-- Baker's cake and pastry sales problem -/
theorem bakers_sales (cakes_made pastries_made cakes_sold pastries_sold : ℕ) 
  (h1 : cakes_made = 475)
  (h2 : pastries_made = 539)
  (h3 : cakes_sold = 358)
  (h4 : pastries_sold = 297) :
  cakes_sold - pastries_sold = 61 := by
  sorry

end NUMINAMATH_CALUDE_bakers_sales_l916_91699


namespace NUMINAMATH_CALUDE_cos_40_plus_sqrt3_tan_10_eq_1_l916_91630

theorem cos_40_plus_sqrt3_tan_10_eq_1 : 
  Real.cos (40 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_40_plus_sqrt3_tan_10_eq_1_l916_91630


namespace NUMINAMATH_CALUDE_sum_of_squares_l916_91690

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 131) → (a + b + c = 18) → (a^2 + b^2 + c^2 = 62) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l916_91690


namespace NUMINAMATH_CALUDE_distance_between_vertices_l916_91668

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y + 2| = 4

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop :=
  graph_equation x y ∧ y ≥ -2

def parabola2 (x y : ℝ) : Prop :=
  graph_equation x y ∧ y < -2

-- Define the vertices
def vertex1 : ℝ × ℝ := (0, 1)
def vertex2 : ℝ × ℝ := (0, -3)

-- Theorem statement
theorem distance_between_vertices :
  ∃ (v1 v2 : ℝ × ℝ),
    (∀ x y, parabola1 x y → (x, y) = v1 ∨ y > v1.2) ∧
    (∀ x y, parabola2 x y → (x, y) = v2 ∨ y < v2.2) ∧
    ‖v1 - v2‖ = 4 :=
  sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l916_91668


namespace NUMINAMATH_CALUDE_custom_operation_theorem_l916_91641

theorem custom_operation_theorem (a b : ℚ) : 
  a ≠ 0 → b ≠ 0 → a - b = 9 → a / b = 20 → 1 / a + 1 / b = 19 / 60 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_theorem_l916_91641


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l916_91671

theorem last_three_digits_of_7_to_103 : 7^103 ≡ 614 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l916_91671


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l916_91640

theorem interest_difference_theorem (P : ℝ) : 
  let r : ℝ := 0.04  -- 4% annual interest rate
  let t : ℕ := 2     -- 2 years time period
  let compound_interest := P * (1 + r)^t - P
  let simple_interest := P * r * t
  compound_interest - simple_interest = 1 → P = 625 := by
sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l916_91640


namespace NUMINAMATH_CALUDE_sector_max_area_l916_91691

/-- Given a sector with perimeter 20 cm, its area is maximized when the central angle is 2 radians, 
    and the maximum area is 25 cm². -/
theorem sector_max_area (r : ℝ) (α : ℝ) (l : ℝ) (S : ℝ) :
  0 < r → r < 10 →
  l + 2 * r = 20 →
  l = r * α →
  S = 1/2 * r * l →
  (∀ r' α' l' S', 
    0 < r' → r' < 10 →
    l' + 2 * r' = 20 →
    l' = r' * α' →
    S' = 1/2 * r' * l' →
    S' ≤ S) →
  α = 2 ∧ S = 25 := by
sorry


end NUMINAMATH_CALUDE_sector_max_area_l916_91691


namespace NUMINAMATH_CALUDE_power_function_through_point_l916_91676

theorem power_function_through_point (a : ℝ) :
  (2 : ℝ) ^ a = (1 / 2 : ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l916_91676


namespace NUMINAMATH_CALUDE_test_questions_count_l916_91646

theorem test_questions_count : ∀ (total_questions : ℕ),
  (total_questions % 4 = 0) →
  (20 : ℚ) / total_questions > (60 : ℚ) / 100 →
  (20 : ℚ) / total_questions < (70 : ℚ) / 100 →
  total_questions = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l916_91646


namespace NUMINAMATH_CALUDE_sum_of_X_and_Y_l916_91679

/-- X is defined as 2 groups of 10 plus 6 units -/
def X : ℕ := 2 * 10 + 6

/-- Y is defined as 4 groups of 10 plus 1 unit -/
def Y : ℕ := 4 * 10 + 1

/-- The sum of X and Y is 67 -/
theorem sum_of_X_and_Y : X + Y = 67 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_X_and_Y_l916_91679


namespace NUMINAMATH_CALUDE_intersection_k_value_l916_91617

-- Define the lines
def line1 (x y k : ℝ) : Prop := 2*x + 3*y - k = 0
def line2 (x y k : ℝ) : Prop := x - k*y + 12 = 0

-- Define the condition that the intersection point lies on the y-axis
def intersection_on_y_axis (k : ℝ) : Prop :=
  ∃ y : ℝ, line1 0 y k ∧ line2 0 y k

-- Theorem statement
theorem intersection_k_value :
  ∀ k : ℝ, intersection_on_y_axis k → (k = 6 ∨ k = -6) :=
by sorry

end NUMINAMATH_CALUDE_intersection_k_value_l916_91617


namespace NUMINAMATH_CALUDE_zeros_order_l916_91678

open Real

noncomputable def f (x : ℝ) := x + log x
noncomputable def g (x : ℝ) := x * log x - 1
noncomputable def h (x : ℝ) := 1 - 1/x + x/2 + x^2/3

theorem zeros_order (a b c : ℝ) 
  (ha : a > 0 ∧ f a = 0)
  (hb : b > 0 ∧ g b = 0)
  (hc : c > 0 ∧ h c = 0)
  (hf : ∀ x, x > 0 → x ≠ a → f x ≠ 0)
  (hg : ∀ x, x > 0 → x ≠ b → g x ≠ 0)
  (hh : ∀ x, x > 0 → x ≠ c → h x ≠ 0) :
  b > c ∧ c > a :=
sorry

end NUMINAMATH_CALUDE_zeros_order_l916_91678


namespace NUMINAMATH_CALUDE_practice_time_ratio_l916_91651

/-- Represents the practice time in minutes for each day of the week -/
structure PracticeTime where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem stating the ratio of practice time on Monday to Tuesday is 2:1 -/
theorem practice_time_ratio (p : PracticeTime) : 
  p.thursday = 50 ∧ 
  p.wednesday = p.thursday + 5 ∧ 
  p.tuesday = p.wednesday - 10 ∧ 
  p.friday = 60 ∧ 
  p.monday + p.tuesday + p.wednesday + p.thursday + p.friday = 300 →
  p.monday = 2 * p.tuesday :=
by sorry

end NUMINAMATH_CALUDE_practice_time_ratio_l916_91651


namespace NUMINAMATH_CALUDE_hyperbola_iff_k_in_range_l916_91643

/-- A curve is defined by the equation (x^2)/(k+4) + (y^2)/(k-1) = 1 -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k + 4) + y^2 / (k - 1) = 1 ∧ (k + 4) * (k - 1) < 0

/-- The range of k values for which the curve represents a hyperbola -/
def hyperbola_k_range : Set ℝ := {k | -4 < k ∧ k < 1}

/-- Theorem stating that the curve represents a hyperbola if and only if k is in the range (-4, 1) -/
theorem hyperbola_iff_k_in_range (k : ℝ) :
  is_hyperbola k ↔ k ∈ hyperbola_k_range :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_iff_k_in_range_l916_91643


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l916_91627

theorem sqrt_expression_simplification :
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l916_91627


namespace NUMINAMATH_CALUDE_smallest_square_multiplier_ten_l916_91642

def is_smallest_square_multiplier (y : ℕ) (n : ℕ) : Prop :=
  y > 0 ∧ ∃ (m : ℕ), y * n = m^2 ∧
  ∀ (k : ℕ), k > 0 → k < y → ¬∃ (m : ℕ), k * n = m^2

theorem smallest_square_multiplier_ten (n : ℕ) :
  is_smallest_square_multiplier 10 n → n = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_multiplier_ten_l916_91642


namespace NUMINAMATH_CALUDE_maddies_mom_milk_consumption_l916_91663

/-- Represents the weekly coffee consumption scenario of Maddie's mom -/
structure CoffeeConsumption where
  cups_per_day : ℕ
  ounces_per_cup : ℚ
  ounces_per_bag : ℚ
  price_per_bag : ℚ
  price_per_gallon_milk : ℚ
  weekly_coffee_budget : ℚ

/-- Calculates the amount of milk in gallons used per week -/
def milk_gallons_per_week (c : CoffeeConsumption) : ℚ :=
  sorry

/-- Theorem stating that given the specific conditions, 
    the amount of milk used per week is 0.5 gallons -/
theorem maddies_mom_milk_consumption :
  let c : CoffeeConsumption := {
    cups_per_day := 2,
    ounces_per_cup := 3/2,
    ounces_per_bag := 21/2,
    price_per_bag := 8,
    price_per_gallon_milk := 4,
    weekly_coffee_budget := 18
  }
  milk_gallons_per_week c = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_maddies_mom_milk_consumption_l916_91663


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l916_91633

theorem quadratic_real_roots_condition (k : ℝ) : 
  (k ≠ 0) → 
  (∃ x : ℝ, k * x^2 - x + 1 = 0) ↔ 
  (k ≤ 1/4 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l916_91633


namespace NUMINAMATH_CALUDE_cube_max_volume_l916_91622

variable (a : ℝ) -- Sum of all edges
variable (x y z : ℝ) -- Dimensions of the parallelepiped

-- Define the volume function
def volume (x y z : ℝ) : ℝ := x * y * z

-- Define the constraint that the sum of edges is fixed
def sum_constraint (x y z : ℝ) : Prop := x + y + z = a

-- State the theorem
theorem cube_max_volume :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → sum_constraint a x y z →
  volume x y z ≤ volume (a/3) (a/3) (a/3) :=
sorry

end NUMINAMATH_CALUDE_cube_max_volume_l916_91622


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l916_91632

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_product : a 3 * a 7 = 8) : 
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l916_91632


namespace NUMINAMATH_CALUDE_min_value_of_f_l916_91688

/-- A cubic function with a constant term. -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x + m

/-- The theorem stating the minimum value of f on [0, 2] given its maximum value. -/
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f m y ≤ f m x) ∧
  (∀ x ∈ Set.Icc 0 2, f m x ≤ 3) →
  ∃ x ∈ Set.Icc 0 2, f m x = -1 ∧ ∀ y ∈ Set.Icc 0 2, -1 ≤ f m y :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l916_91688


namespace NUMINAMATH_CALUDE_function_positivity_implies_m_range_l916_91631

theorem function_positivity_implies_m_range 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (m : ℝ) 
  (h_f : ∀ x, f x = 2 * m * x^2 - 2 * (4 - m) * x + 1) 
  (h_g : ∀ x, g x = m * x) 
  (h_pos : ∀ x, f x > 0 ∨ g x > 0) : 
  0 < m ∧ m < 8 := by
sorry

end NUMINAMATH_CALUDE_function_positivity_implies_m_range_l916_91631


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l916_91656

/-- Given points A, B, C, and D in a Cartesian plane, where D is the midpoint of AB,
    prove that the sum of the slope and y-intercept of the line passing through C and D is 3.6 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 8) →
  B = (0, 0) →
  C = (10, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope + y_intercept = 3.6 := by
sorry


end NUMINAMATH_CALUDE_slope_intercept_sum_l916_91656


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l916_91661

theorem sum_first_150_remainder (n : Nat) (h : n = 150) : 
  (List.range n).sum % 5600 = 125 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l916_91661


namespace NUMINAMATH_CALUDE_waiting_time_is_correct_l916_91629

/-- The total waiting time in minutes for Mark's vaccine appointments -/
def total_waiting_time : ℕ :=
  let days_first_vaccine := 4
  let days_second_vaccine := 20
  let days_first_secondary := 30 + 10  -- 1 month and 10 days
  let days_second_secondary := 14 + 3  -- 2 weeks and 3 days
  let days_full_effectiveness := 3 * 7 -- 3 weeks
  let total_days := days_first_vaccine + days_second_vaccine + days_first_secondary +
                    days_second_secondary + days_full_effectiveness
  let minutes_per_day := 24 * 60
  total_days * minutes_per_day

/-- Theorem stating that the total waiting time is 146,880 minutes -/
theorem waiting_time_is_correct : total_waiting_time = 146880 := by
  sorry

end NUMINAMATH_CALUDE_waiting_time_is_correct_l916_91629


namespace NUMINAMATH_CALUDE_functional_equation_solution_l916_91613

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + (x + 1/2) * f (1 - x) = 1

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
    (f 0 = 2 ∧ f 1 = -2) ∧
    (∀ x ≠ 1/2, f x = 2 / (1 - 2*x)) ∧
    f (1/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l916_91613


namespace NUMINAMATH_CALUDE_jack_additional_money_l916_91612

/-- The amount of additional money Jack needs to buy socks and shoes -/
theorem jack_additional_money (sock_cost shoes_cost jack_money : ℚ)
  (h1 : sock_cost = 19)
  (h2 : shoes_cost = 92)
  (h3 : jack_money = 40) :
  sock_cost + shoes_cost - jack_money = 71 := by
  sorry

end NUMINAMATH_CALUDE_jack_additional_money_l916_91612


namespace NUMINAMATH_CALUDE_two_apples_per_slice_l916_91618

/-- The number of apples in each slice of pie -/
def apples_per_slice (total_apples : ℕ) (num_pies : ℕ) (slices_per_pie : ℕ) : ℚ :=
  total_apples / (num_pies * slices_per_pie)

/-- Theorem: Given the conditions, prove that there are 2 apples in each slice of pie -/
theorem two_apples_per_slice :
  let total_apples : ℕ := 4 * 12  -- 4 dozen apples
  let num_pies : ℕ := 4
  let slices_per_pie : ℕ := 6
  apples_per_slice total_apples num_pies slices_per_pie = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_apples_per_slice_l916_91618


namespace NUMINAMATH_CALUDE_solve_equation_l916_91665

theorem solve_equation (x : ℝ) : 2*x - 3*x + 4*x = 120 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l916_91665


namespace NUMINAMATH_CALUDE_max_xyz_value_l916_91669

theorem max_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y + z = (x + z) * (y + z)) :
  x * y * z ≤ 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_max_xyz_value_l916_91669


namespace NUMINAMATH_CALUDE_banana_bread_flour_calculation_l916_91666

/-- Given the recipe for banana bread, calculate the number of cups of flour needed. -/
theorem banana_bread_flour_calculation 
  (flour_per_mush : ℚ)  -- Cups of flour per cup of mush
  (bananas_per_mush : ℚ)  -- Number of bananas per cup of mush
  (total_bananas : ℚ)  -- Total number of bananas used
  (h1 : flour_per_mush = 3)  -- 3 cups of flour per cup of mush
  (h2 : bananas_per_mush = 4)  -- 4 bananas make one cup of mush
  (h3 : total_bananas = 20)  -- Hannah uses 20 bananas
  : (total_bananas / bananas_per_mush) * flour_per_mush = 15 := by
  sorry

#check banana_bread_flour_calculation

end NUMINAMATH_CALUDE_banana_bread_flour_calculation_l916_91666


namespace NUMINAMATH_CALUDE_initial_average_equals_correct_average_l916_91693

/-- The number of values in the set -/
def n : ℕ := 10

/-- The correct average of the numbers -/
def correct_average : ℚ := 401/10

/-- The difference between the first incorrectly copied number and its actual value -/
def first_error : ℤ := 17

/-- The difference between the second incorrectly copied number and its actual value -/
def second_error : ℤ := 13 - 31

/-- The sum of all errors in the incorrectly copied numbers -/
def total_error : ℤ := first_error + second_error

theorem initial_average_equals_correct_average :
  let S := n * correct_average
  let initial_average := (S + total_error) / n
  initial_average = correct_average := by sorry

end NUMINAMATH_CALUDE_initial_average_equals_correct_average_l916_91693


namespace NUMINAMATH_CALUDE_johnsons_class_size_l916_91650

theorem johnsons_class_size (finley_class : ℕ) (johnson_class : ℕ) 
  (h1 : finley_class = 24) 
  (h2 : johnson_class = finley_class / 2 + 10) : 
  johnson_class = 22 := by
  sorry

end NUMINAMATH_CALUDE_johnsons_class_size_l916_91650


namespace NUMINAMATH_CALUDE_email_sending_combinations_l916_91619

theorem email_sending_combinations (num_addresses : ℕ) (num_emails : ℕ) : 
  num_addresses = 3 → num_emails = 5 → num_addresses ^ num_emails = 243 :=
by sorry

end NUMINAMATH_CALUDE_email_sending_combinations_l916_91619


namespace NUMINAMATH_CALUDE_expression_evaluation_l916_91695

theorem expression_evaluation :
  (4^4 - 4*(4-2)^4)^(4+1) = 14889702426 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l916_91695


namespace NUMINAMATH_CALUDE_trains_crossing_time_l916_91654

/-- The time it takes for two trains moving in opposite directions to cross each other -/
theorem trains_crossing_time (length_A length_B speed_A speed_B : ℝ) : 
  length_A = 108 →
  length_B = 112 →
  speed_A = 50 * (1000 / 3600) →
  speed_B = 82 * (1000 / 3600) →
  let total_length := length_A + length_B
  let relative_speed := speed_A + speed_B
  let crossing_time := total_length / relative_speed
  ∃ ε > 0, |crossing_time - 6| < ε :=
by
  sorry

#check trains_crossing_time

end NUMINAMATH_CALUDE_trains_crossing_time_l916_91654


namespace NUMINAMATH_CALUDE_fraction_value_preservation_l916_91684

theorem fraction_value_preservation (original_numerator original_denominator increase_numerator : ℕ) 
  (h1 : original_numerator = 3)
  (h2 : original_denominator = 16)
  (h3 : increase_numerator = 6) :
  ∃ (increase_denominator : ℕ),
    (original_numerator + increase_numerator) / (original_denominator + increase_denominator) = 
    original_numerator / original_denominator ∧ 
    increase_denominator = 32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_preservation_l916_91684


namespace NUMINAMATH_CALUDE_missing_number_proof_l916_91674

theorem missing_number_proof : ∃ x : ℤ, |7 - 8 * (3 - x)| - |5 - 11| = 73 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l916_91674


namespace NUMINAMATH_CALUDE_set_equality_proof_l916_91694

theorem set_equality_proof (M N : Set ℕ) : M = {3, 2} → N = {2, 3} → M = N := by
  sorry

end NUMINAMATH_CALUDE_set_equality_proof_l916_91694


namespace NUMINAMATH_CALUDE_unique_pronunciations_in_C_l916_91664

/-- Represents a Chinese character with its pronunciation --/
structure ChineseChar :=
  (char : String)
  (pronunciation : String)

/-- Represents a group of words with underlined characters --/
structure WordGroup :=
  (name : String)
  (underlinedChars : List ChineseChar)

/-- Check if all pronunciations in a list are unique --/
def allUniquePronunciations (chars : List ChineseChar) : Prop :=
  ∀ i j, i ≠ j → (chars.get i).pronunciation ≠ (chars.get j).pronunciation

/-- The four word groups from the problem --/
def groupA : WordGroup := sorry
def groupB : WordGroup := sorry
def groupC : WordGroup := sorry
def groupD : WordGroup := sorry

/-- The main theorem to prove --/
theorem unique_pronunciations_in_C :
  allUniquePronunciations groupC.underlinedChars ∧
  ¬allUniquePronunciations groupA.underlinedChars ∧
  ¬allUniquePronunciations groupB.underlinedChars ∧
  ¬allUniquePronunciations groupD.underlinedChars :=
sorry

end NUMINAMATH_CALUDE_unique_pronunciations_in_C_l916_91664


namespace NUMINAMATH_CALUDE_circle_line_intersection_properties_l916_91689

/-- Given a circle and a line in 2D space, prove properties about their intersection and a related circle. -/
theorem circle_line_intersection_properties 
  (x y : ℝ) (m : ℝ) 
  (h_circle : x^2 + y^2 - 2*x - 4*y + m = 0) 
  (h_line : x + 2*y = 4) 
  (h_perpendicular : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 - 2*x₁ - 4*y₁ + m = 0 ∧ 
    x₁ + 2*y₁ = 4 ∧
    x₂^2 + y₂^2 - 2*x₂ - 4*y₂ + m = 0 ∧ 
    x₂ + 2*y₂ = 4 ∧
    x₁*x₂ + y₁*y₂ = 0) :
  m = 8/5 ∧ 
  ∀ (x y : ℝ), x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔ 
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
      x = (1-t)*x₁ + t*x₂ ∧ 
      y = (1-t)*y₁ + t*y₂ :=
by sorry


end NUMINAMATH_CALUDE_circle_line_intersection_properties_l916_91689


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l916_91660

theorem quadratic_always_positive (k : ℝ) : ∀ x : ℝ, x^2 - (k - 4)*x + k - 7 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l916_91660


namespace NUMINAMATH_CALUDE_sparrow_count_l916_91611

theorem sparrow_count (bluebird_count : ℕ) (ratio_bluebird : ℕ) (ratio_sparrow : ℕ) 
  (h1 : bluebird_count = 28)
  (h2 : ratio_bluebird = 4)
  (h3 : ratio_sparrow = 5) :
  (bluebird_count * ratio_sparrow) / ratio_bluebird = 35 :=
by sorry

end NUMINAMATH_CALUDE_sparrow_count_l916_91611


namespace NUMINAMATH_CALUDE_interest_rate_multiple_l916_91600

theorem interest_rate_multiple (P r m : ℝ) 
  (h1 : P * r^2 = 40)
  (h2 : P * (m * r)^2 = 360)
  : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_multiple_l916_91600


namespace NUMINAMATH_CALUDE_area_EFGH_extended_l916_91697

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ
  area : ℝ

/-- Calculates the area of the extended quadrilateral -/
def area_extended_quadrilateral (q : ExtendedQuadrilateral) : ℝ :=
  q.area + 2 * q.area

/-- Theorem stating the area of the extended quadrilateral E'F'G'H' -/
theorem area_EFGH_extended (q : ExtendedQuadrilateral)
  (h_ef : q.ef = 5)
  (h_fg : q.fg = 6)
  (h_gh : q.gh = 7)
  (h_he : q.he = 8)
  (h_area : q.area = 20) :
  area_extended_quadrilateral q = 60 := by
  sorry

end NUMINAMATH_CALUDE_area_EFGH_extended_l916_91697


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l916_91659

-- Define the function g(x)
def g (x : ℝ) : ℝ := 18 * x^4 - 20 * x^2 + 3

-- State the theorem
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = 1 ∧ g r = 0 ∧ ∀ x : ℝ, g x = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l916_91659


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l916_91687

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 1 / 45) : x^2 - y^2 = 8 / 675 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l916_91687


namespace NUMINAMATH_CALUDE_triangle_shape_l916_91667

/-- Given a triangle ABC, prove that it is a right isosceles triangle under certain conditions. -/
theorem triangle_shape (a b c : ℝ) (A B C : ℝ) :
  (Real.log a - Real.log c = Real.log (Real.sin B)) →
  (Real.log (Real.sin B) = -Real.log (Real.sqrt 2)) →
  (0 < B) →
  (B < π / 2) →
  (A + B + C = π) →
  (a * Real.sin C = b * Real.sin A) →
  (b * Real.sin C = c * Real.sin B) →
  (c * Real.sin A = a * Real.sin B) →
  (A = π / 4 ∧ B = π / 4 ∧ C = π / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_shape_l916_91667


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l916_91601

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 5*x₁ + 4 = 0 → x₂^2 - 5*x₂ + 4 = 0 → x₁ ≠ x₂ → 1/x₁ + 1/x₂ = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l916_91601


namespace NUMINAMATH_CALUDE_quadratic_root_values_l916_91698

theorem quadratic_root_values (b c : ℝ) : 
  (Complex.I * Real.sqrt 2 + 1) ^ 2 + b * (Complex.I * Real.sqrt 2 + 1) + c = 0 → 
  b = -2 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_values_l916_91698


namespace NUMINAMATH_CALUDE_intersection_length_l916_91605

/-- The curve C in the Cartesian plane -/
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- The line l passing through (0, 1) -/
def l (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

/-- Point A on the intersection of C and l -/
def A (k x₁ y₁ : ℝ) : Prop := C x₁ y₁ ∧ l k x₁ y₁

/-- Point B on the intersection of C and l -/
def B (k x₂ y₂ : ℝ) : Prop := C x₂ y₂ ∧ l k x₂ y₂

/-- The condition that OA · AB = 0 -/
def orthogonal (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

/-- The main theorem -/
theorem intersection_length 
  (k x₁ y₁ x₂ y₂ : ℝ) 
  (hA : A k x₁ y₁) 
  (hB : B k x₂ y₂) 
  (hO : orthogonal x₁ y₁ x₂ y₂) : 
  ((x₂ - x₁)^2 + (y₂ - y₁)^2) = (4*Real.sqrt 65/17)^2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_length_l916_91605


namespace NUMINAMATH_CALUDE_min_grass_seed_amount_is_75_l916_91603

/-- Represents a bag of grass seed with its weight and price -/
structure GrassSeedBag where
  weight : ℕ
  price : ℚ

/-- Finds the minimum amount of grass seed that can be purchased given the constraints -/
def minGrassSeedAmount (bags : List GrassSeedBag) (maxWeight : ℕ) (exactCost : ℚ) : ℕ :=
  sorry

theorem min_grass_seed_amount_is_75 :
  let bags : List GrassSeedBag := [
    { weight := 5, price := 13.82 },
    { weight := 10, price := 20.43 },
    { weight := 25, price := 32.25 }
  ]
  let maxWeight : ℕ := 80
  let exactCost : ℚ := 98.75

  minGrassSeedAmount bags maxWeight exactCost = 75 := by sorry

end NUMINAMATH_CALUDE_min_grass_seed_amount_is_75_l916_91603


namespace NUMINAMATH_CALUDE_share_face_value_l916_91680

/-- Given shares with a certain dividend rate and market value, 
    calculate the face value that yields a desired interest rate. -/
theorem share_face_value 
  (dividend_rate : ℚ) 
  (desired_interest_rate : ℚ) 
  (market_value : ℚ) 
  (h1 : dividend_rate = 9 / 100)
  (h2 : desired_interest_rate = 12 / 100)
  (h3 : market_value = 45) : 
  ∃ (face_value : ℚ), 
    face_value * dividend_rate = market_value * desired_interest_rate ∧ 
    face_value = 60 := by
  sorry

#check share_face_value

end NUMINAMATH_CALUDE_share_face_value_l916_91680


namespace NUMINAMATH_CALUDE_largest_reciprocal_l916_91628

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem largest_reciprocal :
  let a := (1 : ℚ) / 2
  let b := (3 : ℚ) / 7
  let c := (1 : ℚ) / 2  -- 0.5 as a rational number
  let d := 7
  let e := 2001
  (reciprocal b > reciprocal a) ∧
  (reciprocal b > reciprocal c) ∧
  (reciprocal b > reciprocal d) ∧
  (reciprocal b > reciprocal e) :=
by sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l916_91628


namespace NUMINAMATH_CALUDE_baseball_games_played_l916_91649

theorem baseball_games_played (wins losses played : ℕ) : 
  wins = 5 → 
  played = wins + losses → 
  played = 2 * losses → 
  played = 10 := by
sorry

end NUMINAMATH_CALUDE_baseball_games_played_l916_91649


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l916_91602

theorem necessary_not_sufficient_condition (x : ℝ) :
  (∀ y : ℝ, y > 2 → y > 1) ∧ (∃ z : ℝ, z > 1 ∧ z ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l916_91602


namespace NUMINAMATH_CALUDE_storage_unit_blocks_l916_91648

/-- Represents the dimensions of a rectangular storage unit -/
structure StorageUnit where
  length : ℝ
  width : ℝ
  height : ℝ
  wallThickness : ℝ

/-- Calculates the number of blocks needed for a storage unit -/
def blocksNeeded (unit : StorageUnit) : ℝ :=
  let totalVolume := unit.length * unit.width * unit.height
  let interiorLength := unit.length - 2 * unit.wallThickness
  let interiorWidth := unit.width - 2 * unit.wallThickness
  let interiorHeight := unit.height - unit.wallThickness
  let interiorVolume := interiorLength * interiorWidth * interiorHeight
  totalVolume - interiorVolume

/-- Theorem stating that the storage unit with given dimensions requires 738 blocks -/
theorem storage_unit_blocks :
  let unit : StorageUnit := {
    length := 15,
    width := 12,
    height := 8,
    wallThickness := 1.5
  }
  blocksNeeded unit = 738 := by sorry

end NUMINAMATH_CALUDE_storage_unit_blocks_l916_91648


namespace NUMINAMATH_CALUDE_line_angle_of_inclination_l916_91672

/-- The angle of inclination of the line 2x + 2y - 5 = 0 is 135° -/
theorem line_angle_of_inclination :
  let line := {(x, y) : ℝ × ℝ | 2*x + 2*y - 5 = 0}
  ∃ α : Real, α = 135 * (π / 180) ∧ 
    ∀ (x y : ℝ), (x, y) ∈ line → (Real.tan α = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_angle_of_inclination_l916_91672


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_five_l916_91683

/-- Given vectors in ℝ², prove that if (a - c) is parallel to b, then k = 5 --/
theorem parallel_vectors_imply_k_equals_five (a b c : ℝ × ℝ) (k : ℝ) :
  a = (3, 1) →
  b = (1, 3) →
  c = (k, 7) →
  ∃ (t : ℝ), (a.1 - c.1, a.2 - c.2) = (t * b.1, t * b.2) →
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_five_l916_91683


namespace NUMINAMATH_CALUDE_candle_arrangement_l916_91610

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of candles that satisfies the given conditions -/
def num_candles : ℕ := 4

theorem candle_arrangement :
  (∀ c : ℕ, (choose_2 c * 9 = 54) → c = num_candles) :=
by sorry

end NUMINAMATH_CALUDE_candle_arrangement_l916_91610


namespace NUMINAMATH_CALUDE_cube_root_abs_square_sum_l916_91652

theorem cube_root_abs_square_sum : ∃ (x : ℝ), x^3 = -8 ∧ x + |(-6)| - 2^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_abs_square_sum_l916_91652


namespace NUMINAMATH_CALUDE_product_of_roots_l916_91677

theorem product_of_roots (x : ℝ) : 
  (∃ a b c : ℝ, a * b * c = -9 ∧ 
   ∀ x, 4 * x^3 - 2 * x^2 - 25 * x + 36 = 0 ↔ (x = a ∨ x = b ∨ x = c)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l916_91677


namespace NUMINAMATH_CALUDE_exp_gt_one_plus_x_l916_91645

theorem exp_gt_one_plus_x (x : ℝ) (h : x ≠ 0) : Real.exp x > 1 + x := by
  sorry

end NUMINAMATH_CALUDE_exp_gt_one_plus_x_l916_91645


namespace NUMINAMATH_CALUDE_max_value_at_e_l916_91608

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_at_e :
  ∀ x : ℝ, x > 0 → f x ≤ f (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_at_e_l916_91608


namespace NUMINAMATH_CALUDE_equilateral_triangle_cd_product_l916_91673

/-- An equilateral triangle with vertices at (0,0), (c, 20), and (d, 51) -/
structure EquilateralTriangle where
  c : ℝ
  d : ℝ
  is_equilateral : (c^2 + 20^2 = d^2 + 51^2) ∧ 
                   (c^2 + 20^2 = c^2 + d^2 + 51^2 - 2*c*d - 2*20*51) ∧
                   (d^2 + 51^2 = c^2 + d^2 + 51^2 - 2*c*d - 2*20*51)

/-- The product of c and d in the equilateral triangle equals -5822/3 -/
theorem equilateral_triangle_cd_product (t : EquilateralTriangle) : t.c * t.d = -5822/3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_cd_product_l916_91673


namespace NUMINAMATH_CALUDE_jane_vases_per_day_l916_91623

/-- The number of vases Jane can arrange per day given the total number of vases and the number of days -/
def vases_per_day (total_vases : ℕ) (days : ℕ) : ℚ :=
  (total_vases : ℚ) / (days : ℚ)

/-- Theorem stating that Jane can arrange 15.5 vases per day given the problem conditions -/
theorem jane_vases_per_day :
  vases_per_day 248 16 = 31/2 := by sorry

end NUMINAMATH_CALUDE_jane_vases_per_day_l916_91623


namespace NUMINAMATH_CALUDE_factorization_equality_l916_91644

theorem factorization_equality (a x y : ℝ) : a*x^2 + 2*a*x*y + a*y^2 = a*(x+y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l916_91644


namespace NUMINAMATH_CALUDE_trees_on_promenade_l916_91638

/-- The number of trees planted along a circular promenade -/
def number_of_trees (promenade_length : ℕ) (tree_interval : ℕ) : ℕ :=
  promenade_length / tree_interval

/-- Theorem: The number of trees planted along a circular promenade of length 1200 meters, 
    with trees planted at intervals of 30 meters, is equal to 40. -/
theorem trees_on_promenade : number_of_trees 1200 30 = 40 := by
  sorry

end NUMINAMATH_CALUDE_trees_on_promenade_l916_91638


namespace NUMINAMATH_CALUDE_max_value_expression_l916_91653

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (⨆ x, 2 * (a - x) * (x + c * Real.sqrt (x^2 + b^2))) = a^2 + c^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l916_91653


namespace NUMINAMATH_CALUDE_quadratic_roots_l916_91609

theorem quadratic_roots (d : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + d = 0 ↔ x = (3 + Real.sqrt d) / 2 ∨ x = (3 - Real.sqrt d) / 2) →
  d = 9/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l916_91609


namespace NUMINAMATH_CALUDE_john_earnings_is_80_l916_91692

/-- Calculates the amount of money John makes repairing cars --/
def john_earnings (total_cars : ℕ) (standard_repair_time : ℕ) (longer_repair_percentage : ℚ) (hourly_rate : ℚ) : ℚ :=
  let standard_cars := 3
  let longer_cars := total_cars - standard_cars
  let standard_time := standard_cars * standard_repair_time
  let longer_time := longer_cars * (standard_repair_time * (1 + longer_repair_percentage))
  let total_time := standard_time + longer_time
  let total_hours := total_time / 60
  total_hours * hourly_rate

/-- Theorem stating that John makes $80 repairing cars --/
theorem john_earnings_is_80 :
  john_earnings 5 40 (1/2) 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_john_earnings_is_80_l916_91692


namespace NUMINAMATH_CALUDE_contract_completion_problem_l916_91655

/-- Represents the contract completion problem -/
theorem contract_completion_problem (total_days : ℕ) (initial_hours_per_day : ℕ) 
  (days_worked : ℕ) (work_completed_fraction : ℚ) (additional_men : ℕ) 
  (new_hours_per_day : ℕ) :
  total_days = 46 →
  initial_hours_per_day = 8 →
  days_worked = 33 →
  work_completed_fraction = 4/7 →
  additional_men = 81 →
  new_hours_per_day = 9 →
  ∃ (initial_men : ℕ), 
    (initial_men * days_worked * initial_hours_per_day : ℚ) / (total_days * initial_hours_per_day) = work_completed_fraction ∧
    ((initial_men + additional_men) * (total_days - days_worked) * new_hours_per_day : ℚ) / (total_days * initial_hours_per_day) = 1 - work_completed_fraction ∧
    initial_men = 117 :=
by sorry

end NUMINAMATH_CALUDE_contract_completion_problem_l916_91655


namespace NUMINAMATH_CALUDE_complete_square_d_value_l916_91685

/-- Given a quadratic equation x^2 - 6x + 5 = 0, when transformed
    into the form (x + c)^2 = d, the value of d is 4. -/
theorem complete_square_d_value :
  ∃ c d : ℝ, (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + c)^2 = d) ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_d_value_l916_91685


namespace NUMINAMATH_CALUDE_expression_simplification_l916_91686

theorem expression_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((((x + 2)^2 * (x^2 - 2*x + 2)^2) / (x^3 + 8)^2)^2 * 
   (((x - 2)^2 * (x^2 + 2*x + 2)^2) / (x^3 - 8)^2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l916_91686


namespace NUMINAMATH_CALUDE_yoghurt_cost_l916_91604

/-- Given Tara's purchase of ice cream and yoghurt, prove the cost of each yoghurt carton. -/
theorem yoghurt_cost (ice_cream_cartons : ℕ) (yoghurt_cartons : ℕ) 
  (ice_cream_cost : ℕ) (price_difference : ℕ) :
  ice_cream_cartons = 19 →
  yoghurt_cartons = 4 →
  ice_cream_cost = 7 →
  price_difference = 129 →
  (ice_cream_cartons * ice_cream_cost - price_difference) / yoghurt_cartons = 1 := by
  sorry

end NUMINAMATH_CALUDE_yoghurt_cost_l916_91604


namespace NUMINAMATH_CALUDE_range_of_k_l916_91621

-- Define the propositions p and q
def p (k : ℝ) : Prop := ∃ (x y : ℝ), x^2/k + y^2/(4-k) = 1 ∧ k > 0 ∧ 4 - k > 0

def q (k : ℝ) : Prop := ∃ (x y : ℝ), x^2/(k-1) + y^2/(k-3) = 1 ∧ (k-1)*(k-3) < 0

-- State the theorem
theorem range_of_k (k : ℝ) : (p k ∨ q k) → 1 < k ∧ k < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l916_91621


namespace NUMINAMATH_CALUDE_similar_triangles_segment_length_l916_91657

/-- Triangle similarity is a relation between two triangles -/
def TriangleSimilar (t1 t2 : Type) : Prop := sorry

/-- Length of a segment -/
def SegmentLength (s : Type) : ℝ := sorry

theorem similar_triangles_segment_length 
  (PQR XYZ GHI : Type) 
  (h1 : TriangleSimilar PQR XYZ) 
  (h2 : TriangleSimilar XYZ GHI) 
  (h3 : SegmentLength PQ = 5) 
  (h4 : SegmentLength QR = 15) 
  (h5 : SegmentLength HI = 30) : 
  SegmentLength XY = 2.5 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_segment_length_l916_91657


namespace NUMINAMATH_CALUDE_spinner_sections_l916_91616

theorem spinner_sections (n : ℕ) (n_pos : n > 0) : 
  (1 - 1 / n : ℚ) ^ 2 = 559 / 1000 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_sections_l916_91616


namespace NUMINAMATH_CALUDE_remaining_money_correct_l916_91615

structure Currency where
  usd : ℚ
  eur : ℚ
  gbp : ℚ

def initial_amount : Currency := ⟨5.10, 8.75, 10.30⟩

def spend_usd (amount : ℚ) (c : Currency) : Currency :=
  ⟨c.usd - amount, c.eur, c.gbp⟩

def spend_eur (amount : ℚ) (c : Currency) : Currency :=
  ⟨c.usd, c.eur - amount, c.gbp⟩

def exchange_gbp_to_eur (gbp_amount : ℚ) (eur_amount : ℚ) (c : Currency) : Currency :=
  ⟨c.usd, c.eur + eur_amount, c.gbp - gbp_amount⟩

def final_amount : Currency :=
  initial_amount
  |> spend_usd 1.05
  |> spend_usd 2.00
  |> spend_eur 3.25
  |> exchange_gbp_to_eur 5.00 5.60

theorem remaining_money_correct :
  final_amount.usd = 2.05 ∧
  final_amount.eur = 11.10 ∧
  final_amount.gbp = 5.30 := by
  sorry


end NUMINAMATH_CALUDE_remaining_money_correct_l916_91615


namespace NUMINAMATH_CALUDE_unique_g_3_l916_91637

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the conditions
axiom g_1 : g 1 = -1
axiom g_property : ∀ x y : ℝ, g (x^2 - y^2) = (x - y) * (g x - g y)

-- Define m as the number of possible values of g(3)
def m : ℕ := sorry

-- Define t as the sum of all possible values of g(3)
def t : ℝ := sorry

-- Theorem statement
theorem unique_g_3 : m = 1 ∧ t = -3 := by sorry

end NUMINAMATH_CALUDE_unique_g_3_l916_91637


namespace NUMINAMATH_CALUDE_maximum_marks_correct_l916_91658

/-- The maximum marks in an exam where:
    1. The passing threshold is 33% of the maximum marks.
    2. A student got 92 marks.
    3. The student failed by 40 marks (i.e., needed 40 more marks to pass). -/
def maximum_marks : ℕ := 400

/-- The passing threshold as a fraction of the maximum marks -/
def passing_threshold : ℚ := 33 / 100

/-- The marks obtained by the student -/
def obtained_marks : ℕ := 92

/-- The additional marks needed to pass -/
def additional_marks_needed : ℕ := 40

theorem maximum_marks_correct :
  maximum_marks * (passing_threshold : ℚ) = obtained_marks + additional_marks_needed := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_correct_l916_91658


namespace NUMINAMATH_CALUDE_employee_price_calculation_l916_91624

/-- Calculates the employee's price for a video recorder given the wholesale cost, markup percentage, and employee discount percentage. -/
theorem employee_price_calculation 
  (wholesale_cost : ℝ) 
  (markup_percentage : ℝ) 
  (employee_discount_percentage : ℝ) : 
  wholesale_cost = 200 ∧ 
  markup_percentage = 20 ∧ 
  employee_discount_percentage = 30 → 
  wholesale_cost * (1 + markup_percentage / 100) * (1 - employee_discount_percentage / 100) = 168 := by
  sorry

#check employee_price_calculation

end NUMINAMATH_CALUDE_employee_price_calculation_l916_91624


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l916_91675

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_proof (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 5 = 10)
  (h3 : a 1 + a 2 + a 3 = 3) :
  a 1 = -2 ∧ ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l916_91675


namespace NUMINAMATH_CALUDE_square_root_16_l916_91626

theorem square_root_16 (x : ℝ) : (x + 1)^2 = 16 → x = 3 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_16_l916_91626


namespace NUMINAMATH_CALUDE_triangle_to_decagon_area_ratio_l916_91647

/-- A regular decagon -/
structure RegularDecagon where
  -- Add necessary fields here
  area : ℝ

/-- A triangle within a regular decagon formed by connecting three non-adjacent vertices -/
structure TriangleInDecagon (d : RegularDecagon) where
  -- Add necessary fields here
  area : ℝ

/-- The ratio of the area of a triangle to the area of the regular decagon it's inscribed in is 1/5 -/
theorem triangle_to_decagon_area_ratio 
  (d : RegularDecagon) 
  (t : TriangleInDecagon d) : 
  t.area / d.area = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_triangle_to_decagon_area_ratio_l916_91647


namespace NUMINAMATH_CALUDE_bug_total_distance_l916_91670

def bug_path : List ℤ := [4, -3, 6, 2]

def distance (a b : ℤ) : ℕ := (a - b).natAbs

theorem bug_total_distance :
  (List.zip bug_path bug_path.tail).foldl (λ acc (a, b) => acc + distance a b) 0 = 20 :=
by sorry

end NUMINAMATH_CALUDE_bug_total_distance_l916_91670


namespace NUMINAMATH_CALUDE_m_intersect_n_eq_m_l916_91606

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def N : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem m_intersect_n_eq_m : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_m_intersect_n_eq_m_l916_91606


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l916_91681

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 2) = (a^2 - 3*a + 2) + Complex.I * (a - 2)) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l916_91681


namespace NUMINAMATH_CALUDE_descendants_characterization_l916_91607

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The set of descendants of 1 -/
inductive Descendant : ℚ → Prop where
  | base : Descendant 1
  | left (x : ℚ) : Descendant x → Descendant (x + 1)
  | right (x : ℚ) : Descendant x → Descendant (x / (x + 1))

/-- Theorem: All descendants of 1 are of the form F_(n±1) / F_n, where n > 1 -/
theorem descendants_characterization (q : ℚ) :
  Descendant q ↔ ∃ n : ℕ, n > 1 ∧ (q = (fib (n + 1) : ℚ) / fib n ∨ q = (fib (n - 1) : ℚ) / fib n) :=
sorry

end NUMINAMATH_CALUDE_descendants_characterization_l916_91607


namespace NUMINAMATH_CALUDE_equation_solution_l916_91662

theorem equation_solution :
  ∃ x : ℝ, 3 * (16 : ℝ)^x + 37 * (36 : ℝ)^x = 26 * (81 : ℝ)^x ∧ x = (1/2 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l916_91662


namespace NUMINAMATH_CALUDE_rachel_apples_remaining_l916_91635

/-- The number of apples remaining on trees after picking -/
def apples_remaining (num_trees : ℕ) (apples_per_tree : ℕ) (initial_total : ℕ) : ℕ :=
  initial_total - (num_trees * apples_per_tree)

/-- Theorem: The number of apples remaining on Rachel's trees is 9 -/
theorem rachel_apples_remaining :
  apples_remaining 3 8 33 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apples_remaining_l916_91635


namespace NUMINAMATH_CALUDE_yolanda_three_point_average_l916_91682

theorem yolanda_three_point_average (total_points season_games free_throws_per_game two_point_baskets_per_game : ℕ)
  (h1 : total_points = 345)
  (h2 : season_games = 15)
  (h3 : free_throws_per_game = 4)
  (h4 : two_point_baskets_per_game = 5) :
  (total_points - (free_throws_per_game * 1 + two_point_baskets_per_game * 2) * season_games) / (3 * season_games) = 3 := by
  sorry

end NUMINAMATH_CALUDE_yolanda_three_point_average_l916_91682


namespace NUMINAMATH_CALUDE_valid_parameterizations_l916_91639

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 2 * x - 4

-- Define the parameterizations
def param_A (t : ℝ) : ℝ × ℝ := (2 - t, -2 * t)
def param_B (t : ℝ) : ℝ × ℝ := (5 * t, 10 * t - 4)
def param_C (t : ℝ) : ℝ × ℝ := (-1 + 2 * t, -6 + 4 * t)
def param_D (t : ℝ) : ℝ × ℝ := (3 + t, 2 + 3 * t)
def param_E (t : ℝ) : ℝ × ℝ := (-4 - 2 * t, -12 - 4 * t)

-- Theorem stating which parameterizations are valid
theorem valid_parameterizations :
  (∀ t, line_equation (param_A t).1 (param_A t).2) ∧
  (∀ t, line_equation (param_B t).1 (param_B t).2) ∧
  ¬(∀ t, line_equation (param_C t).1 (param_C t).2) ∧
  ¬(∀ t, line_equation (param_D t).1 (param_D t).2) ∧
  ¬(∀ t, line_equation (param_E t).1 (param_E t).2) := by
  sorry

end NUMINAMATH_CALUDE_valid_parameterizations_l916_91639


namespace NUMINAMATH_CALUDE_coin_toss_probability_l916_91625

/-- The number of coin tosses -/
def n : ℕ := 5

/-- The number of heads -/
def k : ℕ := 4

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def binomial_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1/2)^n

theorem coin_toss_probability :
  binomial_probability n k = 5/32 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l916_91625


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l916_91696

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  ((-1 : ℝ) / (1 + m) = -m / 2) ∧ (m ≠ -2)

/-- The value of m for which the lines x + (1+m)y = 2-m and mx + 2y + 8 = 0 are parallel -/
theorem parallel_lines_m_value : ∃! m : ℝ, parallel_lines m :=
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l916_91696


namespace NUMINAMATH_CALUDE_cross_section_area_l916_91614

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_side : ℝ
  height : ℝ

/-- Plane that intersects the pyramid -/
structure IntersectingPlane where
  perpendicular_to_base : Prop
  bisects_two_sides : Prop

/-- The cross-section created by the intersecting plane -/
def cross_section (p : RegularTriangularPyramid) (plane : IntersectingPlane) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The area of a given set in 3D space -/
def area (s : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area 
  (p : RegularTriangularPyramid) 
  (plane : IntersectingPlane) 
  (h1 : p.base_side = 2) 
  (h2 : p.height = 4) 
  (h3 : plane.perpendicular_to_base) 
  (h4 : plane.bisects_two_sides) : 
  area (cross_section p plane) = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_cross_section_area_l916_91614


namespace NUMINAMATH_CALUDE_stating_fencers_count_correct_l916_91636

/-- The number of fencers participating in the championship. -/
def num_fencers : ℕ := 9

/-- The number of possibilities for awarding first and second place medals. -/
def num_possibilities : ℕ := 72

/-- 
Theorem stating that the number of fencers is correct given the number of possibilities 
for awarding first and second place medals.
-/
theorem fencers_count_correct : 
  num_fencers * (num_fencers - 1) = num_possibilities := by
  sorry

#check fencers_count_correct

end NUMINAMATH_CALUDE_stating_fencers_count_correct_l916_91636
