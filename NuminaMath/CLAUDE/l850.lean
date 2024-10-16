import Mathlib

namespace NUMINAMATH_CALUDE_proportion_equality_l850_85047

theorem proportion_equality (a b c d : ℝ) (h : a / b = c / d) :
  (a + c) / c = (b + d) / d := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l850_85047


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l850_85012

theorem consecutive_non_prime_powers (k : ℕ+) :
  ∃ (n : ℕ), ∀ (i : ℕ), i ∈ Finset.range k →
    ¬∃ (p : ℕ) (e : ℕ), Nat.Prime p ∧ (n + i = p ^ e) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l850_85012


namespace NUMINAMATH_CALUDE_chess_group_size_l850_85048

/-- The number of games played when n players each play every other player once -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that 14 players result in 91 games when each player plays every other player once -/
theorem chess_group_size :
  ∃ (n : ℕ), n > 0 ∧ gamesPlayed n = 91 ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_chess_group_size_l850_85048


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_l850_85073

/-- The volume of a cube with side length s -/
def cube_volume (s : ℕ) : ℕ := s^3

/-- The total volume of n cubes with side length s -/
def total_volume (n : ℕ) (s : ℕ) : ℕ := n * cube_volume s

/-- Carl's cubes -/
def carl_cubes : ℕ × ℕ := (4, 3)

/-- Kate's cubes -/
def kate_cubes : ℕ × ℕ := (6, 1)

theorem total_volume_of_cubes :
  total_volume carl_cubes.1 carl_cubes.2 + total_volume kate_cubes.1 kate_cubes.2 = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_cubes_l850_85073


namespace NUMINAMATH_CALUDE_rectangle_area_l850_85093

/-- Proves that the area of a rectangle with length 3 times its width and width of 4 inches is 48 square inches -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 4 →
  length = 3 * width →
  area = length * width →
  area = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l850_85093


namespace NUMINAMATH_CALUDE_seashell_count_l850_85078

theorem seashell_count (sam_shells joan_shells : ℕ) 
  (h1 : sam_shells = 35) 
  (h2 : joan_shells = 18) : 
  sam_shells + joan_shells = 53 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_l850_85078


namespace NUMINAMATH_CALUDE_white_balls_count_l850_85001

/-- The number of balls in the bag -/
def total_balls : ℕ := 7

/-- The expected number of white balls when drawing 2 balls -/
def expected_white : ℚ := 6/7

/-- Calculates the expected number of white balls drawn -/
def calculate_expected (white_balls : ℕ) : ℚ :=
  (Nat.choose white_balls 2 * 2 + Nat.choose white_balls 1 * Nat.choose (total_balls - white_balls) 1) / Nat.choose total_balls 2

/-- Theorem stating that the number of white balls is 3 -/
theorem white_balls_count :
  ∃ (n : ℕ), n < total_balls ∧ calculate_expected n = expected_white ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l850_85001


namespace NUMINAMATH_CALUDE_range_of_f_l850_85056

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -3 ≤ y ∧ y ≤ 14 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l850_85056


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l850_85068

theorem cubic_equation_solutions (x : ℝ) : 
  2.21 * (((5 + x)^2)^(1/3)) + 4 * (((5 - x)^2)^(1/3)) = 5 * ((25 - x)^(1/3)) ↔ 
  x = 0 ∨ x = 63/13 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l850_85068


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l850_85025

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (x - a) * (x + 1 - a) ≥ 0 → x ≠ 1) → 
  a > 1 ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l850_85025


namespace NUMINAMATH_CALUDE_cells_after_one_week_l850_85010

/-- The number of cells after n days, given that each cell divides into three new cells every day -/
def num_cells (n : ℕ) : ℕ := 3^n

/-- Theorem: After 7 days, there will be 2187 cells -/
theorem cells_after_one_week : num_cells 7 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_cells_after_one_week_l850_85010


namespace NUMINAMATH_CALUDE_diego_payment_is_9800_l850_85042

def total_payment : ℝ := 50000
def celina_payment (diego_payment : ℝ) : ℝ := 1000 + 4 * diego_payment

theorem diego_payment_is_9800 :
  ∃ (diego_payment : ℝ),
    diego_payment + celina_payment diego_payment = total_payment ∧
    diego_payment = 9800 :=
by sorry

end NUMINAMATH_CALUDE_diego_payment_is_9800_l850_85042


namespace NUMINAMATH_CALUDE_triangle_in_radius_l850_85020

/-- Given a triangle with perimeter 36 cm and area 45 cm², prove that its in radius is 2.5 cm. -/
theorem triangle_in_radius (P : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : P = 36) 
  (h_area : A = 45) 
  (h_in_radius : A = r * (P / 2)) : r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_in_radius_l850_85020


namespace NUMINAMATH_CALUDE_integer_triple_sum_product_l850_85085

theorem integer_triple_sum_product : 
  ∀ a b c : ℕ+, 
    (a + b + c = 25 ∧ a * b * c = 360) ↔ 
    ((a = 4 ∧ b = 6 ∧ c = 15) ∨ 
     (a = 3 ∧ b = 10 ∧ c = 12) ∨
     (a = 4 ∧ b = 15 ∧ c = 6) ∨ 
     (a = 6 ∧ b = 4 ∧ c = 15) ∨
     (a = 6 ∧ b = 15 ∧ c = 4) ∨
     (a = 15 ∧ b = 4 ∧ c = 6) ∨
     (a = 15 ∧ b = 6 ∧ c = 4) ∨
     (a = 3 ∧ b = 12 ∧ c = 10) ∨
     (a = 10 ∧ b = 3 ∧ c = 12) ∨
     (a = 10 ∧ b = 12 ∧ c = 3) ∨
     (a = 12 ∧ b = 3 ∧ c = 10) ∨
     (a = 12 ∧ b = 10 ∧ c = 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_sum_product_l850_85085


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_l850_85092

theorem complex_magnitude_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_l850_85092


namespace NUMINAMATH_CALUDE_min_height_box_l850_85006

def box_height (side_length : ℝ) : ℝ := 2 * side_length

def surface_area (side_length : ℝ) : ℝ := 10 * side_length^2

theorem min_height_box (min_area : ℝ) (h_min_area : min_area = 120) :
  ∃ (h : ℝ), h = box_height (Real.sqrt (min_area / 10)) ∧
             h = 8 ∧
             ∀ (s : ℝ), surface_area s ≥ min_area → box_height s ≥ h :=
by sorry

end NUMINAMATH_CALUDE_min_height_box_l850_85006


namespace NUMINAMATH_CALUDE_solve_equation_l850_85000

theorem solve_equation (x y : ℝ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l850_85000


namespace NUMINAMATH_CALUDE_water_leaked_calculation_l850_85019

/-- The amount of water that leaked out of a bucket -/
def water_leaked (initial : ℝ) (remaining : ℝ) : ℝ :=
  initial - remaining

theorem water_leaked_calculation (initial : ℝ) (remaining : ℝ) 
  (h1 : initial = 0.75)
  (h2 : remaining = 0.5) : 
  water_leaked initial remaining = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_water_leaked_calculation_l850_85019


namespace NUMINAMATH_CALUDE_f_max_min_difference_l850_85031

noncomputable def f (x : ℝ) : ℝ := 4 * Real.pi * Real.arcsin x - (Real.arccos (-x))^2

theorem f_max_min_difference :
  ∃ (M m : ℝ), (∀ x : ℝ, f x ≤ M ∧ f x ≥ m) ∧ M - m = 3 * Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_difference_l850_85031


namespace NUMINAMATH_CALUDE_card_drawing_combinations_l850_85016

-- Define the number of piles and cards per pile
def num_piles : ℕ := 3
def cards_per_pile : ℕ := 3

-- Define the total number of cards
def total_cards : ℕ := num_piles * cards_per_pile

-- Define the function to calculate the number of ways to draw the cards
def ways_to_draw_cards : ℕ := (Nat.factorial total_cards) / ((Nat.factorial cards_per_pile) ^ num_piles)

-- Theorem statement
theorem card_drawing_combinations :
  ways_to_draw_cards = 1680 :=
sorry

end NUMINAMATH_CALUDE_card_drawing_combinations_l850_85016


namespace NUMINAMATH_CALUDE_blue_highlighters_count_l850_85071

/-- The number of blue highlighters in Kaya's teacher's desk -/
def blue_highlighters : ℕ := 33 - (10 + 15)

/-- Theorem stating the number of blue highlighters -/
theorem blue_highlighters_count :
  blue_highlighters = 8 := by sorry

end NUMINAMATH_CALUDE_blue_highlighters_count_l850_85071


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l850_85095

theorem jelly_bean_probability (p_red p_orange p_green p_yellow : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_green = 0.25 →
  p_red + p_orange + p_green + p_yellow = 1 →
  p_yellow = 0.25 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l850_85095


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l850_85088

/-- Calculates the corrected mean of a set of observations after fixing an error -/
def corrected_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean - wrong_value + correct_value) / n

/-- Theorem stating that the corrected mean for the given problem is 36.42 -/
theorem corrected_mean_problem : 
  corrected_mean 50 36 23 44 = 36.42 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_problem_l850_85088


namespace NUMINAMATH_CALUDE_transformer_load_calculation_l850_85052

/-- Calculates the minimum current load for a transformer given the number of units,
    running current per unit, and the starting current multiplier. -/
def minTransformerLoad (numUnits : ℕ) (runningCurrent : ℕ) (startingMultiplier : ℕ) : ℕ :=
  numUnits * (startingMultiplier * runningCurrent)

theorem transformer_load_calculation :
  let numUnits : ℕ := 3
  let runningCurrent : ℕ := 40
  let startingMultiplier : ℕ := 2
  minTransformerLoad numUnits runningCurrent startingMultiplier = 240 := by
  sorry

#eval minTransformerLoad 3 40 2

end NUMINAMATH_CALUDE_transformer_load_calculation_l850_85052


namespace NUMINAMATH_CALUDE_equal_distance_trajectory_length_l850_85097

/-- Rectilinear distance between two points -/
def rectilinearDistance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- The set of points C(x, y) with equal rectilinear distance to A and B -/
def equalDistancePoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               0 ≤ x ∧ x ≤ 10 ∧ 0 ≤ y ∧ y ≤ 10 ∧
               rectilinearDistance x y 1 3 = rectilinearDistance x y 6 9}

/-- The sum of the lengths of the trajectories of all points in equalDistancePoints -/
noncomputable def trajectoryLength : ℝ :=
  5 * (Real.sqrt 2 + 1)

theorem equal_distance_trajectory_length :
  trajectoryLength = 5 * (Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_equal_distance_trajectory_length_l850_85097


namespace NUMINAMATH_CALUDE_vector_sum_equals_result_l850_85041

def vector_a : ℝ × ℝ := (0, -1)
def vector_b : ℝ × ℝ := (3, 2)

theorem vector_sum_equals_result : 2 • vector_a + vector_b = (3, 0) := by sorry

end NUMINAMATH_CALUDE_vector_sum_equals_result_l850_85041


namespace NUMINAMATH_CALUDE_max_diff_divisible_sum_digits_l850_85079

def sum_of_digits (n : ℕ) : ℕ := sorry

def has_divisible_sum_between (a b : ℕ) : Prop :=
  ∃ k, a < k ∧ k < b ∧ sum_of_digits k % 7 = 0

theorem max_diff_divisible_sum_digits :
  ∃ a b : ℕ, sum_of_digits a % 7 = 0 ∧
             sum_of_digits b % 7 = 0 ∧
             b - a = 13 ∧
             ¬ has_divisible_sum_between a b ∧
             ∀ c d : ℕ, sum_of_digits c % 7 = 0 →
                        sum_of_digits d % 7 = 0 →
                        ¬ has_divisible_sum_between c d →
                        d - c ≤ 13 := by sorry

end NUMINAMATH_CALUDE_max_diff_divisible_sum_digits_l850_85079


namespace NUMINAMATH_CALUDE_contractor_fine_calculation_l850_85008

/-- Calculates the daily fine for a contractor given contract details -/
def calculate_daily_fine (contract_duration : ℕ) (daily_pay : ℕ) (total_payment : ℕ) (days_absent : ℕ) : ℚ :=
  let days_worked := contract_duration - days_absent
  let total_earned := days_worked * daily_pay
  ((total_earned - total_payment) : ℚ) / days_absent

theorem contractor_fine_calculation :
  let contract_duration : ℕ := 30
  let daily_pay : ℕ := 25
  let total_payment : ℕ := 425
  let days_absent : ℕ := 10
  calculate_daily_fine contract_duration daily_pay total_payment days_absent = 15/2 := by
  sorry

#eval calculate_daily_fine 30 25 425 10

end NUMINAMATH_CALUDE_contractor_fine_calculation_l850_85008


namespace NUMINAMATH_CALUDE_table_movement_l850_85094

theorem table_movement (table_width : ℝ) (table_length : ℝ) : 
  table_width = 8 ∧ table_length = 10 →
  ∃ (S : ℕ), S = 13 ∧ 
  (∀ (T : ℕ), T < S → Real.sqrt (table_width^2 + table_length^2) > T) ∧
  Real.sqrt (table_width^2 + table_length^2) ≤ S :=
by sorry

end NUMINAMATH_CALUDE_table_movement_l850_85094


namespace NUMINAMATH_CALUDE_joan_grew_29_carrots_l850_85028

/-- The number of carrots Joan grew -/
def joans_carrots : ℕ := sorry

/-- The number of watermelons Joan grew -/
def joans_watermelons : ℕ := 14

/-- The number of carrots Jessica grew -/
def jessicas_carrots : ℕ := 11

/-- The total number of carrots grown by Joan and Jessica -/
def total_carrots : ℕ := 40

/-- Theorem stating that Joan grew 29 carrots -/
theorem joan_grew_29_carrots : joans_carrots = 29 := by
  sorry

end NUMINAMATH_CALUDE_joan_grew_29_carrots_l850_85028


namespace NUMINAMATH_CALUDE_rectangle_to_rhombus_l850_85030

-- Define a rectangle
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (width_pos : width > 0)
  (height_pos : height > 0)

-- Define a rhombus
structure Rhombus :=
  (side : ℝ)
  (side_pos : side > 0)

-- Define the theorem
theorem rectangle_to_rhombus (r : Rectangle) : 
  ∃ (rh : Rhombus), r.width * r.height = 4 * rh.side^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_rhombus_l850_85030


namespace NUMINAMATH_CALUDE_bicycle_trip_speed_l850_85046

/-- Proves that given a total distance of 400 km, with the first 100 km traveled at 20 km/h
    and an average speed of 16 km/h for the entire trip, the speed for the remainder of the trip is 15 km/h. -/
theorem bicycle_trip_speed (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (average_speed : ℝ)
  (h1 : total_distance = 400)
  (h2 : first_part_distance = 100)
  (h3 : first_part_speed = 20)
  (h4 : average_speed = 16) :
  let remainder_distance := total_distance - first_part_distance
  let total_time := total_distance / average_speed
  let first_part_time := first_part_distance / first_part_speed
  let remainder_time := total_time - first_part_time
  remainder_distance / remainder_time = 15 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_trip_speed_l850_85046


namespace NUMINAMATH_CALUDE_subgroup_samples_is_ten_l850_85034

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  subgroup_size : ℕ
  total_samples : ℕ
  subgroup_samples : ℕ

/-- Calculates the number of samples from a subgroup in stratified sampling -/
def calculate_subgroup_samples (s : StratifiedSample) : ℚ :=
  s.total_samples * (s.subgroup_size : ℚ) / s.total_population

/-- Theorem stating that for the given scenario, the number of subgroup samples is 10 -/
theorem subgroup_samples_is_ten : 
  let s : StratifiedSample := {
    total_population := 1200,
    subgroup_size := 200,
    total_samples := 60,
    subgroup_samples := 10
  }
  calculate_subgroup_samples s = 10 := by
  sorry


end NUMINAMATH_CALUDE_subgroup_samples_is_ten_l850_85034


namespace NUMINAMATH_CALUDE_treasure_hunt_probability_l850_85033

def probability_gold : ℚ := 1 / 5
def probability_danger : ℚ := 1 / 10
def probability_neither : ℚ := 4 / 5
def total_caves : ℕ := 5
def gold_caves : ℕ := 2

theorem treasure_hunt_probability :
  (Nat.choose total_caves gold_caves : ℚ) *
  probability_gold ^ gold_caves *
  probability_neither ^ (total_caves - gold_caves) =
  128 / 625 := by sorry

end NUMINAMATH_CALUDE_treasure_hunt_probability_l850_85033


namespace NUMINAMATH_CALUDE_triangle_side_length_l850_85017

theorem triangle_side_length (A B C : ℝ) (AB BC AC : ℝ) : 
  BC = 1 → 
  A = π / 3 → 
  Real.sin B = 2 * Real.sin C → 
  AB = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l850_85017


namespace NUMINAMATH_CALUDE_rudolph_stop_signs_l850_85065

/-- Calculates the number of stop signs encountered on a car trip -/
def stop_signs_encountered (base_distance : ℕ) (additional_distance : ℕ) (signs_per_mile : ℕ) : ℕ :=
  (base_distance + additional_distance) * signs_per_mile

/-- Theorem: Rudolph encountered 14 stop signs on his trip -/
theorem rudolph_stop_signs :
  stop_signs_encountered 5 2 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_rudolph_stop_signs_l850_85065


namespace NUMINAMATH_CALUDE_simultaneous_inequality_condition_l850_85009

theorem simultaneous_inequality_condition (a : ℝ) : 
  (∃ x₀ : ℝ, x₀^2 - a*x₀ + a + 3 < 0 ∧ a*x₀ - 2*a < 0) ↔ a > 7 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_inequality_condition_l850_85009


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l850_85083

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- The point P with coordinates (-3, a^2 + 1) lies in the second quadrant for any real number a. -/
theorem point_in_second_quadrant (a : ℝ) : second_quadrant (-3, a^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l850_85083


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l850_85098

-- Define variables
variable (x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 :
  3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 :
  3 * x^2 * y - (2 * x * y - 2 * (x * y - 3/2 * x^2 * y) + x^2 * y^2) = -x^2 * y^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l850_85098


namespace NUMINAMATH_CALUDE_original_weight_correct_l850_85061

/-- Represents the original weight Tom could lift per hand in kg -/
def original_weight : ℝ := 80

/-- Represents the total weight Tom can hold with both hands after training in kg -/
def total_weight : ℝ := 352

/-- Theorem stating that the original weight satisfies the given conditions -/
theorem original_weight_correct : 
  2 * (2 * original_weight * 1.1) = total_weight := by sorry

end NUMINAMATH_CALUDE_original_weight_correct_l850_85061


namespace NUMINAMATH_CALUDE_necessary_condition_not_sufficient_l850_85058

def f (x : ℝ) := |x - 2| + |x + 3|

def proposition_p (a : ℝ) := ∃ x, f x < a

theorem necessary_condition (a : ℝ) :
  (¬ proposition_p a) → a ≥ 5 := by sorry

theorem not_sufficient (a : ℝ) :
  ∃ a, a ≥ 5 ∧ proposition_p a := by sorry

end NUMINAMATH_CALUDE_necessary_condition_not_sufficient_l850_85058


namespace NUMINAMATH_CALUDE_right_triangle_height_l850_85027

/-- Given a right triangle and a rectangle with the same area, where the base of the triangle
    equals the width of the rectangle (5 units), and the area of the rectangle is 45 square units,
    prove that the height of the right triangle is 18 units. -/
theorem right_triangle_height (base width : ℝ) (rect_area tri_area : ℝ) (h : ℝ) : 
  base = width →
  base = 5 →
  rect_area = 45 →
  rect_area = base * (rect_area / base) →
  tri_area = (1 / 2) * base * h →
  rect_area = tri_area →
  h = 18 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_height_l850_85027


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l850_85066

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_x_1 :
  ∃ (m c : ℝ), 
    (∀ x y : ℝ, y = m * x + c ↔ m * x - y + c = 0) ∧
    (m = f' 1) ∧
    (f 1 = m * 1 + c) ∧
    (m * x - y + c = 0 ↔ 4 * x - y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l850_85066


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l850_85062

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + x.repeatingPart / (99 : ℚ)

/-- The repeating decimal 0.overline{45} -/
def a : RepeatingDecimal := ⟨0, 45⟩

/-- The repeating decimal 2.overline{18} -/
def b : RepeatingDecimal := ⟨2, 18⟩

/-- Theorem stating that the ratio of the given repeating decimals equals 5/24 -/
theorem repeating_decimal_ratio : (toRational a) / (toRational b) = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l850_85062


namespace NUMINAMATH_CALUDE_trigonometric_identity_angle_relation_l850_85082

-- Part 1
theorem trigonometric_identity :
  Real.sin (120 * π / 180) ^ 2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) -
  Real.cos (-330 * π / 180) ^ 2 + Real.sin (-210 * π / 180) = 1 / 2 := by sorry

-- Part 2
theorem angle_relation (α β : Real) (h1 : 0 < α) (h2 : α < π) (h3 : 0 < β) (h4 : β < π)
  (h5 : Real.tan (α - β) = 1 / 2) (h6 : Real.tan β = -1 / 7) :
  2 * α - β = -3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_angle_relation_l850_85082


namespace NUMINAMATH_CALUDE_quadratic_factorization_l850_85050

theorem quadratic_factorization (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l850_85050


namespace NUMINAMATH_CALUDE_jana_walking_distance_l850_85051

/-- Jana's walking pattern and distance traveled -/
theorem jana_walking_distance :
  let usual_pace : ℚ := 1 / 30  -- miles per minute
  let half_pace : ℚ := usual_pace / 2
  let double_pace : ℚ := usual_pace * 2
  let first_15_min_distance : ℚ := half_pace * 15
  let next_5_min_distance : ℚ := double_pace * 5
  first_15_min_distance + next_5_min_distance = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_jana_walking_distance_l850_85051


namespace NUMINAMATH_CALUDE_reciprocal_sum_inequality_quadratic_inequality_range_l850_85002

variable (a b c : ℝ)

-- Define the conditions
def sum_condition (a b c : ℝ) : Prop := a + b + c = 3
def positive_condition (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0

-- Theorem 1
theorem reciprocal_sum_inequality (h1 : sum_condition a b c) (h2 : positive_condition a b c) :
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by sorry

-- Theorem 2
theorem quadratic_inequality_range (h1 : sum_condition a b c) (h2 : positive_condition a b c) :
  ∀ m : ℝ, (∀ x : ℝ, -x^2 + m*x + 2 ≤ a^2 + b^2 + c^2) ↔ -2 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_inequality_quadratic_inequality_range_l850_85002


namespace NUMINAMATH_CALUDE_smallest_common_nondivisor_l850_85038

theorem smallest_common_nondivisor : 
  ∃ (a : ℕ), a > 0 ∧ 
  (∀ (k : ℕ), 0 < k ∧ k < a → (Nat.gcd k 77 = 1 ∨ Nat.gcd k 66 = 1)) ∧ 
  Nat.gcd a 77 > 1 ∧ Nat.gcd a 66 > 1 ∧ 
  a = 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_nondivisor_l850_85038


namespace NUMINAMATH_CALUDE_two_digit_number_property_l850_85032

theorem two_digit_number_property : 
  let n : ℕ := 27
  let tens_digit : ℕ := n / 10
  let units_digit : ℕ := n % 10
  (units_digit = tens_digit + 5) →
  (n * (tens_digit + units_digit) = 243) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l850_85032


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_relation_l850_85096

theorem quadratic_equation_roots_relation (p : ℚ) : 
  (∃ x1 x2 : ℚ, 3 * x1^2 - 5*(p-1)*x1 + p^2 + 2 = 0 ∧
                3 * x2^2 - 5*(p-1)*x2 + p^2 + 2 = 0 ∧
                x1 + 4*x2 = 14) ↔ 
  (p = 742/127 ∨ p = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_relation_l850_85096


namespace NUMINAMATH_CALUDE_max_value_theorem_l850_85026

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  2 * a * c * Real.sqrt 2 + 2 * a * b ≤ Real.sqrt 3 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ 
  a'^2 + b'^2 + c'^2 = 1 ∧ 
  2 * a' * c' * Real.sqrt 2 + 2 * a' * b' = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l850_85026


namespace NUMINAMATH_CALUDE_business_partnership_problem_l850_85086

/-- A business partnership problem -/
theorem business_partnership_problem
  (x_capital y_capital z_capital : ℕ)
  (total_profit z_profit : ℕ)
  (x_months y_months z_months : ℕ)
  (h1 : x_capital = 20000)
  (h2 : z_capital = 30000)
  (h3 : total_profit = 50000)
  (h4 : z_profit = 14000)
  (h5 : x_months = 12)
  (h6 : y_months = 12)
  (h7 : z_months = 7)
  (h8 : (z_capital * z_months) / (x_capital * x_months + y_capital * y_months + z_capital * z_months) = z_profit / total_profit) :
  y_capital = 25000 := by
  sorry


end NUMINAMATH_CALUDE_business_partnership_problem_l850_85086


namespace NUMINAMATH_CALUDE_unique_assignment_l850_85067

-- Define the polyhedron structure
structure Polyhedron :=
  (faces : Fin 2022 → ℝ)
  (adjacent : Fin 2022 → Finset (Fin 2022))
  (adjacent_symmetric : ∀ i j, j ∈ adjacent i ↔ i ∈ adjacent j)

-- Define the property of being a valid number assignment
def ValidAssignment (p : Polyhedron) : Prop :=
  ∀ i, p.faces i = if i = 0 then 26
                   else if i = 1 then 4
                   else if i = 2 then 2022
                   else (p.adjacent i).sum p.faces / (p.adjacent i).card

-- Theorem statement
theorem unique_assignment (p : Polyhedron) :
  ∃! f : Fin 2022 → ℝ, ValidAssignment { faces := f, adjacent := p.adjacent, adjacent_symmetric := p.adjacent_symmetric } :=
sorry

end NUMINAMATH_CALUDE_unique_assignment_l850_85067


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l850_85054

theorem quadratic_equation_root (x : ℝ) : x^2 + 6*x - 4 = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l850_85054


namespace NUMINAMATH_CALUDE_company_b_price_l850_85074

/-- Proves that Company B's bottle price is $3.50 given the problem conditions -/
theorem company_b_price (company_a_price : ℝ) (company_a_sold : ℕ) (company_b_sold : ℕ) 
  (revenue_difference : ℝ) :
  company_a_price = 4 →
  company_a_sold = 300 →
  company_b_sold = 350 →
  revenue_difference = 25 →
  ∃ (company_b_price : ℝ), 
    company_b_price = 3.5 ∧ 
    (company_b_sold : ℝ) * company_b_price = (company_a_sold : ℝ) * company_a_price + revenue_difference :=
by sorry

end NUMINAMATH_CALUDE_company_b_price_l850_85074


namespace NUMINAMATH_CALUDE_problem_figure_perimeter_l850_85077

/-- Represents a figure made of unit squares -/
structure UnitSquareFigure where
  bottom_row : Nat
  left_column : Nat
  top_row : Nat
  right_column : Nat

/-- The specific figure described in the problem -/
def problem_figure : UnitSquareFigure :=
  { bottom_row := 3
  , left_column := 2
  , top_row := 4
  , right_column := 3 }

/-- Calculates the perimeter of a UnitSquareFigure -/
def perimeter (figure : UnitSquareFigure) : Nat :=
  figure.bottom_row + figure.left_column + figure.top_row + figure.right_column

theorem problem_figure_perimeter : perimeter problem_figure = 12 := by
  sorry

#eval perimeter problem_figure

end NUMINAMATH_CALUDE_problem_figure_perimeter_l850_85077


namespace NUMINAMATH_CALUDE_valid_numbers_characterization_l850_85003

/-- A function that moves the last digit of a number to the beginning -/
def moveLastDigitToFront (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  let remainingDigits := n / 10
  lastDigit * 10^5 + remainingDigits

/-- A predicate that checks if a number becomes an integer multiple when its last digit is moved to the front -/
def isValidNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, moveLastDigitToFront n = k * n

/-- The set of all valid six-digit numbers -/
def validNumbers : Finset ℕ :=
  {142857, 102564, 128205, 153846, 179487, 205128, 230769}

/-- The main theorem stating that validNumbers contains all and only the six-digit numbers
    that become an integer multiple when the last digit is moved to the beginning -/
theorem valid_numbers_characterization :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 →
    (n ∈ validNumbers ↔ isValidNumber n) := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_characterization_l850_85003


namespace NUMINAMATH_CALUDE_solve_equation_and_expression_l850_85045

theorem solve_equation_and_expression (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 
  3 * (x - 4) + 2 = -16 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_and_expression_l850_85045


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l850_85011

theorem intersection_point_x_coordinate :
  let line1 : ℝ → ℝ := λ x => 3 * x - 22
  let line2 : ℝ → ℝ := λ x => 100 - 3 * x
  ∃ x : ℝ, line1 x = line2 x ∧ x = 61 / 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l850_85011


namespace NUMINAMATH_CALUDE_natural_number_operations_l850_85044

theorem natural_number_operations (x y : ℕ) (h1 : x > y) (h2 : x + y + (x - y) + x * y + x / y = 243) :
  (x = 54 ∧ y = 2) ∨ (x = 24 ∧ y = 8) := by
  sorry

end NUMINAMATH_CALUDE_natural_number_operations_l850_85044


namespace NUMINAMATH_CALUDE_problem_solution_l850_85089

theorem problem_solution : (10^3 - (270 * (1/3))) + Real.sqrt 144 = 922 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l850_85089


namespace NUMINAMATH_CALUDE_count_odd_numbers_300_to_600_l850_85023

theorem count_odd_numbers_300_to_600 : 
  (Finset.filter (fun n => n % 2 = 1) (Finset.Icc 300 600)).card = 150 := by
  sorry

end NUMINAMATH_CALUDE_count_odd_numbers_300_to_600_l850_85023


namespace NUMINAMATH_CALUDE_arithmetic_expression_result_l850_85005

theorem arithmetic_expression_result : 3 + 15 / 3 - 2^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_result_l850_85005


namespace NUMINAMATH_CALUDE_garden_trees_l850_85075

/-- The number of trees in a garden with given specifications -/
def num_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  yard_length / tree_distance + 1

/-- Theorem: The number of trees in a 500-metre garden with 20-metre spacing is 26 -/
theorem garden_trees : num_trees 500 20 = 26 := by
  sorry

end NUMINAMATH_CALUDE_garden_trees_l850_85075


namespace NUMINAMATH_CALUDE_solution_set_min_value_m_plus_n_l850_85040

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x - 2|

-- Part 1: Solution set of f(x) ≥ 3
theorem solution_set (x : ℝ) : f x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2 := by sorry

-- Part 2: Minimum value of m+n
theorem min_value_m_plus_n (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : ∀ x, f x ≥ 1/m + 1/n) : 
  m + n ≥ 8/3 ∧ (m + n = 8/3 ↔ m = n) := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_m_plus_n_l850_85040


namespace NUMINAMATH_CALUDE_correct_sticker_count_l850_85014

/-- Represents the number of stickers per page for each type -/
def stickers_per_page : Fin 4 → ℕ
  | 0 => 5  -- Type A
  | 1 => 3  -- Type B
  | 2 => 2  -- Type C
  | 3 => 1  -- Type D

/-- The total number of pages -/
def total_pages : ℕ := 22

/-- Calculates the total number of stickers for a given type -/
def total_stickers (type : Fin 4) : ℕ :=
  (stickers_per_page type) * total_pages

/-- Theorem stating the correct total number of stickers for each type -/
theorem correct_sticker_count :
  (total_stickers 0 = 110) ∧
  (total_stickers 1 = 66) ∧
  (total_stickers 2 = 44) ∧
  (total_stickers 3 = 22) := by
  sorry

end NUMINAMATH_CALUDE_correct_sticker_count_l850_85014


namespace NUMINAMATH_CALUDE_digit_1500_is_1_l850_85080

/-- The fraction we're considering -/
def f : ℚ := 7/22

/-- The length of the repeating cycle in the decimal expansion of f -/
def cycle_length : ℕ := 6

/-- The position of the digit we're looking for -/
def target_position : ℕ := 1500

/-- The function that returns the nth digit after the decimal point
    in the decimal expansion of f -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_1500_is_1 : nth_digit target_position = 1 := by sorry

end NUMINAMATH_CALUDE_digit_1500_is_1_l850_85080


namespace NUMINAMATH_CALUDE_martha_savings_l850_85024

def daily_allowance : ℚ := 12
def normal_saving_rate : ℚ := 1/2
def exception_saving_rate : ℚ := 1/4
def days_in_week : ℕ := 7
def normal_saving_days : ℕ := 6
def exception_saving_days : ℕ := 1

theorem martha_savings : 
  (normal_saving_days : ℚ) * (daily_allowance * normal_saving_rate) + 
  (exception_saving_days : ℚ) * (daily_allowance * exception_saving_rate) = 39 := by
  sorry

end NUMINAMATH_CALUDE_martha_savings_l850_85024


namespace NUMINAMATH_CALUDE_olympic_mascot_pricing_and_purchasing_l850_85037

theorem olympic_mascot_pricing_and_purchasing
  (small_price large_price : ℝ)
  (h1 : large_price - 2 * small_price = 20)
  (h2 : 3 * small_price + 2 * large_price = 390)
  (budget : ℝ) (total_sets : ℕ)
  (h3 : budget = 1500)
  (h4 : total_sets = 20) :
  small_price = 50 ∧ 
  large_price = 120 ∧ 
  (∃ m : ℕ, m ≤ total_sets ∧ 
    m * large_price + (total_sets - m) * small_price ≤ budget ∧
    ∀ n : ℕ, n > m → n * large_price + (total_sets - n) * small_price > budget) ∧
  (7 : ℕ) * large_price + (total_sets - 7) * small_price ≤ budget :=
by sorry

end NUMINAMATH_CALUDE_olympic_mascot_pricing_and_purchasing_l850_85037


namespace NUMINAMATH_CALUDE_height_relation_l850_85087

/-- Two right circular cylinders with equal volume and related radii -/
structure CylinderPair where
  r₁ : ℝ  -- radius of the first cylinder
  h₁ : ℝ  -- height of the first cylinder
  r₂ : ℝ  -- radius of the second cylinder
  h₂ : ℝ  -- height of the second cylinder
  r₁_pos : 0 < r₁  -- r₁ is positive
  h₁_pos : 0 < h₁  -- h₁ is positive
  r₂_pos : 0 < r₂  -- r₂ is positive
  h₂_pos : 0 < h₂  -- h₂ is positive
  volume_eq : π * r₁^2 * h₁ = π * r₂^2 * h₂  -- volumes are equal
  radius_relation : r₂ = 1.2 * r₁  -- r₂ is 20% more than r₁

/-- The theorem stating the relationship between the heights of the cylinders -/
theorem height_relation (cp : CylinderPair) : cp.h₁ = 1.44 * cp.h₂ := by
  sorry

#check height_relation

end NUMINAMATH_CALUDE_height_relation_l850_85087


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l850_85015

theorem complex_fraction_evaluation : 
  (((7 - 6.35) / 6.5 + 9.9) * (1 / 12.8)) / 
  ((1.2 / 36 + (1 + 1/5) / 0.25 - (1 + 5/6)) * (1 + 1/4)) / 0.125 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l850_85015


namespace NUMINAMATH_CALUDE_soy_milk_calculation_l850_85099

/-- The amount of soy milk drunk by Mitch's family in a week -/
def soy_milk : ℝ := 0.1

/-- The total amount of milk drunk by Mitch's family in a week -/
def total_milk : ℝ := 0.6

/-- The amount of regular milk drunk by Mitch's family in a week -/
def regular_milk : ℝ := 0.5

/-- Theorem stating that the amount of soy milk is the difference between total milk and regular milk -/
theorem soy_milk_calculation : soy_milk = total_milk - regular_milk := by
  sorry

end NUMINAMATH_CALUDE_soy_milk_calculation_l850_85099


namespace NUMINAMATH_CALUDE_binomial_307_307_equals_1_l850_85069

theorem binomial_307_307_equals_1 : Nat.choose 307 307 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_307_307_equals_1_l850_85069


namespace NUMINAMATH_CALUDE_greatest_among_given_numbers_l850_85029

theorem greatest_among_given_numbers :
  let a := (42 : ℚ) * (7 / 11) / 100
  let b := 17 / 23
  let c := (7391 : ℚ) / 10000
  let d := 29 / 47
  b ≥ a ∧ b ≥ c ∧ b ≥ d := by sorry

end NUMINAMATH_CALUDE_greatest_among_given_numbers_l850_85029


namespace NUMINAMATH_CALUDE_range_of_m_l850_85072

-- Define the sets A and B
def A : Set ℝ := {a | a < -1}
def B (m : ℝ) : Set ℝ := {x | 3*m < x ∧ x < m + 2}

-- Define the proposition P
def P (a : ℝ) : Prop := ∃ x : ℝ, a*x^2 + 2*x - 1 = 0

-- State the theorem
theorem range_of_m :
  (∀ a : ℝ, ¬(P a)) →
  (∀ m : ℝ, ∀ x : ℝ, x ∈ B m → x ∉ A) →
  {m : ℝ | -1/3 ≤ m} = {m : ℝ | ∃ x : ℝ, x ∈ B m} := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l850_85072


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l850_85064

theorem quadratic_equation_solution :
  let x₁ : ℝ := (-1 + Real.sqrt 5) / 2
  let x₂ : ℝ := (-1 - Real.sqrt 5) / 2
  (x₁^2 + x₁ - 1 = 0) ∧ (x₂^2 + x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l850_85064


namespace NUMINAMATH_CALUDE_fraction_greater_than_one_necessary_not_sufficient_l850_85076

theorem fraction_greater_than_one_necessary_not_sufficient :
  (∀ a b : ℝ, a > b ∧ b > 0 → a / b > 1) ∧
  (∃ a b : ℝ, a / b > 1 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_greater_than_one_necessary_not_sufficient_l850_85076


namespace NUMINAMATH_CALUDE_books_sum_is_41_l850_85081

/-- The number of books Keith has -/
def keith_books : ℕ := 20

/-- The number of books Jason has -/
def jason_books : ℕ := 21

/-- The total number of books Keith and Jason have together -/
def total_books : ℕ := keith_books + jason_books

theorem books_sum_is_41 : total_books = 41 := by
  sorry

end NUMINAMATH_CALUDE_books_sum_is_41_l850_85081


namespace NUMINAMATH_CALUDE_vacation_cost_division_l850_85091

theorem vacation_cost_division (total_cost : ℝ) (initial_people : ℕ) (cost_reduction : ℝ) (n : ℕ) :
  total_cost = 1000 →
  initial_people = 4 →
  (total_cost / initial_people) - (total_cost / n) = cost_reduction →
  cost_reduction = 50 →
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l850_85091


namespace NUMINAMATH_CALUDE_factorization_equality_l850_85021

theorem factorization_equality (a b : ℝ) : 
  a^2 - b^2 + 4*a + 2*b + 3 = (a + b + 1)*(a - b + 3) := by
sorry

end NUMINAMATH_CALUDE_factorization_equality_l850_85021


namespace NUMINAMATH_CALUDE_q_function_determination_l850_85053

theorem q_function_determination (q : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c) →  -- q is quadratic
  q 3 = 0 →                                       -- vertical asymptote at x = 3
  q (-3) = 0 →                                    -- vertical asymptote at x = -3
  q 2 = 18 →                                      -- given condition
  ∀ x, q x = -((18 : ℝ) / 5) * x^2 + (162 : ℝ) / 5 :=
by sorry

end NUMINAMATH_CALUDE_q_function_determination_l850_85053


namespace NUMINAMATH_CALUDE_maria_gum_count_l850_85060

/-- The number of gum pieces Maria has after receiving gum from Tommy and Luis -/
def total_gum (initial : ℕ) (from_tommy : ℕ) (from_luis : ℕ) : ℕ :=
  initial + from_tommy + from_luis

/-- Theorem stating that Maria has 61 pieces of gum after receiving gum from Tommy and Luis -/
theorem maria_gum_count : total_gum 25 16 20 = 61 := by
  sorry

end NUMINAMATH_CALUDE_maria_gum_count_l850_85060


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l850_85057

/-- For a geometric sequence with positive real terms, if a_1 = 1 and a_5 = 9, then a_3 = 3 -/
theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_pos : ∀ n, a n > 0) 
  (h_a1 : a 1 = 1) 
  (h_a5 : a 5 = 9) : 
  a 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l850_85057


namespace NUMINAMATH_CALUDE_x_value_from_fraction_equality_l850_85004

theorem x_value_from_fraction_equality (x y : ℝ) :
  x ≠ 1 →
  y^2 + 3*y - 3 ≠ 0 →
  (x / (x - 1) = (y^2 + 3*y - 2) / (y^2 + 3*y - 3)) →
  x = (y^2 + 3*y - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_from_fraction_equality_l850_85004


namespace NUMINAMATH_CALUDE_S_is_valid_set_l850_85036

-- Define the set of numbers greater than √2
def S : Set ℝ := {x : ℝ | x > Real.sqrt 2}

-- Theorem stating that S is a valid set
theorem S_is_valid_set : 
  (∀ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y → x ≠ y) ∧  -- Elements are distinct
  (∀ x y, x ∈ S ∧ y ∈ S → y ∈ S ∧ x ∈ S) ∧  -- Elements are unordered
  (∀ x, x ∈ S ↔ x > Real.sqrt 2)  -- Elements are determined
  := by sorry

end NUMINAMATH_CALUDE_S_is_valid_set_l850_85036


namespace NUMINAMATH_CALUDE_solve_system_l850_85013

theorem solve_system (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) : x = 37 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l850_85013


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l850_85018

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (h1 : f a b c (-1) = 0)
  (h2 : f a b c 0 = -3)
  (h3 : f a b c 2 = -3) :
  (∃ x y : ℝ, 
    (∀ z : ℝ, f a b c z = z^2 - 2*z - 3) ∧
    (x = 1 ∧ y = -4 ∧ ∀ z : ℝ, f a b c z ≥ f a b c x) ∧
    (∀ z : ℝ, z > 1 → ∀ w : ℝ, w > z → f a b c w > f a b c z) ∧
    (∀ z : ℝ, -1 < z ∧ z < 2 → -4 < f a b c z ∧ f a b c z < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l850_85018


namespace NUMINAMATH_CALUDE_division_problem_l850_85039

theorem division_problem : (100 : ℚ) / ((6 : ℚ) / 2) = 100 / 3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l850_85039


namespace NUMINAMATH_CALUDE_triangle_count_l850_85055

/-- Calculates the total number of triangles in a triangular figure composed of n rows of small isosceles triangles. -/
def totalTriangles (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The number of rows in our specific triangular figure -/
def numRows : ℕ := 7

/-- Theorem stating that the total number of triangles in our specific figure is 28 -/
theorem triangle_count : totalTriangles numRows = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l850_85055


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l850_85063

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items in n positions --/
def arrange (n k : ℕ) : ℕ := sorry

theorem photo_arrangement_count :
  let total_students : ℕ := 12
  let initial_front_row : ℕ := 4
  let initial_back_row : ℕ := 8
  let students_to_move : ℕ := 2
  let final_front_row : ℕ := initial_front_row + students_to_move
  choose initial_back_row students_to_move * arrange final_front_row students_to_move =
    choose 8 2 * arrange 6 2 := by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l850_85063


namespace NUMINAMATH_CALUDE_only_B_on_x_axis_l850_85043

def point_A : ℝ × ℝ := (-2, -3)
def point_B : ℝ × ℝ := (-3, 0)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (0, 3)

def is_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

theorem only_B_on_x_axis :
  is_on_x_axis point_B ∧
  ¬is_on_x_axis point_A ∧
  ¬is_on_x_axis point_C ∧
  ¬is_on_x_axis point_D :=
by sorry

end NUMINAMATH_CALUDE_only_B_on_x_axis_l850_85043


namespace NUMINAMATH_CALUDE_range_of_x_l850_85035

theorem range_of_x (x : ℝ) : (1 + 2*x ≤ 8 + 3*x) → x ≥ -7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l850_85035


namespace NUMINAMATH_CALUDE_final_display_l850_85090

def special_key (x : ℚ) : ℚ := 1 / (2 - x)

def iterate_key (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | m + 1 => special_key (iterate_key m x)

theorem final_display : iterate_key 50 3 = 49 / 51 := by
  sorry

end NUMINAMATH_CALUDE_final_display_l850_85090


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l850_85059

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 3 ∧ b > 3 ∧
    n = 1 * a + 4 ∧
    n = 2 * b + 3

theorem smallest_dual_base_representation : 
  is_valid_representation 11 ∧ 
  ∀ m : ℕ, m < 11 → ¬(is_valid_representation m) :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l850_85059


namespace NUMINAMATH_CALUDE_alvin_marbles_won_l850_85070

/-- Calculates the number of marbles won in the second game -/
def marbles_won_second_game (initial : ℕ) (lost_first : ℕ) (final : ℕ) : ℕ :=
  final - (initial - lost_first)

/-- Proves that Alvin won 25 marbles in the second game -/
theorem alvin_marbles_won : marbles_won_second_game 57 18 64 = 25 := by
  sorry

end NUMINAMATH_CALUDE_alvin_marbles_won_l850_85070


namespace NUMINAMATH_CALUDE_race_track_cost_l850_85049

theorem race_track_cost (initial_amount : ℚ) (num_cars : ℕ) (car_cost : ℚ) (remaining : ℚ) : 
  initial_amount = 17.80 ∧ 
  num_cars = 4 ∧ 
  car_cost = 0.95 ∧ 
  remaining = 8 → 
  initial_amount - (↑num_cars * car_cost) - remaining = 6 := by
sorry

end NUMINAMATH_CALUDE_race_track_cost_l850_85049


namespace NUMINAMATH_CALUDE_tangent_line_to_curve_l850_85084

/-- A line y = x - 2a is tangent to the curve y = x ln x - x if and only if a = e/2 -/
theorem tangent_line_to_curve (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (x₀ - 2*a = x₀ * Real.log x₀ - x₀) ∧ 
    (1 = Real.log x₀)) ↔ 
  a = Real.exp 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_curve_l850_85084


namespace NUMINAMATH_CALUDE_chair_arrangement_l850_85007

theorem chair_arrangement (total_chairs : ℕ) (h : total_chairs = 10000) :
  ∃ (n : ℕ), n * n = total_chairs :=
sorry

end NUMINAMATH_CALUDE_chair_arrangement_l850_85007


namespace NUMINAMATH_CALUDE_inequality_no_solution_l850_85022

theorem inequality_no_solution : {x : ℝ | x * (2 - x) > 3} = ∅ := by sorry

end NUMINAMATH_CALUDE_inequality_no_solution_l850_85022
