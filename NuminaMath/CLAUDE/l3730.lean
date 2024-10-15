import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l3730_373050

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * x + 1 / x^2 ≥ 4 ∧ 
  (3 * x + 1 / x^2 = 4 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3730_373050


namespace NUMINAMATH_CALUDE_trip_distance_proof_l3730_373061

/-- Represents the total distance of the trip in miles -/
def total_distance : ℝ := 90

/-- Represents the distance traveled on battery power in miles -/
def battery_distance : ℝ := 30

/-- Represents the gasoline consumption rate after battery power in gallons per mile -/
def gasoline_rate : ℝ := 0.03

/-- Represents the overall average fuel efficiency in miles per gallon -/
def average_efficiency : ℝ := 50

/-- Proves that the total trip distance is correct given the conditions -/
theorem trip_distance_proof :
  (total_distance / (gasoline_rate * (total_distance - battery_distance)) = average_efficiency) ∧
  (total_distance > battery_distance) :=
by sorry

end NUMINAMATH_CALUDE_trip_distance_proof_l3730_373061


namespace NUMINAMATH_CALUDE_number_decrease_divide_l3730_373064

theorem number_decrease_divide (x : ℚ) : (x - 4) / 10 = 5 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_decrease_divide_l3730_373064


namespace NUMINAMATH_CALUDE_lizard_spot_wrinkle_ratio_l3730_373037

/-- Represents a three-eyed lizard with wrinkles and spots. -/
structure Lizard where
  eyes : ℕ
  wrinkles : ℕ
  spots : ℕ

/-- The properties of our specific lizard. -/
def specialLizard : Lizard where
  eyes := 3
  wrinkles := 3 * 3
  spots := 3 + (3 * 3) - 69 + 3

theorem lizard_spot_wrinkle_ratio (l : Lizard) 
  (h1 : l.eyes = 3)
  (h2 : l.wrinkles = 3 * l.eyes)
  (h3 : l.eyes = l.spots + l.wrinkles - 69) :
  l.spots / l.wrinkles = 7 := by
  sorry

#eval specialLizard.spots / specialLizard.wrinkles

end NUMINAMATH_CALUDE_lizard_spot_wrinkle_ratio_l3730_373037


namespace NUMINAMATH_CALUDE_trash_cans_veterans_park_l3730_373099

/-- The number of trash cans in Veteran's Park after the transfer -/
def final_trash_cans_veterans_park (initial_veterans_park : ℕ) (initial_central_park : ℕ) : ℕ :=
  initial_veterans_park + initial_central_park / 2

/-- Theorem stating the final number of trash cans in Veteran's Park -/
theorem trash_cans_veterans_park :
  ∃ (initial_central_park : ℕ),
    (initial_central_park = 24 / 2 + 8) ∧
    (final_trash_cans_veterans_park 24 initial_central_park = 34) := by
  sorry

#check trash_cans_veterans_park

end NUMINAMATH_CALUDE_trash_cans_veterans_park_l3730_373099


namespace NUMINAMATH_CALUDE_largest_angle_is_E_l3730_373094

/-- Represents a hexagon with specific angle properties -/
structure Hexagon where
  /-- Angle A is 100 degrees -/
  angle_A : ℝ
  angle_A_eq : angle_A = 100

  /-- Angle B is 120 degrees -/
  angle_B : ℝ
  angle_B_eq : angle_B = 120

  /-- Angles C and D are equal -/
  angle_C : ℝ
  angle_D : ℝ
  angle_C_eq_D : angle_C = angle_D

  /-- Angle E is 30 degrees more than the average of angles C, D, and F -/
  angle_E : ℝ
  angle_F : ℝ
  angle_E_eq : angle_E = (angle_C + angle_D + angle_F) / 3 + 30

  /-- The sum of all angles in a hexagon is 720 degrees -/
  sum_of_angles : angle_A + angle_B + angle_C + angle_D + angle_E + angle_F = 720

/-- Theorem: The largest angle in the hexagon is 147.5 degrees -/
theorem largest_angle_is_E (h : Hexagon) : h.angle_E = 147.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_is_E_l3730_373094


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3730_373078

-- Define the function
def f (x : ℝ) : ℝ := 2*x^3 - 6*x^2 - 18*x + 7

-- Define the derivative of the function
def f_derivative (x : ℝ) : ℝ := 6*x^2 - 12*x - 18

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (x > -1 ∧ x < 3) ↔ (f_derivative x < 0) :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3730_373078


namespace NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_17_l3730_373075

theorem three_digit_numbers_divisible_by_17 : 
  (Finset.filter (fun k => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_17_l3730_373075


namespace NUMINAMATH_CALUDE_perpendicular_lines_line_slope_l3730_373038

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
theorem perpendicular_lines (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) (hB₁ : B₁ ≠ 0) (hB₂ : B₂ ≠ 0) :
  (A₁ * x + B₁ * y + C₁ = 0 ∧ A₂ * x + B₂ * y + C₂ = 0) →
  ((-A₁ / B₁) * (-A₂ / B₂) = -1 ↔ (A₁ * A₂ + B₁ * B₂ = 0)) :=
by sorry

/-- The slope of a line Ax + By + C = 0 is -A/B -/
theorem line_slope (A B C : ℝ) (hB : B ≠ 0) :
  (A * x + B * y + C = 0) → (y = (-A / B) * x - C / B) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_line_slope_l3730_373038


namespace NUMINAMATH_CALUDE_slopes_negative_reciprocals_min_area_ANB_l3730_373079

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define points M and N
def M : ℝ × ℝ := (1, 0)
def N : ℝ × ℝ := (-1, 0)

-- Define a line passing through M
def line_through_M (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define points A and B as intersections of the line and parabola
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry

-- Define slopes of NA and NB
def slope_NA (k : ℝ) : ℝ := sorry
def slope_NB (k : ℝ) : ℝ := sorry

-- Define area of triangle ANB
def area_ANB (k : ℝ) : ℝ := sorry

theorem slopes_negative_reciprocals :
  ∀ k : ℝ, k ≠ 0 → slope_NA k * slope_NB k = -1 :=
sorry

theorem min_area_ANB :
  ∃ min_area : ℝ, min_area = 4 ∧ ∀ k : ℝ, k ≠ 0 → area_ANB k ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_slopes_negative_reciprocals_min_area_ANB_l3730_373079


namespace NUMINAMATH_CALUDE_linda_furniture_fraction_l3730_373023

def original_savings : ℚ := 1200
def tv_cost : ℚ := 300

def furniture_cost : ℚ := original_savings - tv_cost

theorem linda_furniture_fraction :
  furniture_cost / original_savings = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_linda_furniture_fraction_l3730_373023


namespace NUMINAMATH_CALUDE_correct_probability_l3730_373063

/-- The number of options for the first three digits -/
def first_three_options : ℕ := 3

/-- The number of permutations of the last four digits (0, 1, 6, 6) -/
def last_four_permutations : ℕ := 12

/-- The total number of possible phone numbers -/
def total_possible_numbers : ℕ := first_three_options * last_four_permutations

/-- The probability of dialing the correct number -/
def probability_correct : ℚ := 1 / total_possible_numbers

theorem correct_probability : probability_correct = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_probability_l3730_373063


namespace NUMINAMATH_CALUDE_alpha_squared_gt_beta_squared_l3730_373066

theorem alpha_squared_gt_beta_squared 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (-π/2) (π/2)) 
  (h2 : β ∈ Set.Icc (-π/2) (π/2)) 
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 :=
sorry

end NUMINAMATH_CALUDE_alpha_squared_gt_beta_squared_l3730_373066


namespace NUMINAMATH_CALUDE_apples_left_l3730_373086

theorem apples_left (initial_apples used_apples : ℕ) 
  (h1 : initial_apples = 43)
  (h2 : used_apples = 41) :
  initial_apples - used_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l3730_373086


namespace NUMINAMATH_CALUDE_triangle_and_squares_area_l3730_373001

theorem triangle_and_squares_area (x : ℝ) : 
  let triangle_area := (1/2) * (3*x) * (4*x)
  let square1_area := (3*x)^2
  let square2_area := (4*x)^2
  let square3_area := (6*x)^2
  let total_area := triangle_area + square1_area + square2_area + square3_area
  total_area = 1288 → x = Real.sqrt (1288/67) := by
sorry

end NUMINAMATH_CALUDE_triangle_and_squares_area_l3730_373001


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_gasoline_tank_capacity_proof_l3730_373012

theorem gasoline_tank_capacity : ℝ → Prop :=
  fun capacity =>
    let initial_fraction : ℝ := 5/6
    let final_fraction : ℝ := 1/3
    let used_amount : ℝ := 15
    initial_fraction * capacity - final_fraction * capacity = used_amount →
    capacity = 30

-- The proof goes here
theorem gasoline_tank_capacity_proof : gasoline_tank_capacity 30 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_gasoline_tank_capacity_proof_l3730_373012


namespace NUMINAMATH_CALUDE_accidental_vs_correct_calculation_l3730_373069

theorem accidental_vs_correct_calculation (x : ℚ) : 
  7 * ((x + 24) / 5) = 70 → (5 * x + 24) / 7 = 22 := by
  sorry

end NUMINAMATH_CALUDE_accidental_vs_correct_calculation_l3730_373069


namespace NUMINAMATH_CALUDE_group_average_age_l3730_373015

theorem group_average_age 
  (num_women : ℕ) 
  (num_men : ℕ) 
  (avg_age_women : ℚ) 
  (avg_age_men : ℚ) 
  (h1 : num_women = 12) 
  (h2 : num_men = 18) 
  (h3 : avg_age_women = 28) 
  (h4 : avg_age_men = 40) : 
  (num_women * avg_age_women + num_men * avg_age_men) / (num_women + num_men : ℚ) = 352 / 10 := by
  sorry

end NUMINAMATH_CALUDE_group_average_age_l3730_373015


namespace NUMINAMATH_CALUDE_total_jump_rope_time_l3730_373081

/-- The total jump rope time for four girls given their relative jump times -/
theorem total_jump_rope_time (cindy betsy tina sarah : ℕ) : 
  cindy = 12 →
  betsy = cindy / 2 →
  tina = betsy * 3 →
  sarah = cindy + tina →
  cindy + betsy + tina + sarah = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_jump_rope_time_l3730_373081


namespace NUMINAMATH_CALUDE_accidents_in_four_minutes_l3730_373040

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The duration of the observation period in minutes -/
def observation_period : ℕ := 4

/-- The interval between car collisions in seconds -/
def car_collision_interval : ℕ := 10

/-- The interval between big crashes in seconds -/
def big_crash_interval : ℕ := 20

/-- The total number of accidents in the observation period -/
def total_accidents : ℕ := 36

theorem accidents_in_four_minutes :
  (observation_period * seconds_per_minute) / car_collision_interval +
  (observation_period * seconds_per_minute) / big_crash_interval =
  total_accidents := by
  sorry

end NUMINAMATH_CALUDE_accidents_in_four_minutes_l3730_373040


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l3730_373055

def numbers : List ℝ := [1030, 1560, 1980, 2025, 2140, 2250, 2450, 2600, 2780, 2910]

theorem mean_of_remaining_numbers :
  let total_sum := numbers.sum
  let seven_mean := 2300
  let seven_sum := 7 * seven_mean
  let remaining_sum := total_sum - seven_sum
  (remaining_sum / 3 : ℝ) = 2108.33 := by
sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l3730_373055


namespace NUMINAMATH_CALUDE_charles_initial_bananas_l3730_373035

/-- The initial number of bananas Willie has -/
def willie_initial : ℕ := 48

/-- The final number of bananas Willie has -/
def willie_final : ℕ := 13

/-- The number of bananas Charles loses -/
def charles_loss : ℕ := 35

/-- The initial number of bananas Charles has -/
def charles_initial : ℕ := charles_loss

theorem charles_initial_bananas :
  charles_initial = 35 :=
sorry

end NUMINAMATH_CALUDE_charles_initial_bananas_l3730_373035


namespace NUMINAMATH_CALUDE_anna_initial_ham_slices_l3730_373031

/-- The number of slices of ham Anna puts in each sandwich. -/
def slices_per_sandwich : ℕ := 3

/-- The number of sandwiches Anna wants to make. -/
def total_sandwiches : ℕ := 50

/-- The additional number of ham slices Anna needs. -/
def additional_slices : ℕ := 119

/-- The initial number of ham slices Anna has. -/
def initial_slices : ℕ := total_sandwiches * slices_per_sandwich - additional_slices

theorem anna_initial_ham_slices :
  initial_slices = 31 := by sorry

end NUMINAMATH_CALUDE_anna_initial_ham_slices_l3730_373031


namespace NUMINAMATH_CALUDE_karen_has_32_quarters_l3730_373028

/-- Calculates the number of quarters Karen has given the conditions of the problem -/
def karens_quarters (christopher_quarters : ℕ) (dollar_difference : ℕ) : ℕ :=
  let christopher_value := christopher_quarters * 25  -- Value in cents
  let karen_value := christopher_value - dollar_difference * 100  -- Value in cents
  karen_value / 25  -- Convert back to quarters

/-- Proves that Karen has 32 quarters given the problem conditions -/
theorem karen_has_32_quarters :
  karens_quarters 64 8 = 32 := by sorry

end NUMINAMATH_CALUDE_karen_has_32_quarters_l3730_373028


namespace NUMINAMATH_CALUDE_simplify_expression_l3730_373003

theorem simplify_expression : 
  Real.sqrt 15 + Real.sqrt 45 - (Real.sqrt (4/3) - Real.sqrt 108) = 
  Real.sqrt 15 + 3 * Real.sqrt 5 + (16 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3730_373003


namespace NUMINAMATH_CALUDE_base_k_conversion_l3730_373059

theorem base_k_conversion (k : ℕ) : 
  (0 < k ∧ k < 10) → (k^2 + 7*k + 5 = 125) → k = 8 :=
by sorry

end NUMINAMATH_CALUDE_base_k_conversion_l3730_373059


namespace NUMINAMATH_CALUDE_product_xy_l3730_373087

theorem product_xy (x y : ℚ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x * y = -1/5 := by
sorry

end NUMINAMATH_CALUDE_product_xy_l3730_373087


namespace NUMINAMATH_CALUDE_problem_solving_probability_l3730_373041

theorem problem_solving_probability :
  let p_A : ℚ := 1/2
  let p_B : ℚ := 1/3
  let p_C : ℚ := 1/4
  let p_at_least_one : ℚ := 1 - (1 - p_A) * (1 - p_B) * (1 - p_C)
  p_at_least_one = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l3730_373041


namespace NUMINAMATH_CALUDE_younger_person_age_l3730_373091

/-- 
Given two persons whose ages differ by 20 years, and 10 years ago the elder was 5 times as old as the younger,
prove that the present age of the younger person is 15 years.
-/
theorem younger_person_age (y e : ℕ) : 
  e = y + 20 → 
  e - 10 = 5 * (y - 10) → 
  y = 15 := by
  sorry

end NUMINAMATH_CALUDE_younger_person_age_l3730_373091


namespace NUMINAMATH_CALUDE_expression_value_l3730_373004

theorem expression_value (x y : ℝ) (h : |x + 1| + (y - 2)^2 = 0) :
  4 * x^2 * y - (6 * x * y - 3 * (4 * x * y - 2) - x^2 * y) + 1 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3730_373004


namespace NUMINAMATH_CALUDE_shoe_multiple_l3730_373074

/-- Given the following conditions:
  - Bonny has 13 pairs of shoes
  - Bonny's shoes are 5 less than a certain multiple of Becky's shoes
  - Bobby has 3 times as many shoes as Becky
  - Bobby has 27 pairs of shoes
  Prove that the multiple of Becky's shoes that is 5 more than Bonny's shoes is 2. -/
theorem shoe_multiple (bonny_shoes : ℕ) (bobby_shoes : ℕ) (becky_shoes : ℕ) (m : ℕ) :
  bonny_shoes = 13 →
  ∃ m, bonny_shoes + 5 = m * becky_shoes →
  bobby_shoes = 3 * becky_shoes →
  bobby_shoes = 27 →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_shoe_multiple_l3730_373074


namespace NUMINAMATH_CALUDE_equation_solutions_l3730_373036

def is_solution (a b c : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (1 : ℚ) / a + 1 / b + 1 / c = 1

def solution_set : Set (ℤ × ℤ × ℤ) :=
  {(3, 3, 3), (2, 3, 6), (2, 4, 4)} ∪ {(1, t, -t) | t : ℤ}

theorem equation_solutions :
  ∀ (a b c : ℤ), is_solution a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3730_373036


namespace NUMINAMATH_CALUDE_scarf_wool_calculation_l3730_373034

/-- The number of balls of wool used for a scarf -/
def scarf_wool : ℕ := sorry

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for a sweater -/
def sweater_wool : ℕ := 4

/-- The total number of balls of wool used -/
def total_wool : ℕ := 82

theorem scarf_wool_calculation :
  scarf_wool * aaron_scarves + 
  sweater_wool * (aaron_sweaters + enid_sweaters) = 
  total_wool ∧ scarf_wool = 3 := by sorry

end NUMINAMATH_CALUDE_scarf_wool_calculation_l3730_373034


namespace NUMINAMATH_CALUDE_not_perfect_square_l3730_373056

theorem not_perfect_square (n : ℤ) (h : n > 4) : ¬∃ (k : ℕ), n^2 - 3*n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3730_373056


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3730_373057

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3730_373057


namespace NUMINAMATH_CALUDE_inequalities_hold_l3730_373011

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((a + b) * (1 / a + 1 / b) ≥ 4) ∧
  (a^2 + b^2 + 2 ≥ 2*a + 2*b) ∧
  (Real.sqrt (abs (a - b)) ≥ Real.sqrt a - Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_hold_l3730_373011


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l3730_373009

theorem largest_integer_inequality : ∀ x : ℤ, x ≤ 4 ↔ (x : ℚ) / 4 - 3 / 7 < 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l3730_373009


namespace NUMINAMATH_CALUDE_runners_meet_time_l3730_373017

/-- Two runners on a circular track meet after approximately 15 seconds --/
theorem runners_meet_time (track_length : ℝ) (speed1 speed2 : ℝ) : 
  track_length = 250 →
  speed1 = 20 * (1000 / 3600) →
  speed2 = 40 * (1000 / 3600) →
  abs (15 - track_length / (speed1 + speed2)) < 0.1 := by
  sorry

#check runners_meet_time

end NUMINAMATH_CALUDE_runners_meet_time_l3730_373017


namespace NUMINAMATH_CALUDE_ball_radius_l3730_373084

theorem ball_radius (hole_diameter : ℝ) (hole_depth : ℝ) (ball_radius : ℝ) : 
  hole_diameter = 30 ∧ hole_depth = 10 → ball_radius = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_l3730_373084


namespace NUMINAMATH_CALUDE_unique_digit_for_divisibility_by_nine_l3730_373010

def sum_of_digits (n : ℕ) : ℕ := 8 + 6 + 5 + n + 7 + 4 + 3 + 2

theorem unique_digit_for_divisibility_by_nine :
  ∃! n : ℕ, n ≤ 9 ∧ (sum_of_digits n) % 9 = 0 ∧ n = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_for_divisibility_by_nine_l3730_373010


namespace NUMINAMATH_CALUDE_apprentice_work_time_l3730_373007

/-- Proves that given the master's and apprentice's production rates, 
    the apprentice needs 4 hours to match the master's 3-hour output. -/
theorem apprentice_work_time 
  (master_rate : ℕ) 
  (apprentice_rate : ℕ) 
  (master_time : ℕ) 
  (h1 : master_rate = 64)
  (h2 : apprentice_rate = 48)
  (h3 : master_time = 3) :
  (master_rate * master_time) / apprentice_rate = 4 := by
  sorry

#check apprentice_work_time

end NUMINAMATH_CALUDE_apprentice_work_time_l3730_373007


namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l3730_373049

theorem rectangle_cylinder_volume_ratio :
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 9
  let cylinder_a_radius : ℝ := rectangle_width / (2 * Real.pi)
  let cylinder_a_height : ℝ := rectangle_height
  let cylinder_b_radius : ℝ := rectangle_height / (2 * Real.pi)
  let cylinder_b_height : ℝ := rectangle_width
  let volume_a : ℝ := Real.pi * cylinder_a_radius^2 * cylinder_a_height
  let volume_b : ℝ := Real.pi * cylinder_b_radius^2 * cylinder_b_height
  volume_b / volume_a = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l3730_373049


namespace NUMINAMATH_CALUDE_tank_fill_time_l3730_373097

/-- Represents the state of the tank and pipes -/
structure TankSystem where
  pipeA : ℝ  -- Rate at which Pipe A fills the tank (fraction of tank per minute)
  pipeB : ℝ  -- Rate at which Pipe B empties the tank (fraction of tank per minute)
  closeBTime : ℝ  -- Time at which Pipe B is closed (in minutes)

/-- Calculates the time taken to fill the tank given the tank system parameters -/
def timeTakenToFill (system : TankSystem) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the tank will be filled in 70 minutes -/
theorem tank_fill_time (system : TankSystem) 
  (hA : system.pipeA = 1 / 8)
  (hB : system.pipeB = 1 / 24)
  (hClose : system.closeBTime = 66) :
  timeTakenToFill system = 70 :=
sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3730_373097


namespace NUMINAMATH_CALUDE_binomial_coefficient_8_4_l3730_373076

theorem binomial_coefficient_8_4 : Nat.choose 8 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_8_4_l3730_373076


namespace NUMINAMATH_CALUDE_data_tape_cost_calculation_l3730_373019

/-- The cost of mounting a data tape for a computer program run. -/
def data_tape_cost : ℝ := 5.35

/-- The operating-system overhead cost per run. -/
def os_overhead_cost : ℝ := 1.07

/-- The cost of computer time per millisecond. -/
def computer_time_cost_per_ms : ℝ := 0.023

/-- The total cost for one run of the program. -/
def total_cost : ℝ := 40.92

/-- The duration of the computer program run in seconds. -/
def run_duration_seconds : ℝ := 1.5

theorem data_tape_cost_calculation :
  data_tape_cost = total_cost - (os_overhead_cost + computer_time_cost_per_ms * (run_duration_seconds * 1000)) :=
by sorry

end NUMINAMATH_CALUDE_data_tape_cost_calculation_l3730_373019


namespace NUMINAMATH_CALUDE_natural_raisin_cost_l3730_373068

/-- The cost per scoop of golden seedless raisins in dollars -/
def golden_cost : ℚ := 255/100

/-- The number of scoops of golden seedless raisins -/
def golden_scoops : ℕ := 20

/-- The number of scoops of natural seedless raisins -/
def natural_scoops : ℕ := 20

/-- The cost per scoop of the mixture in dollars -/
def mixture_cost : ℚ := 3

/-- The cost per scoop of natural seedless raisins in dollars -/
def natural_cost : ℚ := 345/100

theorem natural_raisin_cost : 
  (golden_cost * golden_scoops + natural_cost * natural_scoops) / (golden_scoops + natural_scoops) = mixture_cost :=
sorry

end NUMINAMATH_CALUDE_natural_raisin_cost_l3730_373068


namespace NUMINAMATH_CALUDE_oriented_knight_moves_l3730_373016

/-- An Oriented Knight's move on a chess board -/
inductive OrientedKnightMove
| right_up : OrientedKnightMove  -- Two squares right, one square up
| up_right : OrientedKnightMove  -- Two squares up, one square right

/-- A sequence of Oriented Knight moves -/
def MoveSequence := List OrientedKnightMove

/-- The size of the chess board -/
def boardSize : ℕ := 16

/-- Checks if a sequence of moves is valid (reaches the top-right corner) -/
def isValidSequence (moves : MoveSequence) : Prop :=
  let finalPosition := moves.foldl
    (fun pos move => match move with
      | OrientedKnightMove.right_up => (pos.1 + 2, pos.2 + 1)
      | OrientedKnightMove.up_right => (pos.1 + 1, pos.2 + 2))
    (0, 0)
  finalPosition = (boardSize - 1, boardSize - 1)

/-- The number of valid move sequences for an Oriented Knight -/
def validSequenceCount : ℕ := 252

theorem oriented_knight_moves :
  (validSequences : Finset MoveSequence).card = validSequenceCount :=
by
  sorry

end NUMINAMATH_CALUDE_oriented_knight_moves_l3730_373016


namespace NUMINAMATH_CALUDE_hotdog_count_l3730_373027

theorem hotdog_count (initial : ℕ) (sold : ℕ) (remaining : ℕ) : 
  sold = 2 → remaining = 97 → initial = remaining + sold :=
by sorry

end NUMINAMATH_CALUDE_hotdog_count_l3730_373027


namespace NUMINAMATH_CALUDE_rectangular_prism_space_diagonal_l3730_373000

/-- A rectangular prism with given surface area and edge length sum has a space diagonal of length 5 -/
theorem rectangular_prism_space_diagonal : 
  ∀ (x y z : ℝ), 
  (2 * x * y + 2 * y * z + 2 * x * z = 11) →
  (4 * (x + y + z) = 24) →
  Real.sqrt (x^2 + y^2 + z^2) = 5 := by
sorry


end NUMINAMATH_CALUDE_rectangular_prism_space_diagonal_l3730_373000


namespace NUMINAMATH_CALUDE_range_of_f_l3730_373080

/-- A monotonically increasing odd function f with f(1) = 2 and f(2) = 3 -/
def f : ℝ → ℝ :=
  sorry

/-- f is monotonically increasing -/
axiom f_increasing (x y : ℝ) : x < y → f x < f y

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f(1) = 2 -/
axiom f_1 : f 1 = 2

/-- f(2) = 3 -/
axiom f_2 : f 2 = 3

/-- The main theorem -/
theorem range_of_f (x : ℝ) : 
  (-3 < f (x - 3) ∧ f (x - 3) < 2) ↔ (1 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3730_373080


namespace NUMINAMATH_CALUDE_five_thursdays_in_august_l3730_373092

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- A month with its dates -/
structure Month :=
  (dates : List Date)
  (numDays : Nat)

def july : Month := sorry
def august : Month := sorry

/-- Counts the number of occurrences of a specific day in a month -/
def countDaysInMonth (m : Month) (d : DayOfWeek) : Nat := sorry

theorem five_thursdays_in_august 
  (h1 : july.numDays = 31)
  (h2 : august.numDays = 31)
  (h3 : countDaysInMonth july DayOfWeek.Tuesday = 5) :
  countDaysInMonth august DayOfWeek.Thursday = 5 := by sorry

end NUMINAMATH_CALUDE_five_thursdays_in_august_l3730_373092


namespace NUMINAMATH_CALUDE_power_of_power_l3730_373071

theorem power_of_power (a : ℝ) : (a^5)^2 = a^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3730_373071


namespace NUMINAMATH_CALUDE_tile_arrangement_count_l3730_373045

/-- The number of distinguishable arrangements of tiles -/
def tile_arrangements (brown green yellow : ℕ) (purple : ℕ) : ℕ :=
  Nat.factorial (brown + green + yellow + purple) /
  (Nat.factorial brown * Nat.factorial green * Nat.factorial yellow * Nat.factorial purple)

/-- Theorem stating that the number of distinguishable arrangements
    of 2 brown, 3 green, 2 yellow, and 1 purple tile is 1680 -/
theorem tile_arrangement_count :
  tile_arrangements 2 3 2 1 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangement_count_l3730_373045


namespace NUMINAMATH_CALUDE_train_length_l3730_373022

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 54 →
  crossing_time = 58.9952803775698 →
  bridge_length = 720 →
  ∃ (train_length : ℝ), abs (train_length - 164.93) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3730_373022


namespace NUMINAMATH_CALUDE_quadrilateral_equation_implies_rhombus_l3730_373048

-- Define a quadrilateral
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_d : d > 0

-- Define the condition from the problem
def satisfiesEquation (q : Quadrilateral) : Prop :=
  q.a^4 + q.b^4 + q.c^4 + q.d^4 = 4 * q.a * q.b * q.c * q.d

-- Define a rhombus
def isRhombus (q : Quadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

-- Theorem statement
theorem quadrilateral_equation_implies_rhombus (q : Quadrilateral) :
  satisfiesEquation q → isRhombus q :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_equation_implies_rhombus_l3730_373048


namespace NUMINAMATH_CALUDE_triangle_6_8_10_is_right_l3730_373042

-- Define a triangle with sides a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a triangle is right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

-- Theorem: A triangle with sides 6, 8, and 10 is a right triangle
theorem triangle_6_8_10_is_right : 
  let t : Triangle := { a := 6, b := 8, c := 10 }
  isRightTriangle t := by
  sorry


end NUMINAMATH_CALUDE_triangle_6_8_10_is_right_l3730_373042


namespace NUMINAMATH_CALUDE_tan_sum_alpha_beta_l3730_373013

theorem tan_sum_alpha_beta (α β : Real) (h : 2 * Real.tan α = 3 * Real.tan β) :
  Real.tan (α + β) = (5 * Real.sin (2 * β)) / (5 * Real.cos (2 * β) - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_alpha_beta_l3730_373013


namespace NUMINAMATH_CALUDE_three_face_painted_subcubes_count_l3730_373006

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  painted_faces : ℕ := 6

/-- Counts the number of subcubes with at least three painted faces -/
def count_three_face_painted_subcubes (c : PaintedCube 4) : ℕ :=
  8

/-- Theorem: In a 4x4x4 painted cube, there are exactly 8 subcubes with at least three painted faces -/
theorem three_face_painted_subcubes_count (c : PaintedCube 4) :
  count_three_face_painted_subcubes c = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_face_painted_subcubes_count_l3730_373006


namespace NUMINAMATH_CALUDE_sqrt_three_plus_two_power_l3730_373072

theorem sqrt_three_plus_two_power : (Real.sqrt 3 + Real.sqrt 2) ^ 2023 * (Real.sqrt 3 - Real.sqrt 2) ^ 2022 = Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_plus_two_power_l3730_373072


namespace NUMINAMATH_CALUDE_train_length_l3730_373033

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 225 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3730_373033


namespace NUMINAMATH_CALUDE_first_player_can_force_odd_result_l3730_373047

/-- A game where two players insert operations between numbers 1 to 100 --/
def NumberGame : Type := List (Fin 100 → ℕ) → Prop

/-- The set of possible operations in the game --/
inductive Operation
| Add
| Subtract
| Multiply

/-- A strategy for a player in the game --/
def Strategy : Type := List Operation → Operation

/-- The result of applying operations to a list of numbers --/
def applyOperations (nums : List ℕ) (ops : List Operation) : ℕ := sorry

/-- A winning strategy ensures an odd result --/
def winningStrategy (s : Strategy) : Prop :=
  ∀ (opponent : Strategy), 
    ∃ (finalOps : List Operation), 
      Odd (applyOperations (List.range 100) finalOps)

/-- Theorem: There exists a winning strategy for the first player --/
theorem first_player_can_force_odd_result :
  ∃ (s : Strategy), winningStrategy s :=
sorry

end NUMINAMATH_CALUDE_first_player_can_force_odd_result_l3730_373047


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3730_373088

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 3}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3730_373088


namespace NUMINAMATH_CALUDE_cake_sector_angle_l3730_373098

theorem cake_sector_angle (total_sectors : ℕ) (probability : ℚ) : 
  total_sectors = 6 → probability = 1/8 → 
  ∃ (angle : ℚ), angle = 45 ∧ probability = angle / 360 := by
  sorry

end NUMINAMATH_CALUDE_cake_sector_angle_l3730_373098


namespace NUMINAMATH_CALUDE_square_divisibility_l3730_373005

theorem square_divisibility (n : ℕ+) (h : ∀ q : ℕ+, q ∣ n → q ≤ 12) :
  144 ∣ n^2 := by
sorry

end NUMINAMATH_CALUDE_square_divisibility_l3730_373005


namespace NUMINAMATH_CALUDE_polynomial_divisibility_implies_specific_coefficients_l3730_373083

theorem polynomial_divisibility_implies_specific_coefficients :
  ∀ (p q : ℝ),
  (∀ x : ℝ, (x + 3) * (x - 2) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x + 9)) →
  p = -19.5 ∧ q = -55.5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_implies_specific_coefficients_l3730_373083


namespace NUMINAMATH_CALUDE_square_area_l3730_373044

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + 5*x + 6

/-- The line function -/
def g (x : ℝ) : ℝ := 10

theorem square_area : ∃ (x₁ x₂ : ℝ), 
  f x₁ = g x₁ ∧ 
  f x₂ = g x₂ ∧ 
  x₁ ≠ x₂ ∧ 
  (x₂ - x₁)^2 = 41 := by
sorry

end NUMINAMATH_CALUDE_square_area_l3730_373044


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l3730_373070

def ends_with_6 (n : ℕ) : Prop := n % 10 = 6

def move_6_to_front (n : ℕ) : ℕ :=
  let d := (Nat.log 10 n) + 1
  6 * 10^d + n / 10

theorem smallest_number_with_properties : ℕ := by
  let n := 1538466
  have h1 : ends_with_6 n := by sorry
  have h2 : move_6_to_front n = 4 * n := by sorry
  have h3 : ∀ m < n, ¬(ends_with_6 m ∧ move_6_to_front m = 4 * m) := by sorry
  exact n

end NUMINAMATH_CALUDE_smallest_number_with_properties_l3730_373070


namespace NUMINAMATH_CALUDE_sqrt_3_minus_2_squared_l3730_373062

theorem sqrt_3_minus_2_squared : (Real.sqrt 3 - 2) * (Real.sqrt 3 - 2) = 7 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_2_squared_l3730_373062


namespace NUMINAMATH_CALUDE_six_distinct_one_repeat_probability_l3730_373067

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 7

/-- The probability of rolling seven standard six-sided dice and getting exactly six distinct numbers, with one number repeating once -/
theorem six_distinct_one_repeat_probability : 
  (num_sides.choose 1 * (num_sides - 1).factorial * num_dice.choose 2) / num_sides ^ num_dice = 5 / 186 := by
  sorry

end NUMINAMATH_CALUDE_six_distinct_one_repeat_probability_l3730_373067


namespace NUMINAMATH_CALUDE_missing_number_proof_l3730_373058

theorem missing_number_proof (x : ℤ) : (4 + 3) + (x - 3 - 1) = 11 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3730_373058


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3730_373020

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos x))) = x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3730_373020


namespace NUMINAMATH_CALUDE_complex_quadrant_l3730_373002

theorem complex_quadrant (z : ℂ) (h : (z + Complex.I) * Complex.I = 1 + z) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3730_373002


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l3730_373060

theorem triangle_angle_sum (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Ensures angles are positive
  A + B + C = 180 →        -- Sum of angles in a triangle is 180°
  A = 90 →                 -- Given: Angle A is 90°
  B = 50 →                 -- Given: Angle B is 50°
  C = 40 :=                -- To prove: Angle C is 40°
by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l3730_373060


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l3730_373093

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Represents the height of an object in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Converts a Height to total inches -/
def heightToInches (h : Height) : ℕ := feetToInches h.feet + h.inches

/-- Calculates the total height when placing an object on a base -/
def totalHeight (objectHeight : Height) (baseHeight : ℕ) : ℕ :=
  heightToInches objectHeight + baseHeight

theorem sculpture_and_base_height :
  let sculptureHeight : Height := { feet := 2, inches := 10 }
  let baseHeight : ℕ := 4
  totalHeight sculptureHeight baseHeight = 38 := by sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l3730_373093


namespace NUMINAMATH_CALUDE_cosine_equality_l3730_373021

theorem cosine_equality (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 270) :
  Real.cos (n * π / 180) = Real.cos (890 * π / 180) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l3730_373021


namespace NUMINAMATH_CALUDE_minimum_buses_needed_l3730_373052

def bus_capacity : ℕ := 48
def total_passengers : ℕ := 1230

def buses_needed (capacity : ℕ) (passengers : ℕ) : ℕ :=
  (passengers + capacity - 1) / capacity

theorem minimum_buses_needed : 
  buses_needed bus_capacity total_passengers = 26 := by
  sorry

end NUMINAMATH_CALUDE_minimum_buses_needed_l3730_373052


namespace NUMINAMATH_CALUDE_stating_transfer_equality_l3730_373014

/-- Represents a glass containing a mixture of wine and water -/
structure Glass where
  total_volume : ℝ
  wine_volume : ℝ
  water_volume : ℝ
  volume_constraint : total_volume = wine_volume + water_volume

/-- Represents the state of two glasses after the transfer process -/
structure TransferState where
  wine_glass : Glass
  water_glass : Glass
  volume_conserved : wine_glass.total_volume = water_glass.total_volume

/-- 
Theorem stating that after the transfer process, the volume of wine in the water glass 
is equal to the volume of water in the wine glass 
-/
theorem transfer_equality (state : TransferState) : 
  state.wine_glass.water_volume = state.water_glass.wine_volume := by
  sorry

#check transfer_equality

end NUMINAMATH_CALUDE_stating_transfer_equality_l3730_373014


namespace NUMINAMATH_CALUDE_count_numbers_with_property_l3730_373073

-- Define a two-digit number
def two_digit_number (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

-- Define the property we're interested in
def has_property (a b : ℕ) : Prop :=
  two_digit_number a b ∧ (10 * a + b - (a + b)) % 10 = 6

-- The theorem to prove
theorem count_numbers_with_property :
  ∃ (S : Finset ℕ), S.card = 10 ∧ 
  (∀ n, n ∈ S ↔ ∃ a b, has_property a b ∧ n = 10 * a + b) :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_property_l3730_373073


namespace NUMINAMATH_CALUDE_smallest_primer_l3730_373065

/-- A number is primer if it has a prime number of distinct prime factors -/
def isPrimer (n : ℕ) : Prop :=
  Nat.Prime (Finset.card (Nat.factors n).toFinset)

/-- 6 is the smallest primer number -/
theorem smallest_primer : ∀ k : ℕ, k > 0 → k < 6 → ¬ isPrimer k ∧ isPrimer 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_primer_l3730_373065


namespace NUMINAMATH_CALUDE_tangency_iff_condition_l3730_373054

/-- The line equation -/
def line_equation (x y : ℝ) : Prop :=
  y = 2 * x + 1

/-- The ellipse equation -/
def ellipse_equation (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The tangency condition -/
def tangency_condition (a b : ℝ) : Prop :=
  4 * a^2 + b^2 = 1

/-- Theorem stating that the tangency condition is necessary and sufficient -/
theorem tangency_iff_condition (a b : ℝ) :
  (∃! p : ℝ × ℝ, line_equation p.1 p.2 ∧ ellipse_equation p.1 p.2 a b) ↔ tangency_condition a b :=
sorry

end NUMINAMATH_CALUDE_tangency_iff_condition_l3730_373054


namespace NUMINAMATH_CALUDE_partner_a_receives_4800_l3730_373032

/-- Calculates the money received by partner a in a business partnership --/
def money_received_by_a (a_investment b_investment total_profit : ℚ) : ℚ :=
  let management_fee := 0.1 * total_profit
  let remaining_profit := total_profit - management_fee
  let total_investment := a_investment + b_investment
  let a_profit_share := (a_investment / total_investment) * remaining_profit
  management_fee + a_profit_share

/-- Theorem stating that given the problem conditions, partner a receives 4800 rs --/
theorem partner_a_receives_4800 :
  money_received_by_a 20000 25000 9600 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_partner_a_receives_4800_l3730_373032


namespace NUMINAMATH_CALUDE_shoe_difference_l3730_373024

/-- Given information about shoe boxes and quantities, prove the difference in pairs of shoes. -/
theorem shoe_difference (pairs_per_box : ℕ) (boxes_of_A : ℕ) (B_to_A_ratio : ℕ) : 
  pairs_per_box = 20 →
  boxes_of_A = 8 →
  B_to_A_ratio = 5 →
  B_to_A_ratio * (pairs_per_box * boxes_of_A) - (pairs_per_box * boxes_of_A) = 640 := by
  sorry

end NUMINAMATH_CALUDE_shoe_difference_l3730_373024


namespace NUMINAMATH_CALUDE_square_sum_de_l3730_373085

theorem square_sum_de (a b c d e : ℕ+) 
  (eq1 : (a + 1) * (3 * b * c + 1) = d + 3 * e + 1)
  (eq2 : (b + 1) * (3 * c * a + 1) = 3 * d + e + 13)
  (eq3 : (c + 1) * (3 * a * b + 1) = 4 * (26 - d - e) - 1) :
  d ^ 2 + e ^ 2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_de_l3730_373085


namespace NUMINAMATH_CALUDE_log_one_fifth_25_l3730_373039

-- Define the logarithm function for an arbitrary base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem statement
theorem log_one_fifth_25 : log (1/5) 25 = -2 := by sorry

end NUMINAMATH_CALUDE_log_one_fifth_25_l3730_373039


namespace NUMINAMATH_CALUDE_intersection_probability_formula_l3730_373089

/-- The number of points evenly spaced around the circle -/
def n : ℕ := 2023

/-- The probability of selecting six distinct points A, B, C, D, E, F from n evenly spaced points 
    on a circle, such that chord AB intersects chord CD but neither intersects chord EF -/
def intersection_probability : ℚ :=
  2 * (Nat.choose (n / 2) 2) / Nat.choose n 6

/-- Theorem stating the probability calculation -/
theorem intersection_probability_formula : 
  intersection_probability = 2 * (Nat.choose (n / 2) 2) / Nat.choose n 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_probability_formula_l3730_373089


namespace NUMINAMATH_CALUDE_exam_items_count_l3730_373090

theorem exam_items_count :
  ∀ (total_items : ℕ) (liza_correct : ℕ) (rose_correct : ℕ) (rose_incorrect : ℕ),
    liza_correct = (90 * total_items) / 100 →
    rose_correct = liza_correct + 2 →
    rose_incorrect = 4 →
    total_items = rose_correct + rose_incorrect →
    total_items = 60 := by
  sorry

end NUMINAMATH_CALUDE_exam_items_count_l3730_373090


namespace NUMINAMATH_CALUDE_count_divisible_sum_l3730_373008

theorem count_divisible_sum : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (10 * n) % (n * (n + 1) / 2) = 0) ∧ 
    (∀ n ∉ S, n > 0 → (10 * n) % (n * (n + 1) / 2) ≠ 0) ∧ 
    Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_sum_l3730_373008


namespace NUMINAMATH_CALUDE_proposition_is_false_l3730_373030

theorem proposition_is_false : ∃ (angle1 angle2 : ℝ),
  angle1 + angle2 = 90 ∧ angle1 = angle2 :=
by sorry

end NUMINAMATH_CALUDE_proposition_is_false_l3730_373030


namespace NUMINAMATH_CALUDE_room_puzzle_solution_l3730_373046

/-- Represents a person who can either be a truth-teller or a liar -/
inductive Person
| TruthTeller
| Liar

/-- Represents a statement made by a person -/
structure Statement where
  content : Prop
  speaker : Person

/-- The environment of the problem -/
structure Room where
  people : Nat
  liars : Nat
  statements : List Statement

/-- The correct solution to the problem -/
def correct_solution : Room := { people := 4, liars := 2, statements := [] }

/-- Checks if a given solution is consistent with the statements made -/
def is_consistent (room : Room) : Prop :=
  let s1 := Statement.mk (room.people ≤ 3 ∧ room.liars = room.people) Person.Liar
  let s2 := Statement.mk (room.people ≤ 4 ∧ room.liars < room.people) Person.TruthTeller
  let s3 := Statement.mk (room.people = 5 ∧ room.liars = 3) Person.Liar
  room.statements = [s1, s2, s3]

/-- The main theorem to prove -/
theorem room_puzzle_solution :
  ∀ room : Room, is_consistent room → room = correct_solution :=
sorry

end NUMINAMATH_CALUDE_room_puzzle_solution_l3730_373046


namespace NUMINAMATH_CALUDE_correct_allocation_schemes_l3730_373029

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 6

/-- Represents the number of venues -/
def num_venues : ℕ := 3

/-- Represents the number of volunteers per group -/
def group_size : ℕ := 2

/-- Represents that volunteers A and B must be in the same group -/
def fixed_pair : ℕ := 1

/-- The number of ways to allocate volunteers to venues -/
def allocation_schemes : ℕ := 18

/-- Theorem stating that the number of allocation schemes is correct -/
theorem correct_allocation_schemes :
  (num_volunteers.choose group_size * (num_volunteers - group_size).choose group_size / 2) *
  num_venues.factorial = allocation_schemes := by
  sorry

end NUMINAMATH_CALUDE_correct_allocation_schemes_l3730_373029


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l3730_373026

theorem inscribed_triangle_area (r : ℝ) (A B C : ℝ) :
  r = 18 / Real.pi →
  A = 60 * Real.pi / 180 →
  B = 120 * Real.pi / 180 →
  C = 180 * Real.pi / 180 →
  (1/2) * r^2 * (Real.sin A + Real.sin B + Real.sin C) = 162 * Real.sqrt 3 / Real.pi^2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l3730_373026


namespace NUMINAMATH_CALUDE_cube_color_probability_l3730_373095

def cube_face_colors := Fin 3
def num_faces : Nat := 6

-- Probability of each color
def color_prob : ℚ := 1 / 3

-- Total number of possible color arrangements
def total_arrangements : Nat := 3^num_faces

-- Number of arrangements where all faces are the same color
def all_same_color : Nat := 3

-- Number of arrangements where 5 faces are the same color and 1 is different
def five_same_one_different : Nat := 3 * 6 * 2

-- Number of arrangements where 4 faces are the same color and opposite faces are different
def four_same_opposite_different : Nat := 3 * 3 * 6

-- Total number of suitable arrangements
def suitable_arrangements : Nat := all_same_color + five_same_one_different + four_same_opposite_different

-- Probability of suitable arrangements
def prob_suitable_arrangements : ℚ := suitable_arrangements / total_arrangements

theorem cube_color_probability :
  prob_suitable_arrangements = 31 / 243 :=
sorry

end NUMINAMATH_CALUDE_cube_color_probability_l3730_373095


namespace NUMINAMATH_CALUDE_bird_cost_problem_l3730_373096

/-- The cost of birds in a pet store -/
theorem bird_cost_problem (small_bird_cost big_bird_cost : ℚ) : 
  big_bird_cost = 2 * small_bird_cost →
  5 * big_bird_cost + 3 * small_bird_cost = 5 * small_bird_cost + 3 * big_bird_cost + 20 →
  small_bird_cost = 10 ∧ big_bird_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_bird_cost_problem_l3730_373096


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l3730_373043

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x + 2/x
  else if x > 0 then x * Real.log x - a
  else 0

theorem f_has_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  (-1 / Real.exp 1 < a ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l3730_373043


namespace NUMINAMATH_CALUDE_job_completion_time_equivalence_l3730_373018

/-- Represents the number of days required to complete a job -/
def days_to_complete (num_men : ℕ) (man_days : ℕ) : ℚ :=
  man_days / num_men

theorem job_completion_time_equivalence :
  let initial_men : ℕ := 30
  let initial_days : ℕ := 8
  let new_men : ℕ := 40
  let man_days : ℕ := initial_men * initial_days
  days_to_complete new_men man_days = 6 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_equivalence_l3730_373018


namespace NUMINAMATH_CALUDE_school_students_count_l3730_373053

/-- Proves that given the specified conditions, the total number of students in the school is 387 -/
theorem school_students_count : ∃ (boys girls : ℕ), 
  boys ≥ 150 ∧ 
  boys % 6 = 0 ∧ 
  girls = boys + boys / 20 * 3 ∧ 
  boys + girls ≤ 400 ∧
  boys + girls = 387 := by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l3730_373053


namespace NUMINAMATH_CALUDE_sum_of_digits_squared_difference_l3730_373077

def x : ℕ := 777777777777777
def y : ℕ := 222222222222223

def digit_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + digit_sum (n / 10)

theorem sum_of_digits_squared_difference : 
  digit_sum ((x^2 : ℕ) - (y^2 : ℕ)) = 74 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_squared_difference_l3730_373077


namespace NUMINAMATH_CALUDE_negation_equivalence_l3730_373025

theorem negation_equivalence : 
  ¬(∀ x : ℝ, (x ≠ 3 ∧ x ≠ 2) → (x^2 - 5*x + 6 ≠ 0)) ↔ 
  (∀ x : ℝ, (x = 3 ∨ x = 2) → (x^2 - 5*x + 6 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3730_373025


namespace NUMINAMATH_CALUDE_age_difference_l3730_373082

/-- Given that the total age of A and B is 13 years more than the total age of B and C,
    prove that C is 13 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 13) : A = C + 13 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3730_373082


namespace NUMINAMATH_CALUDE_isosceles_triangle_third_side_l3730_373051

/-- An isosceles triangle with side lengths a, b, and c, where a = b -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : a = b
  isPositive : 0 < a ∧ 0 < b ∧ 0 < c
  triangleInequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: In an isosceles triangle with two sides of lengths 13 and 6, the third side is 13 -/
theorem isosceles_triangle_third_side 
  (t : IsoscelesTriangle) 
  (h1 : t.a = 13 ∨ t.b = 13 ∨ t.c = 13) 
  (h2 : t.a = 6 ∨ t.b = 6 ∨ t.c = 6) : 
  t.c = 13 := by
  sorry

#check isosceles_triangle_third_side

end NUMINAMATH_CALUDE_isosceles_triangle_third_side_l3730_373051
