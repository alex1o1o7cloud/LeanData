import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l3462_346249

theorem system_solution (x y z : ℝ) : 
  ((x + 1) * y * z = 12 ∧ 
   (y + 1) * z * x = 4 ∧ 
   (z + 1) * x * y = 4) ↔ 
  ((x = 2 ∧ y = -2 ∧ z = -2) ∨ 
   (x = 1/3 ∧ y = 3 ∧ z = 3)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3462_346249


namespace NUMINAMATH_CALUDE_solve_equation_l3462_346205

theorem solve_equation (y : ℚ) : (5 * y + 2) / (6 * y - 3) = 3 / 4 ↔ y = -17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3462_346205


namespace NUMINAMATH_CALUDE_tony_sand_and_water_problem_l3462_346291

/-- Represents the problem of Tony filling his sandbox with sand and drinking water --/
theorem tony_sand_and_water_problem 
  (bucket_capacity : ℕ)
  (sandbox_depth sandbox_width sandbox_length : ℕ)
  (sand_weight_per_cubic_foot : ℕ)
  (water_per_session : ℕ)
  (water_bottle_volume : ℕ)
  (water_bottle_cost : ℕ)
  (initial_money : ℕ)
  (change_after_buying : ℕ)
  (h1 : bucket_capacity = 2)
  (h2 : sandbox_depth = 2)
  (h3 : sandbox_width = 4)
  (h4 : sandbox_length = 5)
  (h5 : sand_weight_per_cubic_foot = 3)
  (h6 : water_per_session = 3)
  (h7 : water_bottle_volume = 15)
  (h8 : water_bottle_cost = 2)
  (h9 : initial_money = 10)
  (h10 : change_after_buying = 4) :
  (sandbox_depth * sandbox_width * sandbox_length * sand_weight_per_cubic_foot) / bucket_capacity / 
  ((initial_money - change_after_buying) / water_bottle_cost * water_bottle_volume / water_per_session) = 4 :=
by sorry

end NUMINAMATH_CALUDE_tony_sand_and_water_problem_l3462_346291


namespace NUMINAMATH_CALUDE_lap_length_l3462_346203

/-- Proves that the length of one lap is 1/4 mile, given the total distance and number of laps. -/
theorem lap_length (total_distance : ℚ) (num_laps : ℕ) :
  total_distance = 13/4 ∧ num_laps = 13 →
  total_distance / num_laps = 1/4 := by
sorry

end NUMINAMATH_CALUDE_lap_length_l3462_346203


namespace NUMINAMATH_CALUDE_man_speed_against_current_l3462_346243

/-- A river with three sections and a man traveling along it -/
structure River :=
  (current_speed1 : ℝ)
  (current_speed2 : ℝ)
  (current_speed3 : ℝ)
  (man_speed_with_current1 : ℝ)

/-- Calculate the man's speed against the current in each section -/
def speed_against_current (r : River) : ℝ × ℝ × ℝ :=
  let speed_still_water := r.man_speed_with_current1 - r.current_speed1
  (speed_still_water - r.current_speed1,
   speed_still_water - r.current_speed2,
   speed_still_water - r.current_speed3)

/-- Theorem stating the man's speed against the current in each section -/
theorem man_speed_against_current (r : River) 
  (h1 : r.current_speed1 = 1.5)
  (h2 : r.current_speed2 = 2.5)
  (h3 : r.current_speed3 = 3.5)
  (h4 : r.man_speed_with_current1 = 25) :
  speed_against_current r = (22, 21, 20) :=
sorry


end NUMINAMATH_CALUDE_man_speed_against_current_l3462_346243


namespace NUMINAMATH_CALUDE_tyler_meal_combinations_l3462_346226

def meat_options : ℕ := 3
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 4
def drink_options : ℕ := 3
def vegetables_to_choose : ℕ := 3

def meal_combinations : ℕ :=
  meat_options * Nat.choose vegetable_options vegetables_to_choose * dessert_options * drink_options

theorem tyler_meal_combinations :
  meal_combinations = 360 :=
by sorry

end NUMINAMATH_CALUDE_tyler_meal_combinations_l3462_346226


namespace NUMINAMATH_CALUDE_slower_whale_length_is_45_l3462_346223

/-- The length of a slower whale given the speeds of two whales and the time for the faster to cross the slower -/
def slower_whale_length (faster_speed slower_speed crossing_time : ℝ) : ℝ :=
  (faster_speed - slower_speed) * crossing_time

/-- Theorem stating that the length of the slower whale is 45 meters given the problem conditions -/
theorem slower_whale_length_is_45 :
  slower_whale_length 18 15 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_slower_whale_length_is_45_l3462_346223


namespace NUMINAMATH_CALUDE_bus_driver_distance_to_destination_l3462_346209

theorem bus_driver_distance_to_destination :
  ∀ (distance_to_destination : ℝ),
    (distance_to_destination / 30 + (distance_to_destination + 10) / 30 + 2 = 6) →
    distance_to_destination = 55 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_distance_to_destination_l3462_346209


namespace NUMINAMATH_CALUDE_equation_roots_l3462_346253

theorem equation_roots (c d : ℝ) : 
  (∀ x, (x + c) * (x + d) * (x - 5) / ((x + 4)^2) = 0 → x = -c ∨ x = -d ∨ x = 5) ∧
  (∀ x, x ≠ -4 → (x + c) * (x + d) * (x - 5) / ((x + 4)^2) ≠ 0) ∧
  (∀ x, (x + 2*c) * (x + 6) * (x + 9) / ((x + d) * (x - 5)) = 0 ↔ x = -4) →
  c = 1 ∧ d ≠ -6 ∧ d ≠ -9 ∧ 100 * c + d = 93 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l3462_346253


namespace NUMINAMATH_CALUDE_laundry_calculation_correct_l3462_346206

/-- Represents the laundry problem setup -/
structure LaundrySetup where
  tub_capacity : Real
  clothes_weight : Real
  required_concentration : Real
  initial_detergent : Real

/-- Calculates the additional detergent and water needed for the laundry -/
def calculate_additions (setup : LaundrySetup) : Real × Real :=
  let additional_detergent := setup.tub_capacity * setup.required_concentration - setup.initial_detergent - setup.clothes_weight
  let additional_water := setup.tub_capacity - setup.clothes_weight - setup.initial_detergent - additional_detergent
  (additional_detergent, additional_water)

/-- The main theorem stating the correct additional amounts -/
theorem laundry_calculation_correct (setup : LaundrySetup) 
  (h1 : setup.tub_capacity = 15)
  (h2 : setup.clothes_weight = 4)
  (h3 : setup.required_concentration = 0.004)
  (h4 : setup.initial_detergent = 0.04) :
  calculate_additions setup = (0.004, 10.956) := by
  sorry

#eval calculate_additions { 
  tub_capacity := 15, 
  clothes_weight := 4, 
  required_concentration := 0.004, 
  initial_detergent := 0.04 
}

end NUMINAMATH_CALUDE_laundry_calculation_correct_l3462_346206


namespace NUMINAMATH_CALUDE_angle_of_inclination_sqrt_3_l3462_346228

/-- The angle of inclination (in radians) for a line with slope √3 is π/3 (60°) -/
theorem angle_of_inclination_sqrt_3 :
  let slope : ℝ := Real.sqrt 3
  let angle_of_inclination : ℝ := Real.arctan slope
  angle_of_inclination = π / 3 := by
sorry


end NUMINAMATH_CALUDE_angle_of_inclination_sqrt_3_l3462_346228


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3462_346263

theorem quadratic_root_range (m : ℝ) : 
  (∃ x y : ℝ, x < 1 ∧ y > 1 ∧ 
   x^2 + (m-1)*x + m^2 - 2 = 0 ∧
   y^2 + (m-1)*y + m^2 - 2 = 0) →
  -2 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3462_346263


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_l3462_346219

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define parallelism between planes
def parallel (p1 p2 : Plane3D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

-- Theorem statement
theorem planes_parallel_to_same_plane_are_parallel (p1 p2 p3 : Plane3D) :
  parallel p1 p3 → parallel p2 p3 → parallel p1 p2 := by
  sorry


end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_l3462_346219


namespace NUMINAMATH_CALUDE_soccer_team_non_players_l3462_346246

theorem soccer_team_non_players (total_players : ℕ) (starting_players : ℕ) (first_half_subs : ℕ) :
  total_players = 24 →
  starting_players = 11 →
  first_half_subs = 2 →
  total_players - (starting_players + first_half_subs + 2 * first_half_subs) = 7 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_non_players_l3462_346246


namespace NUMINAMATH_CALUDE_little_john_money_distribution_l3462_346261

theorem little_john_money_distribution 
  (initial_amount : ℚ)
  (spent_on_sweets : ℚ)
  (num_friends : ℕ)
  (remaining_amount : ℚ)
  (h1 : initial_amount = 10.10)
  (h2 : spent_on_sweets = 3.25)
  (h3 : num_friends = 2)
  (h4 : remaining_amount = 2.45) :
  (initial_amount - spent_on_sweets - remaining_amount) / num_friends = 2.20 :=
by sorry

end NUMINAMATH_CALUDE_little_john_money_distribution_l3462_346261


namespace NUMINAMATH_CALUDE_triangle_isosceles_condition_l3462_346210

theorem triangle_isosceles_condition (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = π →          -- Sum of angles in a triangle
  a * Real.cos B + b * Real.cos C + c * Real.cos A = (a + b + c) / 2 →
  (a = b ∨ b = c ∨ c = a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_isosceles_condition_l3462_346210


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3462_346257

theorem expression_simplification_and_evaluation (x : ℝ) (h : x ≠ 1) :
  let expr := ((2 * x + 1) / (x - 1) - 1) / ((x + 2) / ((x - 1)^2))
  expr = x - 1 ∧ (x = 5 → expr = 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3462_346257


namespace NUMINAMATH_CALUDE_v_closed_under_multiplication_v_not_closed_under_add_cube_root_v_not_closed_under_division_v_not_closed_under_cube_cube_root_l3462_346275

-- Define the set v of cubes of positive integers
def v : Set ℕ := {n : ℕ | ∃ m : ℕ, n = m ^ 3}

-- Closure under multiplication
theorem v_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v :=
sorry

-- Not closed under addition followed by cube root
theorem v_not_closed_under_add_cube_root :
  ∃ a b : ℕ, a ∈ v ∧ b ∈ v ∧ (∃ c : ℕ, c ^ 3 = a + b) → (∃ d : ℕ, d ^ 3 = a + b) :=
sorry

-- Not closed under division
theorem v_not_closed_under_division :
  ∃ a b : ℕ, a ∈ v ∧ b ∈ v ∧ b ≠ 0 → (a / b) ∉ v :=
sorry

-- Not closed under cubing followed by cube root
theorem v_not_closed_under_cube_cube_root :
  ∃ a : ℕ, a ∈ v ∧ (∃ b : ℕ, b ^ 3 = a ^ 3) → (∃ c : ℕ, c ^ 3 = a ^ 3) :=
sorry

end NUMINAMATH_CALUDE_v_closed_under_multiplication_v_not_closed_under_add_cube_root_v_not_closed_under_division_v_not_closed_under_cube_cube_root_l3462_346275


namespace NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l3462_346245

/-- A polynomial over the complex numbers. -/
def ComplexPolynomial := ℂ → ℂ

/-- Predicate for even functions. -/
def IsEven (P : ComplexPolynomial) : Prop :=
  ∀ z : ℂ, P z = P (-z)

/-- The main theorem: A complex polynomial is even if and only if
    it can be expressed as the product of a polynomial and its negation. -/
theorem even_polynomial_iff_product_with_negation (P : ComplexPolynomial) :
  IsEven P ↔ ∃ Q : ComplexPolynomial, ∀ z : ℂ, P z = (Q z) * (Q (-z)) := by
  sorry

end NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l3462_346245


namespace NUMINAMATH_CALUDE_consecutive_balls_probability_l3462_346221

theorem consecutive_balls_probability (n : ℕ) : n = 48 → 
  (Nat.choose (n - 6) 7 : ℚ) / (Nat.choose n 7 : ℚ) = 
  (6 * (Nat.choose (n - 7) 6 : ℚ)) / (Nat.choose n 7 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_balls_probability_l3462_346221


namespace NUMINAMATH_CALUDE_quadratic_properties_l3462_346297

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

/-- Theorem stating that f satisfies the required conditions -/
theorem quadratic_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3462_346297


namespace NUMINAMATH_CALUDE_pond_volume_l3462_346278

/-- The volume of a rectangular prism given its length, width, and height -/
def volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a rectangular prism with dimensions 20 m × 10 m × 8 m is 1600 cubic meters -/
theorem pond_volume : volume 20 10 8 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_l3462_346278


namespace NUMINAMATH_CALUDE_gcd_490_910_l3462_346215

theorem gcd_490_910 : Nat.gcd 490 910 = 70 := by
  sorry

end NUMINAMATH_CALUDE_gcd_490_910_l3462_346215


namespace NUMINAMATH_CALUDE_square_area_from_string_length_l3462_346202

theorem square_area_from_string_length (string_length : ℝ) (h : string_length = 32) :
  let side_length := string_length / 4
  side_length * side_length = 64 := by sorry

end NUMINAMATH_CALUDE_square_area_from_string_length_l3462_346202


namespace NUMINAMATH_CALUDE_hyperbola_center_l3462_346229

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) : 
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (2, 3) → f2 = (-4, 7) → center = (-1, 5) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l3462_346229


namespace NUMINAMATH_CALUDE_place_value_comparison_l3462_346293

def number : ℚ := 52648.2097

def tens_place_value : ℚ := 10
def tenths_place_value : ℚ := 0.1

theorem place_value_comparison : 
  tens_place_value / tenths_place_value = 100 := by sorry

end NUMINAMATH_CALUDE_place_value_comparison_l3462_346293


namespace NUMINAMATH_CALUDE_stating_lens_screen_distance_l3462_346208

/-- Represents the focal length of a thin lens in centimeters -/
def focal_length : ℝ := 150

/-- Represents the distance the screen is moved in centimeters -/
def screen_movement : ℝ := 40

/-- Represents the possible initial distances from the lens to the screen in centimeters -/
def initial_distances : Set ℝ := {130, 170}

/-- 
Theorem stating that given a thin lens with focal length of 150 cm and a screen
that produces the same diameter spot when moved 40 cm, the initial distance
from the lens to the screen is either 130 cm or 170 cm.
-/
theorem lens_screen_distance 
  (s : ℝ) 
  (h1 : s ∈ initial_distances) 
  (h2 : s = focal_length + screen_movement / 2 ∨ s = focal_length - screen_movement / 2) : 
  s ∈ initial_distances :=
sorry

end NUMINAMATH_CALUDE_stating_lens_screen_distance_l3462_346208


namespace NUMINAMATH_CALUDE_vegetable_factory_profit_profit_function_correct_l3462_346230

/-- Represents the net profit function for a vegetable processing factory -/
def net_profit (n : ℕ) : ℚ :=
  -4 * n^2 + 80 * n - 144

/-- Represents the year when the business starts making a net profit -/
def profit_start_year : ℕ := 3

theorem vegetable_factory_profit :
  (∀ n : ℕ, n < profit_start_year → net_profit n ≤ 0) ∧
  (∀ n : ℕ, n ≥ profit_start_year → net_profit n > 0) :=
sorry

theorem profit_function_correct (n : ℕ) :
  net_profit n = n * 1 - (0.24 * n + n * (n - 1) / 2 * 0.08) - 1.44 :=
sorry

end NUMINAMATH_CALUDE_vegetable_factory_profit_profit_function_correct_l3462_346230


namespace NUMINAMATH_CALUDE_pencil_cost_l3462_346213

/-- The cost of a pencil given the total cost with an eraser and the price difference -/
theorem pencil_cost (total : ℝ) (difference : ℝ) (h1 : total = 3.4) (h2 : difference = 3) :
  ∃ (pencil eraser : ℝ),
    pencil + eraser = total ∧
    pencil = eraser + difference ∧
    pencil = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l3462_346213


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l3462_346282

/-- The number of members in the basketball team -/
def team_size : ℕ := 12

/-- The number of positions in the starting lineup -/
def lineup_size : ℕ := 5

/-- The number of ways to choose a starting lineup -/
def lineup_choices : ℕ := team_size * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4)

theorem starting_lineup_combinations : 
  lineup_choices = 95040 :=
sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l3462_346282


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3462_346294

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The problem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 2)
  (h2 : seq.S 3 = 12) :
  seq.a 5 = 10 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3462_346294


namespace NUMINAMATH_CALUDE_olivias_birthday_meals_l3462_346212

/-- Given that each meal costs 7 dollars and Olivia's dad spent a total of 21 dollars,
    prove that the number of meals he paid for is 3. -/
theorem olivias_birthday_meals (cost_per_meal : ℕ) (total_spent : ℕ) (num_meals : ℕ) :
  cost_per_meal = 7 →
  total_spent = 21 →
  num_meals * cost_per_meal = total_spent →
  num_meals = 3 := by
  sorry

end NUMINAMATH_CALUDE_olivias_birthday_meals_l3462_346212


namespace NUMINAMATH_CALUDE_smallest_divisor_for_perfect_cube_l3462_346227

theorem smallest_divisor_for_perfect_cube (n : ℕ) : 
  (n > 0 ∧ ∃ (k : ℕ), 3600 / n = k^3 ∧ ∀ (m : ℕ), m > 0 → m < n → ¬∃ (j : ℕ), 3600 / m = j^3) → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_for_perfect_cube_l3462_346227


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l3462_346296

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 1

-- State the theorem
theorem even_function_implies_a_zero :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l3462_346296


namespace NUMINAMATH_CALUDE_max_b_in_box_l3462_346281

theorem max_b_in_box (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_max_b_in_box_l3462_346281


namespace NUMINAMATH_CALUDE_brown_family_seating_l3462_346237

/-- The number of ways to seat n children in a circle. -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to seat b boys and g girls in a circle
    such that at least two boys are next to each other. -/
def boysNextToEachOther (b g : ℕ) : ℕ :=
  if b > g + 1 then circularArrangements (b + g) else 0

theorem brown_family_seating :
  boysNextToEachOther 5 3 = 5040 := by sorry

end NUMINAMATH_CALUDE_brown_family_seating_l3462_346237


namespace NUMINAMATH_CALUDE_no_fourth_power_sum_1599_l3462_346295

theorem no_fourth_power_sum_1599 :
  ¬ ∃ (s : Finset ℕ), (∀ n ∈ s, ∃ k, n = k^4) ∧ s.card ≤ 14 ∧ s.sum id = 1599 := by
  sorry

end NUMINAMATH_CALUDE_no_fourth_power_sum_1599_l3462_346295


namespace NUMINAMATH_CALUDE_min_value_theorem_l3462_346220

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 3) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → 1/x + 4/(5+y) ≥ 9/8) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 3 ∧ 1/x + 4/(5+y) = 9/8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3462_346220


namespace NUMINAMATH_CALUDE_sarah_bus_time_l3462_346234

-- Define the problem parameters
def leave_time : Nat := 7 * 60 + 45  -- 7:45 AM in minutes
def return_time : Nat := 17 * 60 + 15  -- 5:15 PM in minutes
def num_classes : Nat := 8
def class_duration : Nat := 45
def lunch_break : Nat := 30
def extracurricular_time : Nat := 90  -- 1 hour and 30 minutes in minutes

-- Define the theorem
theorem sarah_bus_time :
  let total_time := return_time - leave_time
  let school_time := num_classes * class_duration + lunch_break + extracurricular_time
  total_time - school_time = 90 := by
  sorry

end NUMINAMATH_CALUDE_sarah_bus_time_l3462_346234


namespace NUMINAMATH_CALUDE_sams_total_nickels_l3462_346289

/-- Sam's initial number of nickels -/
def initial_nickels : ℕ := 24

/-- Number of nickels Sam's dad gave him -/
def additional_nickels : ℕ := 39

/-- Theorem: Sam's total number of nickels after receiving more from his dad -/
theorem sams_total_nickels : initial_nickels + additional_nickels = 63 := by
  sorry

end NUMINAMATH_CALUDE_sams_total_nickels_l3462_346289


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3462_346251

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3462_346251


namespace NUMINAMATH_CALUDE_percentage_problem_l3462_346268

theorem percentage_problem (P : ℝ) : P = 0.7 ↔ 
  0.8 * 90 = P * 60.00000000000001 + 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3462_346268


namespace NUMINAMATH_CALUDE_exactly_one_topic_not_chosen_l3462_346267

/-- The number of ways for n teachers to choose from m topics with replacement. -/
def choose_with_replacement (n m : ℕ) : ℕ := m ^ n

/-- The number of ways to arrange n items. -/
def arrangement (n : ℕ) : ℕ := n.factorial

/-- The number of ways for n teachers to choose from m topics with replacement,
    such that exactly one topic is not chosen. -/
def one_topic_not_chosen (n m : ℕ) : ℕ :=
  choose_with_replacement n m -
  (m * choose_with_replacement (n - 1) (m - 1)) -
  arrangement m

theorem exactly_one_topic_not_chosen :
  one_topic_not_chosen 4 4 = 112 := by sorry

end NUMINAMATH_CALUDE_exactly_one_topic_not_chosen_l3462_346267


namespace NUMINAMATH_CALUDE_divisors_sum_and_product_l3462_346269

theorem divisors_sum_and_product (p : ℕ) (hp : Prime p) :
  let a := p^106
  ∀ (divisors : Finset ℕ), 
    (∀ d ∈ divisors, d ∣ a) ∧ 
    (∀ d : ℕ, d ∣ a → d ∈ divisors) ∧ 
    (Finset.card divisors = 107) →
    (divisors.sum id = (p^107 - 1) / (p - 1)) ∧
    (divisors.prod id = p^11321) := by
  sorry

end NUMINAMATH_CALUDE_divisors_sum_and_product_l3462_346269


namespace NUMINAMATH_CALUDE_find_multiple_l3462_346271

theorem find_multiple (x y m : ℝ) : 
  x + y = 8 → 
  y - m * x = 7 → 
  y - x = 7.5 → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_find_multiple_l3462_346271


namespace NUMINAMATH_CALUDE_train_passing_platform_l3462_346254

/-- Time taken for a train to pass a platform -/
theorem train_passing_platform (train_length platform_length : ℝ) (train_speed : ℝ) : 
  train_length = 360 →
  platform_length = 140 →
  train_speed = 45 →
  (train_length + platform_length) / (train_speed * 1000 / 3600) = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_train_passing_platform_l3462_346254


namespace NUMINAMATH_CALUDE_number_times_power_of_five_l3462_346277

theorem number_times_power_of_five (x : ℝ) : x * (5^4) = 75625 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_number_times_power_of_five_l3462_346277


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3462_346290

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- If a₂ + a₄ = 2 and S₂ + S₄ = 1 for an arithmetic sequence, then a₁₀ = 8 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 2 + seq.a 4 = 2) 
  (h2 : seq.S 2 + seq.S 4 = 1) : 
  seq.a 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3462_346290


namespace NUMINAMATH_CALUDE_divisible_by_21_l3462_346216

theorem divisible_by_21 (N : Finset ℕ) 
  (h_card : N.card = 46)
  (h_div_3 : (N.filter (fun n => n % 3 = 0)).card = 35)
  (h_div_7 : (N.filter (fun n => n % 7 = 0)).card = 12) :
  ∃ n ∈ N, n % 21 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_21_l3462_346216


namespace NUMINAMATH_CALUDE_division_by_fraction_twelve_divided_by_one_fourth_l3462_346286

theorem division_by_fraction (a b : ℚ) (hb : b ≠ 0) :
  a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_fourth :
  12 / (1 / 4) = 48 := by sorry

end NUMINAMATH_CALUDE_division_by_fraction_twelve_divided_by_one_fourth_l3462_346286


namespace NUMINAMATH_CALUDE_bank_deposit_exceeds_400_l3462_346266

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem bank_deposit_exceeds_400 :
  let a := 2  -- Initial deposit in cents
  let r := 3  -- Common ratio
  let target := 40000  -- Target amount in cents
  ∀ n : ℕ, n < 10 → geometric_sum a r n ≤ target ∧
  geometric_sum a r 10 > target ∧
  day_of_week 10 = "Tuesday" :=
by sorry

end NUMINAMATH_CALUDE_bank_deposit_exceeds_400_l3462_346266


namespace NUMINAMATH_CALUDE_eliana_steps_proof_l3462_346225

-- Define the number of steps for each day
def first_day_morning_steps : ℕ := 200
def first_day_additional_steps : ℕ := 300
def third_day_additional_steps : ℕ := 100

-- Define the total steps for the first day
def first_day_total : ℕ := first_day_morning_steps + first_day_additional_steps

-- Define the total steps for the second day
def second_day_total : ℕ := 2 * first_day_total

-- Define the total steps for all three days
def total_steps : ℕ := first_day_total + second_day_total + third_day_additional_steps

-- Theorem statement
theorem eliana_steps_proof : total_steps = 1600 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_proof_l3462_346225


namespace NUMINAMATH_CALUDE_removed_triangles_area_l3462_346256

-- Define the square side length
def square_side : ℝ := 16

-- Define the ratio of r to s
def r_to_s_ratio : ℝ := 3

-- Theorem statement
theorem removed_triangles_area (r s : ℝ) : 
  r / s = r_to_s_ratio →
  (r + s)^2 + (r - s)^2 = square_side^2 →
  4 * (1/2 * r * s) = 76.8 := by
  sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l3462_346256


namespace NUMINAMATH_CALUDE_room_freezer_temp_difference_l3462_346201

-- Define the temperatures
def freezer_temp : Int := -4
def room_temp : Int := 18

-- Define the temperature difference function
def temp_difference (room : Int) (freezer : Int) : Int :=
  room - freezer

-- Theorem to prove
theorem room_freezer_temp_difference :
  temp_difference room_temp freezer_temp = 22 := by
  sorry

end NUMINAMATH_CALUDE_room_freezer_temp_difference_l3462_346201


namespace NUMINAMATH_CALUDE_hotel_room_charge_difference_l3462_346232

theorem hotel_room_charge_difference (G : ℝ) (h1 : G > 0) : 
  let R := 1.5000000000000002 * G
  let P := 0.6 * R
  (G - P) / G * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_hotel_room_charge_difference_l3462_346232


namespace NUMINAMATH_CALUDE_chord_length_squared_l3462_346280

/-- Given three circles with radii 4, 7, and 9, where the circles with radii 4 and 7 
    are externally tangent to each other and internally tangent to the circle with radius 9, 
    the square of the length of the chord of the circle with radius 9 that is a common 
    external tangent to the other two circles is equal to 224. -/
theorem chord_length_squared (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 7) (h₃ : r₃ = 9) 
  (h_ext_tangent : r₃ = r₁ + r₂) 
  (h_int_tangent₁ : r₃ - r₁ = r₂) (h_int_tangent₂ : r₃ - r₂ = r₁) : 
  ∃ (chord_length : ℝ), chord_length^2 = 224 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l3462_346280


namespace NUMINAMATH_CALUDE_circle_radius_proof_l3462_346236

theorem circle_radius_proof (r : ℝ) : 
  r > 0 → 
  (π * r^2 = 3 * (2 * π * r)) → 
  (π * r^2 + 2 * π * r = 100 * π) → 
  r = 12.5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l3462_346236


namespace NUMINAMATH_CALUDE_solve_barnyard_owl_problem_l3462_346272

def barnyard_owl_problem (hoots_per_owl : ℕ) (total_hoots : ℕ) : Prop :=
  let num_owls := (20 - 5) / hoots_per_owl
  hoots_per_owl = 5 ∧ total_hoots = 20 - 5 → num_owls = 3

theorem solve_barnyard_owl_problem :
  ∃ (hoots_per_owl total_hoots : ℕ), barnyard_owl_problem hoots_per_owl total_hoots :=
sorry

end NUMINAMATH_CALUDE_solve_barnyard_owl_problem_l3462_346272


namespace NUMINAMATH_CALUDE_milkshake_cost_l3462_346292

theorem milkshake_cost (initial_amount : ℕ) (hamburger_cost : ℕ) (num_hamburgers : ℕ) (num_milkshakes : ℕ) (remaining_amount : ℕ) :
  initial_amount = 120 →
  hamburger_cost = 4 →
  num_hamburgers = 8 →
  num_milkshakes = 6 →
  remaining_amount = 70 →
  ∃ (milkshake_cost : ℕ), 
    initial_amount - (hamburger_cost * num_hamburgers) - (milkshake_cost * num_milkshakes) = remaining_amount ∧
    milkshake_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_milkshake_cost_l3462_346292


namespace NUMINAMATH_CALUDE_oil_purchase_amount_l3462_346250

/-- Proves that the amount spent on oil is Rs. 600 given the conditions of the problem -/
theorem oil_purchase_amount (original_price : ℝ) (reduced_price : ℝ) (additional_oil : ℝ) 
  (h1 : reduced_price = original_price * 0.75)
  (h2 : reduced_price = 30)
  (h3 : additional_oil = 5) :
  ∃ (amount_spent : ℝ), 
    amount_spent / reduced_price - amount_spent / original_price = additional_oil ∧ 
    amount_spent = 600 := by
  sorry

end NUMINAMATH_CALUDE_oil_purchase_amount_l3462_346250


namespace NUMINAMATH_CALUDE_simplify_expression_l3462_346287

theorem simplify_expression (x : ℝ) (h : x^2 ≠ 1) :
  Real.sqrt (1 + ((x^4 + 1) / (2 * x^2))^2) = (Real.sqrt (x^8 + 6 * x^4 + 1)) / (2 * x^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3462_346287


namespace NUMINAMATH_CALUDE_pudding_cups_problem_l3462_346273

theorem pudding_cups_problem (students : ℕ) (additional_cups : ℕ) 
  (h1 : students = 218) 
  (h2 : additional_cups = 121) : 
  ∃ initial_cups : ℕ, 
    initial_cups + additional_cups = students ∧ 
    initial_cups = 97 := by
  sorry

end NUMINAMATH_CALUDE_pudding_cups_problem_l3462_346273


namespace NUMINAMATH_CALUDE_rocket_max_height_l3462_346262

/-- Rocket's maximum height calculation --/
theorem rocket_max_height (a g : ℝ) (τ : ℝ) (h : a > g) (h_a : a = 30) (h_g : g = 10) (h_τ : τ = 30) :
  let v₀ := a * τ
  let y₀ := a * τ^2 / 2
  let t := v₀ / g
  let y_max := y₀ + v₀ * t - g * t^2 / 2
  y_max = 54000 ∧ y_max > 50000 := by
  sorry

#check rocket_max_height

end NUMINAMATH_CALUDE_rocket_max_height_l3462_346262


namespace NUMINAMATH_CALUDE_max_AB_is_five_l3462_346240

/-- Represents a convex quadrilateral ABCD inscribed in a circle -/
structure CyclicQuadrilateral where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  AB_shortest : AB ≤ BC ∧ AB ≤ CD ∧ AB ≤ DA
  distinct_sides : AB ≠ BC ∧ AB ≠ CD ∧ AB ≠ DA ∧ BC ≠ CD ∧ BC ≠ DA ∧ CD ≠ DA
  max_side_10 : AB ≤ 10 ∧ BC ≤ 10 ∧ CD ≤ 10 ∧ DA ≤ 10
  area_ratio_int : ∃ k : ℕ, BC * CD = k * DA * AB

/-- The maximum possible value of AB in a CyclicQuadrilateral is 5 -/
theorem max_AB_is_five (q : CyclicQuadrilateral) : q.AB ≤ 5 :=
  sorry

end NUMINAMATH_CALUDE_max_AB_is_five_l3462_346240


namespace NUMINAMATH_CALUDE_nacho_triple_divya_age_l3462_346274

/-- Represents the number of years in the future when Nacho will be three times older than Divya -/
def future_years : ℕ := 10

/-- Divya's current age -/
def divya_age : ℕ := 5

/-- The sum of Nacho's and Divya's current ages -/
def total_current_age : ℕ := 40

/-- Nacho's current age -/
def nacho_age : ℕ := total_current_age - divya_age

theorem nacho_triple_divya_age : 
  nacho_age + future_years = 3 * (divya_age + future_years) :=
sorry

end NUMINAMATH_CALUDE_nacho_triple_divya_age_l3462_346274


namespace NUMINAMATH_CALUDE_max_garden_area_l3462_346265

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Calculates the fencing required for three sides of a rectangular garden -/
def fencingRequired (d : GardenDimensions) : ℝ :=
  d.length + 2 * d.width

/-- Theorem: The maximum area of a rectangular garden with 400 feet of fencing
    for three sides is 20000 square feet -/
theorem max_garden_area :
  ∃ (d : GardenDimensions),
    fencingRequired d = 400 ∧
    ∀ (d' : GardenDimensions), fencingRequired d' = 400 →
      gardenArea d' ≤ gardenArea d ∧
      gardenArea d = 20000 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l3462_346265


namespace NUMINAMATH_CALUDE_moon_speed_km_per_hour_l3462_346252

/-- The speed of the moon around the earth in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.05

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_sec_to_km_per_hour (speed_km_per_sec : ℝ) : ℝ :=
  speed_km_per_sec * seconds_per_hour

/-- Theorem stating that the moon's speed in kilometers per hour is 3780 -/
theorem moon_speed_km_per_hour :
  km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3780 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_km_per_hour_l3462_346252


namespace NUMINAMATH_CALUDE_total_selling_price_l3462_346235

/-- Calculate the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
theorem total_selling_price
  (quantity : ℕ)
  (profit_per_meter : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : quantity = 85)
  (h2 : profit_per_meter = 15)
  (h3 : cost_price_per_meter = 85) :
  quantity * (cost_price_per_meter + profit_per_meter) = 8500 := by
  sorry

#check total_selling_price

end NUMINAMATH_CALUDE_total_selling_price_l3462_346235


namespace NUMINAMATH_CALUDE_average_price_is_16_l3462_346238

/-- The average price of books bought by Rahim -/
def average_price_per_book (books_shop1 books_shop2 : ℕ) (price_shop1 price_shop2 : ℕ) : ℚ :=
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2)

/-- Theorem stating that the average price per book is 16 given the problem conditions -/
theorem average_price_is_16 :
  average_price_per_book 55 60 1500 340 = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_price_is_16_l3462_346238


namespace NUMINAMATH_CALUDE_magnitude_b_cos_angle_ab_l3462_346298

-- Define the vectors
def a : ℝ × ℝ := (4, 3)
def b : ℝ × ℝ := (-1, 2)

-- Theorem for the magnitude of vector b
theorem magnitude_b : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = Real.sqrt 5 := by sorry

-- Theorem for the cosine of the angle between vectors a and b
theorem cos_angle_ab : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) * Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))) 
  = (2 * Real.sqrt 5) / 25 := by sorry

end NUMINAMATH_CALUDE_magnitude_b_cos_angle_ab_l3462_346298


namespace NUMINAMATH_CALUDE_apple_sale_theorem_l3462_346241

/-- Calculates the total number of apples sold given the number of red apples and the ratio of red:green:yellow apples -/
def total_apples (red_apples : ℕ) (red_ratio green_ratio yellow_ratio : ℕ) : ℕ :=
  let total_ratio := red_ratio + green_ratio + yellow_ratio
  let apples_per_part := red_apples / red_ratio
  red_apples + (green_ratio * apples_per_part) + (yellow_ratio * apples_per_part)

/-- Theorem stating that given 32 red apples and a ratio of 8:3:5 for red:green:yellow apples, the total number of apples sold is 64 -/
theorem apple_sale_theorem : total_apples 32 8 3 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_apple_sale_theorem_l3462_346241


namespace NUMINAMATH_CALUDE_product_evaluation_l3462_346239

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3462_346239


namespace NUMINAMATH_CALUDE_quadratic_solution_value_l3462_346218

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The solution set of the inequality f(x) < c -/
structure SolutionSet (f : ℝ → ℝ) (c : ℝ) where
  m : ℝ
  property : Set.Ioo m (m + 6) = {x | f x < c}

/-- The theorem stating that c = 9 given the conditions -/
theorem quadratic_solution_value
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h_f : f = QuadraticFunction a b)
  (h_range : Set.range f = Set.Ici 0)
  (h_solution : SolutionSet f c)
  : c = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_value_l3462_346218


namespace NUMINAMATH_CALUDE_travel_time_is_50_minutes_l3462_346284

/-- Represents a tram system with stations A and B -/
structure TramSystem where
  departure_interval : ℕ  -- Interval between tram departures from A in minutes
  journey_time : ℕ        -- Time for a tram to travel from A to B in minutes

/-- Represents a person cycling from B to A -/
structure Cyclist where
  trams_encountered : ℕ   -- Number of trams encountered during the journey

/-- Calculates the time taken for the cyclist to travel from B to A -/
def travel_time (system : TramSystem) (cyclist : Cyclist) : ℕ :=
  cyclist.trams_encountered * system.departure_interval

/-- Theorem stating the travel time for the given scenario -/
theorem travel_time_is_50_minutes 
  (system : TramSystem) 
  (cyclist : Cyclist) 
  (h1 : system.departure_interval = 5)
  (h2 : system.journey_time = 15)
  (h3 : cyclist.trams_encountered = 10) :
  travel_time system cyclist = 50 := by
  sorry

#eval travel_time ⟨5, 15⟩ ⟨10⟩

end NUMINAMATH_CALUDE_travel_time_is_50_minutes_l3462_346284


namespace NUMINAMATH_CALUDE_pelicans_in_shark_bite_cove_l3462_346222

/-- The number of Pelicans remaining in Shark Bite Cove after some have moved to Pelican Bay -/
def remaining_pelicans (initial_pelicans : ℕ) : ℕ :=
  initial_pelicans - initial_pelicans / 3

/-- The theorem stating the number of remaining Pelicans in Shark Bite Cove -/
theorem pelicans_in_shark_bite_cove :
  ∃ (initial_pelicans : ℕ),
    (2 * initial_pelicans = 60) ∧
    (remaining_pelicans initial_pelicans = 20) := by
  sorry

end NUMINAMATH_CALUDE_pelicans_in_shark_bite_cove_l3462_346222


namespace NUMINAMATH_CALUDE_smallest_x_abs_equation_l3462_346255

theorem smallest_x_abs_equation : ∃ x : ℝ, (∀ y : ℝ, |y + 3| = 15 → x ≤ y) ∧ |x + 3| = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_abs_equation_l3462_346255


namespace NUMINAMATH_CALUDE_shoe_pairs_calculation_shoe_pairs_proof_l3462_346299

/-- Given a total number of shoes and the probability of selecting two shoes of the same color
    without replacement, calculate the number of pairs of shoes. -/
theorem shoe_pairs_calculation (total_shoes : ℕ) (probability : ℚ) : ℕ :=
  let pairs := total_shoes / 2
  let calculated_prob := 1 / (total_shoes - 1 : ℚ)
  if total_shoes = 12 ∧ probability = 1/11 ∧ calculated_prob = probability
  then pairs
  else 0

/-- Prove that given 12 shoes in total and a probability of 1/11 for selecting 2 shoes
    of the same color without replacement, the number of pairs of shoes is 6. -/
theorem shoe_pairs_proof :
  shoe_pairs_calculation 12 (1/11) = 6 := by
  sorry

end NUMINAMATH_CALUDE_shoe_pairs_calculation_shoe_pairs_proof_l3462_346299


namespace NUMINAMATH_CALUDE_investment_calculation_correct_l3462_346288

/-- Calculates the total investment given share details and dividend income -/
def calculate_investment (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : ℚ :=
  let dividend_per_share := (dividend_rate / 100) * face_value
  let num_shares := annual_income / dividend_per_share
  num_shares * quoted_price

/-- Theorem stating that the investment calculation is correct for the given problem -/
theorem investment_calculation_correct :
  calculate_investment 10 8.25 12 648 = 4455 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_correct_l3462_346288


namespace NUMINAMATH_CALUDE_cos_15_degrees_l3462_346242

theorem cos_15_degrees : Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_15_degrees_l3462_346242


namespace NUMINAMATH_CALUDE_students_either_not_both_is_38_l3462_346285

/-- The number of students taking either geometry or history but not both -/
def students_either_not_both (students_both : ℕ) (students_geometry : ℕ) (students_only_history : ℕ) : ℕ :=
  (students_geometry - students_both) + students_only_history

/-- Theorem stating the number of students taking either geometry or history but not both -/
theorem students_either_not_both_is_38 :
  students_either_not_both 15 35 18 = 38 := by
  sorry

#check students_either_not_both_is_38

end NUMINAMATH_CALUDE_students_either_not_both_is_38_l3462_346285


namespace NUMINAMATH_CALUDE_integer_root_values_l3462_346276

theorem integer_root_values (b : ℤ) : 
  (∃ x : ℤ, x^3 + 2*x^2 + b*x + 18 = 0) ↔ b ∈ ({-21, 19, -17, -4, 3} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_root_values_l3462_346276


namespace NUMINAMATH_CALUDE_rulers_in_drawer_l3462_346211

/-- The number of rulers remaining in a drawer after some are removed -/
def rulers_remaining (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem: Given 46 initial rulers and 25 removed, 21 rulers remain -/
theorem rulers_in_drawer : rulers_remaining 46 25 = 21 := by
  sorry

end NUMINAMATH_CALUDE_rulers_in_drawer_l3462_346211


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3462_346200

-- Define the universal set I
def I : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x * (x - 1) ≥ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3462_346200


namespace NUMINAMATH_CALUDE_union_and_complement_of_sets_l3462_346207

-- Define the sets A and B
def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem union_and_complement_of_sets :
  (A ∪ B = {x | x ≤ 3 ∨ x > 4}) ∧
  ((Set.univ \ A) ∪ (Set.univ \ B) = {x | x < -4 ∨ x ≥ -1}) := by
  sorry

end NUMINAMATH_CALUDE_union_and_complement_of_sets_l3462_346207


namespace NUMINAMATH_CALUDE_sqrt_seven_to_six_l3462_346214

theorem sqrt_seven_to_six : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_six_l3462_346214


namespace NUMINAMATH_CALUDE_abs_diff_geq_sum_abs_iff_product_nonpositive_l3462_346224

theorem abs_diff_geq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  a * b ≤ 0 ↔ |a - b| ≥ |a| + |b| := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_geq_sum_abs_iff_product_nonpositive_l3462_346224


namespace NUMINAMATH_CALUDE_canoe_trip_distance_l3462_346279

theorem canoe_trip_distance 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (total_time : ℝ) 
  (distance : ℝ) :
  upstream_speed = 3 →
  downstream_speed = 9 →
  total_time = 8 →
  distance / upstream_speed + distance / downstream_speed = total_time →
  distance = 18 := by
sorry

end NUMINAMATH_CALUDE_canoe_trip_distance_l3462_346279


namespace NUMINAMATH_CALUDE_solutions_of_equation_1_sum_of_reciprocals_squared_difference_of_solutions_l3462_346231

-- Question 1
theorem solutions_of_equation_1 (x : ℝ) :
  (x + 5 / x = -6) ↔ (x = -1 ∨ x = -5) :=
sorry

-- Question 2
theorem sum_of_reciprocals (m n : ℝ) :
  (m - 3 / m = 4) ∧ (n - 3 / n = 4) → 1 / m + 1 / n = -4 / 3 :=
sorry

-- Question 3
theorem squared_difference_of_solutions (a : ℝ) (x₁ x₂ : ℝ) :
  a ≠ 0 →
  (x₁ + (a^2 + 2*a) / (x₁ + 1) = 2*a + 1) →
  (x₂ + (a^2 + 2*a) / (x₂ + 1) = 2*a + 1) →
  (x₁ - x₂)^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_solutions_of_equation_1_sum_of_reciprocals_squared_difference_of_solutions_l3462_346231


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_of_factorials_15_l3462_346264

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_sum_of_factorials_15 :
  last_two_digits (sum_of_factorials 15) = 13 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_of_factorials_15_l3462_346264


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_equals_four_l3462_346270

theorem sqrt_neg_four_squared_equals_four : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_equals_four_l3462_346270


namespace NUMINAMATH_CALUDE_someone_next_to_two_economists_l3462_346259

/-- Represents the profession of a person -/
inductive Profession
| Accountant
| Manager
| Economist

/-- Represents a circular arrangement of people -/
def CircularArrangement := List Profession

/-- Counts the number of accountants sitting next to at least one economist -/
def accountantsNextToEconomist (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Counts the number of managers sitting next to at least one economist -/
def managersNextToEconomist (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Checks if there's someone sitting next to two economists -/
def someoneNextToTwoEconomists (arrangement : CircularArrangement) : Bool :=
  sorry

theorem someone_next_to_two_economists 
  (arrangement : CircularArrangement) : 
  accountantsNextToEconomist arrangement = 20 →
  managersNextToEconomist arrangement = 25 →
  someoneNextToTwoEconomists arrangement = true :=
by sorry

end NUMINAMATH_CALUDE_someone_next_to_two_economists_l3462_346259


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3462_346283

/-- The distance between Stockholm and Uppsala on a map in centimeters -/
def map_distance : ℝ := 45

/-- The scale of the map, representing how many kilometers in reality one centimeter on the map represents -/
def map_scale : ℝ := 10

/-- The actual distance between Stockholm and Uppsala in kilometers -/
def actual_distance : ℝ := map_distance * map_scale

theorem stockholm_uppsala_distance :
  actual_distance = 450 :=
by sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3462_346283


namespace NUMINAMATH_CALUDE_derangement_even_index_odd_l3462_346260

/-- Definition of derangement numbers -/
def D : ℕ → ℕ
  | 0 => 0  -- D₀ is defined as 0 for completeness
  | 1 => 0
  | 2 => 1
  | 3 => 2
  | 4 => 9
  | (n + 5) => (n + 4) * (D (n + 4) + D (n + 3))

/-- Theorem: D₂ₙ is odd for all positive natural numbers n -/
theorem derangement_even_index_odd (n : ℕ+) : Odd (D (2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_derangement_even_index_odd_l3462_346260


namespace NUMINAMATH_CALUDE_angle_cosine_in_3d_space_l3462_346244

/-- Given a point P(x, y, z) in the first octant of 3D space, if the cosines of the angles between OP
    and the x-axis (α) and y-axis (β) are 1/3 and 1/5 respectively, then the cosine of the angle
    between OP and the z-axis (γ) is √(191)/15. -/
theorem angle_cosine_in_3d_space (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) :
  let magnitude := Real.sqrt (x^2 + y^2 + z^2)
  (x / magnitude = 1 / 3) → (y / magnitude = 1 / 5) → (z / magnitude = Real.sqrt 191 / 15) := by
  sorry

end NUMINAMATH_CALUDE_angle_cosine_in_3d_space_l3462_346244


namespace NUMINAMATH_CALUDE_average_of_seven_thirteen_and_n_l3462_346233

theorem average_of_seven_thirteen_and_n (N : ℝ) (h1 : 15 < N) (h2 : N < 25) :
  (7 + 13 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_of_seven_thirteen_and_n_l3462_346233


namespace NUMINAMATH_CALUDE_fraction_equals_decimal_l3462_346247

theorem fraction_equals_decimal : (1 : ℚ) / 4 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_decimal_l3462_346247


namespace NUMINAMATH_CALUDE_complex_number_location_l3462_346217

/-- The complex number z = 3 / (1 + 2i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_location (z : ℂ) (h : z = 3 / (1 + 2*I)) : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3462_346217


namespace NUMINAMATH_CALUDE_suji_age_is_16_l3462_346248

/-- Represents the ages of Abi, Suji, and Ravi -/
structure Ages where
  x : ℕ
  deriving Repr

def Ages.abi (a : Ages) : ℕ := 5 * a.x
def Ages.suji (a : Ages) : ℕ := 4 * a.x
def Ages.ravi (a : Ages) : ℕ := 3 * a.x

def Ages.future_abi (a : Ages) : ℕ := a.abi + 6
def Ages.future_suji (a : Ages) : ℕ := a.suji + 6
def Ages.future_ravi (a : Ages) : ℕ := a.ravi + 6

/-- The theorem stating that Suji's present age is 16 years -/
theorem suji_age_is_16 (a : Ages) : 
  (a.future_abi / a.future_suji = 13 / 11) ∧ 
  (a.future_suji / a.future_ravi = 11 / 9) → 
  a.suji = 16 := by
  sorry

#eval Ages.suji { x := 4 }

end NUMINAMATH_CALUDE_suji_age_is_16_l3462_346248


namespace NUMINAMATH_CALUDE_fraction_simplification_l3462_346258

theorem fraction_simplification :
  ((2^2010)^2 - (2^2008)^2) / ((2^2009)^2 - (2^2007)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3462_346258


namespace NUMINAMATH_CALUDE_sum_of_roots_l3462_346204

theorem sum_of_roots (a b : ℝ) (ha : a * (a - 6) = 7) (hb : b * (b - 6) = 7) (hab : a ≠ b) :
  a + b = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3462_346204
