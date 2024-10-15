import Mathlib

namespace NUMINAMATH_CALUDE_alcohol_mixture_concentration_l562_56211

-- Define the concentrations and volumes
def x_concentration : ℝ := 0.10
def y_concentration : ℝ := 0.30
def target_concentration : ℝ := 0.22
def x_volume : ℝ := 300
def y_volume : ℝ := 450

-- Theorem statement
theorem alcohol_mixture_concentration :
  (x_concentration * x_volume + y_concentration * y_volume) / (x_volume + y_volume) = target_concentration := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_concentration_l562_56211


namespace NUMINAMATH_CALUDE_equation_solution_l562_56236

theorem equation_solution : ∃ x : ℝ, 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ∧ 
  x = 31 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l562_56236


namespace NUMINAMATH_CALUDE_jakes_drink_volume_l562_56278

/-- A drink recipe with parts of different ingredients -/
structure DrinkRecipe where
  coke_parts : ℕ
  sprite_parts : ℕ
  mountain_dew_parts : ℕ

/-- Calculate the total volume of a drink given its recipe and the volume of one ingredient -/
def total_volume (recipe : DrinkRecipe) (coke_volume : ℚ) : ℚ :=
  let total_parts := recipe.coke_parts + recipe.sprite_parts + recipe.mountain_dew_parts
  let volume_per_part := coke_volume / recipe.coke_parts
  total_parts * volume_per_part

/-- Theorem stating that for the given recipe and Coke volume, the total volume is 22 ounces -/
theorem jakes_drink_volume : 
  let recipe := DrinkRecipe.mk 4 2 5
  total_volume recipe 8 = 22 := by
  sorry

end NUMINAMATH_CALUDE_jakes_drink_volume_l562_56278


namespace NUMINAMATH_CALUDE_max_value_of_sum_l562_56223

theorem max_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≤ 1 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 1 ∧
    1 / (a' + b' + 1) + 1 / (b' + c' + 1) + 1 / (c' + a' + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l562_56223


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l562_56288

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The function f(x) = x^2 - |x + a| -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 - |x + a|

/-- If f(x) = x^2 - |x + a| is an even function, then a = 0 -/
theorem even_function_implies_a_zero (a : ℝ) : IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l562_56288


namespace NUMINAMATH_CALUDE_octagon_circle_circumference_l562_56206

/-- The circumference of a circle containing an inscribed regular octagon -/
theorem octagon_circle_circumference (side_length : ℝ) (h : side_length = 5) :
  ∃ (circumference : ℝ), circumference = (5 * π) / Real.sin (22.5 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_octagon_circle_circumference_l562_56206


namespace NUMINAMATH_CALUDE_equation_result_is_55_l562_56242

/-- The result of 4 times a number plus 7 times the same number, given the number is 5.0 -/
def equation_result (n : ℝ) : ℝ := 4 * n + 7 * n

/-- Theorem stating that the result of the equation is 55.0 when the number is 5.0 -/
theorem equation_result_is_55 : equation_result 5.0 = 55.0 := by
  sorry

end NUMINAMATH_CALUDE_equation_result_is_55_l562_56242


namespace NUMINAMATH_CALUDE_valid_intersection_numbers_l562_56294

/-- A circle with arcs that intersect each other. -/
structure CircleWithArcs where
  num_arcs : ℕ
  intersections_per_arc : ℕ

/-- Predicate to check if a number is not a multiple of 8. -/
def not_multiple_of_eight (n : ℕ) : Prop :=
  n % 8 ≠ 0

/-- Theorem stating the conditions for valid intersection numbers in a circle with 100 arcs. -/
theorem valid_intersection_numbers (circle : CircleWithArcs) :
    circle.num_arcs = 100 →
    1 ≤ circle.intersections_per_arc ∧
    circle.intersections_per_arc ≤ 99 ∧
    not_multiple_of_eight (circle.intersections_per_arc + 1) :=
by sorry

end NUMINAMATH_CALUDE_valid_intersection_numbers_l562_56294


namespace NUMINAMATH_CALUDE_team_b_mean_tasks_l562_56295

/-- Represents the office with two teams -/
structure Office :=
  (total_members : ℕ)
  (team_a_members : ℕ)
  (team_b_members : ℕ)
  (team_a_mean_tasks : ℝ)
  (team_b_mean_tasks : ℝ)

/-- The conditions of the office as described in the problem -/
def office_conditions (o : Office) : Prop :=
  o.total_members = 260 ∧
  o.team_a_members = (13 * o.team_b_members) / 10 ∧
  o.team_a_mean_tasks = 80 ∧
  o.team_b_mean_tasks = (6 * o.team_a_mean_tasks) / 5

/-- The theorem stating that under the given conditions, Team B's mean tasks is 96 -/
theorem team_b_mean_tasks (o : Office) (h : office_conditions o) : 
  o.team_b_mean_tasks = 96 := by
  sorry


end NUMINAMATH_CALUDE_team_b_mean_tasks_l562_56295


namespace NUMINAMATH_CALUDE_adult_meal_cost_l562_56246

/-- Proves that the cost of each adult meal is $3 given the specified conditions -/
theorem adult_meal_cost (total_people : Nat) (kids : Nat) (total_cost : Nat) :
  total_people = 12 →
  kids = 7 →
  total_cost = 15 →
  (total_cost / (total_people - kids) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_adult_meal_cost_l562_56246


namespace NUMINAMATH_CALUDE_sum_of_digits_seven_to_seventeen_l562_56221

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_seven_to_seventeen (h : ℕ) :
  h = 7^17 →
  tens_digit h + ones_digit h = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_seven_to_seventeen_l562_56221


namespace NUMINAMATH_CALUDE_prime_sum_problem_l562_56228

theorem prime_sum_problem (a b c : ℕ) 
  (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (hab : a + b = 49) (hbc : b + c = 60) : c = 13 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l562_56228


namespace NUMINAMATH_CALUDE_white_washing_cost_per_square_foot_l562_56205

/-- Calculates the cost per square foot for white washing a room --/
theorem white_washing_cost_per_square_foot
  (room_length room_width room_height : ℝ)
  (door_length door_width : ℝ)
  (window_length window_width : ℝ)
  (num_windows : ℕ)
  (total_cost : ℝ)
  (h_room_length : room_length = 25)
  (h_room_width : room_width = 15)
  (h_room_height : room_height = 12)
  (h_door_length : door_length = 6)
  (h_door_width : door_width = 3)
  (h_window_length : window_length = 4)
  (h_window_width : window_width = 3)
  (h_num_windows : num_windows = 3)
  (h_total_cost : total_cost = 8154) :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_length * door_width
  let window_area := num_windows * (window_length * window_width)
  let net_area := wall_area - door_area - window_area
  let cost_per_square_foot := total_cost / net_area
  cost_per_square_foot = 9 :=
sorry

end NUMINAMATH_CALUDE_white_washing_cost_per_square_foot_l562_56205


namespace NUMINAMATH_CALUDE_second_element_of_sequence_l562_56224

theorem second_element_of_sequence (n : ℕ) : 
  n > 1 → (n * (n + 1)) / 2 = 78 → 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_element_of_sequence_l562_56224


namespace NUMINAMATH_CALUDE_cube_volume_l562_56249

theorem cube_volume (edge : ℝ) (h : edge = 7) : edge^3 = 343 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l562_56249


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l562_56263

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l562_56263


namespace NUMINAMATH_CALUDE_officer_election_ways_l562_56279

def club_size : ℕ := 12
def num_officers : ℕ := 5

theorem officer_election_ways :
  (club_size * (club_size - 1) * (club_size - 2) * (club_size - 3) * (club_size - 4) : ℕ) = 95040 := by
  sorry

end NUMINAMATH_CALUDE_officer_election_ways_l562_56279


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l562_56217

/-- Proves that given a train of length 360 meters traveling at 36 km/hour,
    if it takes 50 seconds to pass a bridge, then the length of the bridge is 140 meters. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  train_length = 360 →
  train_speed_kmh = 36 →
  time_to_pass = 50 →
  (train_speed_kmh * 1000 / 3600) * time_to_pass - train_length = 140 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l562_56217


namespace NUMINAMATH_CALUDE_two_stretches_to_similar_triangle_l562_56266

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

-- Define a stretch transformation
structure Stretch where
  center : Point2D
  coefficient : ℝ

-- Define similarity between triangles
def Similar (t1 t2 : Triangle) : Prop := sorry

-- Define the application of a stretch to a triangle
def ApplyStretch (s : Stretch) (t : Triangle) : Triangle := sorry

-- Theorem statement
theorem two_stretches_to_similar_triangle 
  (ABC : Triangle) (DEF : Triangle) (h : DEF.A.x = DEF.A.y ∧ DEF.B.x = DEF.B.y) :
  ∃ (S1 S2 : Stretch), Similar (ApplyStretch S2 (ApplyStretch S1 ABC)) DEF := by
  sorry

end NUMINAMATH_CALUDE_two_stretches_to_similar_triangle_l562_56266


namespace NUMINAMATH_CALUDE_max_x_over_y_l562_56210

theorem max_x_over_y (x y a b : ℝ) (h1 : x ≥ y) (h2 : y > 0)
  (h3 : 0 ≤ a) (h4 : a ≤ x) (h5 : 0 ≤ b) (h6 : b ≤ y)
  (h7 : (x - a)^2 + (y - b)^2 = x^2 + b^2)
  (h8 : x^2 + b^2 = y^2 + a^2) :
  ∃ (x' y' : ℝ), x' ≥ y' ∧ y' > 0 ∧
  ∃ (a' b' : ℝ), 0 ≤ a' ∧ a' ≤ x' ∧ 0 ≤ b' ∧ b' ≤ y' ∧
  (x' - a')^2 + (y' - b')^2 = x'^2 + b'^2 ∧ x'^2 + b'^2 = y'^2 + a'^2 ∧
  x' / y' = 2 * Real.sqrt 3 / 3 ∧
  ∀ (x'' y'' : ℝ), x'' ≥ y'' → y'' > 0 →
  ∃ (a'' b'' : ℝ), 0 ≤ a'' ∧ a'' ≤ x'' ∧ 0 ≤ b'' ∧ b'' ≤ y'' ∧
  (x'' - a'')^2 + (y'' - b'')^2 = x''^2 + b''^2 ∧ x''^2 + b''^2 = y''^2 + a''^2 →
  x'' / y'' ≤ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_x_over_y_l562_56210


namespace NUMINAMATH_CALUDE_remaining_fruits_count_l562_56207

/-- Represents the number of fruits on each tree type -/
structure FruitTrees :=
  (apples : ℕ)
  (plums : ℕ)
  (pears : ℕ)
  (cherries : ℕ)

/-- Represents the fraction of fruits picked from each tree -/
structure PickedFractions :=
  (apples : ℚ)
  (plums : ℚ)
  (pears : ℚ)
  (cherries : ℚ)

def original_fruits : FruitTrees :=
  { apples := 180
  , plums := 60
  , pears := 120
  , cherries := 720 }

def picked_fractions : PickedFractions :=
  { apples := 3/5
  , plums := 2/3
  , pears := 3/4
  , cherries := 7/10 }

theorem remaining_fruits_count 
  (orig : FruitTrees) 
  (picked : PickedFractions) 
  (h1 : orig.apples = 3 * orig.plums)
  (h2 : orig.pears = 2 * orig.plums)
  (h3 : orig.cherries = 4 * orig.apples)
  (h4 : orig = original_fruits)
  (h5 : picked = picked_fractions) :
  (orig.apples - (picked.apples * orig.apples).num) +
  (orig.plums - (picked.plums * orig.plums).num) +
  (orig.pears - (picked.pears * orig.pears).num) +
  (orig.cherries - (picked.cherries * orig.cherries).num) = 338 :=
by sorry

end NUMINAMATH_CALUDE_remaining_fruits_count_l562_56207


namespace NUMINAMATH_CALUDE_sequence_properties_l562_56230

def b (n : ℕ) : ℝ := 2 * n - 1

def c (n : ℕ) : ℝ := 3 * n - 2

def a (n : ℕ) (x y : ℝ) : ℝ := x * b n + y * c n

theorem sequence_properties
  (x y : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : x + y = 1) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) x y - a n x y = d) ∧
  (∃ y' : ℝ, ∀ n : ℕ, a n x y' = (b n + c n) / 2) ∧
  (∀ n : ℕ, n ≥ 2 → b n < a n x y ∧ a n x y < c n) ∧
  (∀ n : ℕ, n ≥ 2 → a n x y + b n > c n ∧ a n x y + c n > b n ∧ b n + c n > a n x y) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l562_56230


namespace NUMINAMATH_CALUDE_inequality_comparison_l562_56254

theorem inequality_comparison (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c ≠ 0) :
  (∀ c, a + c > b + c) ∧
  (∀ c, a - 3*c > b - 3*c) ∧
  (¬∀ c, a*c > b*c) ∧
  (∀ c, a/c^2 > b/c^2) ∧
  (∀ c, a*c^3 > b*c^3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_comparison_l562_56254


namespace NUMINAMATH_CALUDE_sphere_equal_volume_surface_area_l562_56209

theorem sphere_equal_volume_surface_area (r k S : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = S ∧ 
  4 * Real.pi * r^2 = S ∧ 
  k * r = S → 
  r = 3 ∧ k = 12 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sphere_equal_volume_surface_area_l562_56209


namespace NUMINAMATH_CALUDE_flagpole_height_l562_56237

/-- Given a 3-meter pole with a 1.2-meter shadow and a flagpole with a 4.8-meter shadow,
    the height of the flagpole is 12 meters. -/
theorem flagpole_height
  (pole_height : Real)
  (pole_shadow : Real)
  (flagpole_shadow : Real)
  (h_pole_height : pole_height = 3)
  (h_pole_shadow : pole_shadow = 1.2)
  (h_flagpole_shadow : flagpole_shadow = 4.8) :
  pole_height / pole_shadow = 12 / flagpole_shadow := by
  sorry

end NUMINAMATH_CALUDE_flagpole_height_l562_56237


namespace NUMINAMATH_CALUDE_initial_red_marbles_l562_56292

theorem initial_red_marbles (r g : ℕ) : 
  r * 3 = g * 5 → 
  (r - 20) * 5 = (g + 40) * 1 → 
  r = 317 := by
sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l562_56292


namespace NUMINAMATH_CALUDE_circle_intersection_range_l562_56247

theorem circle_intersection_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x - 2*Real.sqrt 3*y - m = 0 ∧ x^2 + y^2 = 1) ↔ 
  -3 ≤ m ∧ m ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l562_56247


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l562_56285

/-- Given two lines l₁ and l₂ with equations 3x + my - 1 = 0 and (m+2)x - (m-2)y + 2 = 0 respectively,
    if l₁ is parallel to l₂, then m = -6 or m = 1. -/
theorem parallel_lines_m_values (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | 3 * x + m * y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (m + 2) * x - (m - 2) * y + 2 = 0}
  (∀ (a b c d : ℝ), a * (m + 2) = 3 * c ∧ b * (m - 2) = -m * d → (a, b) = (c, d)) →
  m = -6 ∨ m = 1 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_m_values_l562_56285


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l562_56281

/-- The minimum bailing rate problem -/
theorem minimum_bailing_rate
  (distance : ℝ)
  (water_entry_rate : ℝ)
  (max_water_capacity : ℝ)
  (rowing_speed : ℝ)
  (h1 : distance = 2)
  (h2 : water_entry_rate = 8)
  (h3 : max_water_capacity = 50)
  (h4 : rowing_speed = 2)
  : ∃ (bailing_rate : ℝ),
    bailing_rate = 8 ∧
    (∀ r : ℝ, r < 8 →
      (distance / rowing_speed) * (water_entry_rate - r) > max_water_capacity) ∧
    (distance / rowing_speed) * (water_entry_rate - bailing_rate) ≤ max_water_capacity :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l562_56281


namespace NUMINAMATH_CALUDE_intersection_A_B_l562_56297

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l562_56297


namespace NUMINAMATH_CALUDE_regular_polygon_is_pentagon_with_perimeter_125_l562_56214

/-- A regular polygon where the length of a side is 25 when the perimeter is divided by 5 -/
structure RegularPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  h1 : perimeter = sides * side_length
  h2 : perimeter / 5 = side_length
  h3 : side_length = 25

theorem regular_polygon_is_pentagon_with_perimeter_125 (p : RegularPolygon) :
  p.sides = 5 ∧ p.perimeter = 125 := by
  sorry

#check regular_polygon_is_pentagon_with_perimeter_125

end NUMINAMATH_CALUDE_regular_polygon_is_pentagon_with_perimeter_125_l562_56214


namespace NUMINAMATH_CALUDE_intercept_triangle_area_zero_l562_56202

/-- The cubic function f(x) = x³ - x --/
def f (x : ℝ) : ℝ := x^3 - x

/-- The set of x-intercepts of the curve y = x³ - x --/
def x_intercepts : Set ℝ := {x : ℝ | f x = 0}

/-- The y-intercept of the curve y = x³ - x --/
def y_intercept : ℝ × ℝ := (0, f 0)

/-- The area of the triangle formed by the intercepts of the curve y = x³ - x --/
def triangle_area : ℝ := sorry

/-- Theorem: The area of the triangle formed by the intercepts of y = x³ - x is 0 --/
theorem intercept_triangle_area_zero : triangle_area = 0 := by sorry

end NUMINAMATH_CALUDE_intercept_triangle_area_zero_l562_56202


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l562_56299

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2003 : ℝ) ^ x + (2004 : ℝ) ^ x = (2005 : ℝ) ^ x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l562_56299


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l562_56274

/-- Given two circles in the plane, this theorem proves that the equation of the line
    passing through their intersection points has a specific form. -/
theorem intersection_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 = 10) ∧ ((x-1)^2 + (y-3)^2 = 10) → x + 3*y - 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l562_56274


namespace NUMINAMATH_CALUDE_local_minimum_at_one_l562_56245

/-- The function f(x) = ax³ - 2x² + a²x has a local minimum at x=1 if and only if a = 1 -/
theorem local_minimum_at_one (a : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), 
    a*x^3 - 2*x^2 + a^2*x ≥ a*1^3 - 2*1^2 + a^2*1) ↔ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_local_minimum_at_one_l562_56245


namespace NUMINAMATH_CALUDE_car_distance_proof_l562_56233

theorem car_distance_proof (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 30 →
  (∃ (initial_speed : ℝ), 
    initial_speed * initial_time = new_speed * (initial_time * (2/3))) →
  (∃ (distance : ℝ), distance = 120) :=
by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l562_56233


namespace NUMINAMATH_CALUDE_odd_prime_sum_iff_floor_sum_odd_l562_56227

theorem odd_prime_sum_iff_floor_sum_odd (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (a b : ℕ) (ha : 0 < a ∧ a < p) (hb : 0 < b ∧ b < p) :
  a + b = p ↔
  ∀ n : ℕ, 0 < n → n < p →
    ∃ k : ℕ, Int.floor ((2 * a * n : ℚ) / p) + Int.floor ((2 * b * n : ℚ) / p) = 2 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_prime_sum_iff_floor_sum_odd_l562_56227


namespace NUMINAMATH_CALUDE_geometric_sum_proof_l562_56201

/-- The sum of a geometric sequence with first term 9, common ratio 3, and 7 terms -/
def geometric_sum : ℕ := 9827

/-- The first term of the geometric sequence -/
def a : ℕ := 9

/-- The common ratio of the geometric sequence -/
def r : ℕ := 3

/-- The number of terms in the geometric sequence -/
def n : ℕ := 7

/-- Theorem stating that the sum of the geometric sequence equals 9827 -/
theorem geometric_sum_proof : 
  a * (r^n - 1) / (r - 1) = geometric_sum :=
sorry

end NUMINAMATH_CALUDE_geometric_sum_proof_l562_56201


namespace NUMINAMATH_CALUDE_divisibility_implication_l562_56273

theorem divisibility_implication (k n : ℤ) :
  (13 ∣ (k + 4*n)) → (13 ∣ (10*k + n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l562_56273


namespace NUMINAMATH_CALUDE_lines_perpendicular_when_a_is_neg_six_l562_56275

/-- Given two lines l₁ and l₂ defined by their equations, prove that they are perpendicular when a = -6 -/
theorem lines_perpendicular_when_a_is_neg_six (a : ℝ) :
  a = -6 →
  let l₁ := {(x, y) : ℝ × ℝ | a * x + (1 - a) * y - 3 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (a - 1) * x + 2 * (a + 3) * y - 2 = 0}
  let m₁ := a / (1 - a)
  let m₂ := (a - 1) / (2 * (a + 3))
  m₁ * m₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_lines_perpendicular_when_a_is_neg_six_l562_56275


namespace NUMINAMATH_CALUDE_calculations_proof_l562_56239

-- Define the calculations
def calc1 : ℝ := 70.8 - 1.25 - 1.75
def calc2 : ℝ := (8 + 0.8) * 1.25
def calc3 : ℝ := 125 * 0.48
def calc4 : ℝ := 6.7 * (9.3 * (6.2 + 1.7))

-- Theorem to prove the calculations
theorem calculations_proof :
  calc1 = 67.8 ∧
  calc2 = 11 ∧
  calc3 = 600 ∧
  calc4 = 554.559 := by
  sorry

#eval calc1
#eval calc2
#eval calc3
#eval calc4

end NUMINAMATH_CALUDE_calculations_proof_l562_56239


namespace NUMINAMATH_CALUDE_grapes_count_l562_56241

/-- The number of grapes in Rob's bowl -/
def rob_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allie_grapes : ℕ := rob_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyn_grapes : ℕ := allie_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := rob_grapes + allie_grapes + allyn_grapes

theorem grapes_count : total_grapes = 83 := by
  sorry

end NUMINAMATH_CALUDE_grapes_count_l562_56241


namespace NUMINAMATH_CALUDE_line_equation_through_points_line_equation_specific_points_l562_56264

/-- The equation of a line passing through two points -/
theorem line_equation_through_points (x₁ y₁ x₂ y₂ : ℝ) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (x₂ ≠ x₁) →
  (∀ x y : ℝ, y = m * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)) :=
by sorry

/-- The equation of the line passing through (0, -5) and (1, 0) is y = 5x - 5 -/
theorem line_equation_specific_points :
  ∀ x y : ℝ, y = 5 * x - 5 ↔ (x = 0 ∧ y = -5) ∨ (x = 1 ∧ y = 0) ∨ (y - (-5)) * (1 - 0) = (x - 0) * (0 - (-5)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_line_equation_specific_points_l562_56264


namespace NUMINAMATH_CALUDE_evaluate_expression_l562_56213

theorem evaluate_expression : -(14 / 2 * 9 - 60 + 3 * 9) = -30 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l562_56213


namespace NUMINAMATH_CALUDE_class_savings_theorem_l562_56255

/-- Calculates the total amount saved by a class for a field trip over a given period. -/
def total_savings (num_students : ℕ) (contribution_per_student : ℕ) (weeks_per_month : ℕ) (num_months : ℕ) : ℕ :=
  num_students * contribution_per_student * weeks_per_month * num_months

/-- Theorem stating that a class of 30 students contributing $2 each week will save $480 in 2 months. -/
theorem class_savings_theorem :
  total_savings 30 2 4 2 = 480 := by
  sorry

#eval total_savings 30 2 4 2

end NUMINAMATH_CALUDE_class_savings_theorem_l562_56255


namespace NUMINAMATH_CALUDE_solution_value_l562_56240

theorem solution_value (a b t : ℝ) : 
  a^2 + 4*b = t^2 →
  a^2 - b^2 = 4 →
  b > 0 →
  b = t - 2 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l562_56240


namespace NUMINAMATH_CALUDE_no_perfect_square_solution_l562_56284

theorem no_perfect_square_solution (n : ℕ) : ¬∃ (m : ℕ), n^5 - 5*n^3 + 4*n + 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_solution_l562_56284


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l562_56261

theorem triangle_angle_sum (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- Ensures positive angles
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A = 25 →  -- Given angle A
  B = 55 →  -- Given angle B
  C = 100 :=  -- Conclusion: angle C
by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l562_56261


namespace NUMINAMATH_CALUDE_age_ratio_sandy_molly_l562_56265

/-- Given that Sandy is 42 years old and Molly is 12 years older than Sandy,
    prove that the ratio of their ages is 7:9. -/
theorem age_ratio_sandy_molly :
  let sandy_age : ℕ := 42
  let molly_age : ℕ := sandy_age + 12
  (sandy_age : ℚ) / molly_age = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_sandy_molly_l562_56265


namespace NUMINAMATH_CALUDE_sin_1320_degrees_l562_56272

theorem sin_1320_degrees : Real.sin (1320 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1320_degrees_l562_56272


namespace NUMINAMATH_CALUDE_only_seven_has_integer_solution_solutions_for_seven_l562_56219

/-- The product of terms (1 + 1/(x+k)) from k = 0 to n -/
def productTerm (x : ℤ) (n : ℕ) : ℚ :=
  (List.range (n + 1)).foldl (fun acc k => acc * (1 + 1 / (x + k))) 1

/-- The main theorem stating that 7 is the only positive integer solution -/
theorem only_seven_has_integer_solution :
  ∀ a : ℕ+, (∃ x : ℤ, productTerm x a = a - x) ↔ a = 7 := by
  sorry

/-- Verification of the two integer solutions for a = 7 -/
theorem solutions_for_seven :
  productTerm 2 7 = 5 ∧ productTerm 4 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_only_seven_has_integer_solution_solutions_for_seven_l562_56219


namespace NUMINAMATH_CALUDE_irrigation_flux_theorem_l562_56258

-- Define the irrigation system
structure IrrigationSystem where
  channels : List Char
  entry : Char
  exit : Char
  flux : Char → Char → ℝ

-- Define the properties of the irrigation system
def has_constant_flux_sum (sys : IrrigationSystem) : Prop :=
  ∀ (p q r : Char), p ∈ sys.channels → q ∈ sys.channels → r ∈ sys.channels →
    sys.flux p q + sys.flux q r = sys.flux p r

-- Define the theorem
theorem irrigation_flux_theorem (sys : IrrigationSystem) 
  (h_channels : sys.channels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
  (h_entry : sys.entry = 'A')
  (h_exit : sys.exit = 'E')
  (h_constant_flux : has_constant_flux_sum sys)
  (h_flux_bc : sys.flux 'B' 'C' = q₀) :
  sys.flux 'A' 'B' = 2 * q₀ ∧ 
  sys.flux 'A' 'H' = 3/2 * q₀ ∧ 
  sys.flux 'A' 'B' + sys.flux 'A' 'H' = 7/2 * q₀ := by
  sorry

end NUMINAMATH_CALUDE_irrigation_flux_theorem_l562_56258


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l562_56244

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x^2 + y^2 = 0 → x * y = 0) ∧
  (∃ x y, x * y = 0 ∧ x^2 + y^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l562_56244


namespace NUMINAMATH_CALUDE_intersection_and_union_range_of_m_l562_56282

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2*m}

-- Theorem for part (Ⅰ)
theorem intersection_and_union :
  (A ∩ B = {x | 2 ≤ x ∧ x < 5}) ∧
  ((Aᶜ ∪ B) = {x | -3 < x ∧ x < 5}) := by sorry

-- Theorem for part (Ⅱ)
theorem range_of_m (m : ℝ) :
  (B ∩ C m = C m) → (m < -1 ∨ (2 < m ∧ m < 5/2)) := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_range_of_m_l562_56282


namespace NUMINAMATH_CALUDE_gift_packaging_combinations_l562_56256

/-- The number of varieties of packaging paper -/
def paper_varieties : ℕ := 10

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 4

/-- The number of types of decorative stickers -/
def sticker_types : ℕ := 5

/-- The total number of gift packaging combinations -/
def total_combinations : ℕ := paper_varieties * ribbon_colors * sticker_types

theorem gift_packaging_combinations :
  total_combinations = 200 := by sorry

end NUMINAMATH_CALUDE_gift_packaging_combinations_l562_56256


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l562_56276

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l562_56276


namespace NUMINAMATH_CALUDE_smallest_four_digit_numbers_l562_56293

def is_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1

theorem smallest_four_digit_numbers :
  let valid_numbers := [1021, 1081, 1141, 1201]
  (∀ n ∈ valid_numbers, is_valid n) ∧
  (∀ m, is_valid m → m ≥ 1021) ∧
  (∀ n ∈ valid_numbers, ∀ m, is_valid m ∧ m < n → m ∈ valid_numbers) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_numbers_l562_56293


namespace NUMINAMATH_CALUDE_possible_m_values_l562_56243

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem possible_m_values (m : ℝ) : A ∪ B m = A → m = 0 ∨ m = -1/3 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_possible_m_values_l562_56243


namespace NUMINAMATH_CALUDE_constant_product_of_distances_l562_56287

/-- Hyperbola type representing x^2 - y^2/4 = 1 -/
structure Hyperbola where
  x : ℝ
  y : ℝ
  eq : x^2 - y^2/4 = 1

/-- Line type representing a line passing through a point on the hyperbola -/
structure Line (h : Hyperbola) where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  passes_through : m * h.x + b = h.y

/-- Intersection point of a line with an asymptote -/
structure AsymptoteIntersection (h : Hyperbola) (l : Line h) where
  x : ℝ
  y : ℝ
  on_asymptote : y = 2*x ∨ y = -2*x
  on_line : y = l.m * x + l.b

/-- Theorem: Product of distances from origin to asymptote intersections is constant -/
theorem constant_product_of_distances (h : Hyperbola) (l : Line h) 
  (a b : AsymptoteIntersection h l) 
  (midpoint : h.x = (a.x + b.x)/2 ∧ h.y = (a.y + b.y)/2) :
  (a.x^2 + a.y^2) * (b.x^2 + b.y^2) = 25 := by sorry

end NUMINAMATH_CALUDE_constant_product_of_distances_l562_56287


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l562_56222

theorem problems_left_to_grade 
  (problems_per_worksheet : ℕ) 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (h1 : problems_per_worksheet = 4)
  (h2 : total_worksheets = 9)
  (h3 : graded_worksheets = 5) :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 16 :=
by sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l562_56222


namespace NUMINAMATH_CALUDE_number_problem_l562_56296

theorem number_problem (x : ℕ) (h1 : x + 3927 = 13800) : x = 9873 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l562_56296


namespace NUMINAMATH_CALUDE_log_of_expression_l562_56203

theorem log_of_expression (x : ℝ) : 
  x = 125 * Real.rpow 25 (1/3) * Real.sqrt 25 → 
  Real.log x / Real.log 5 = 14/3 := by
sorry

end NUMINAMATH_CALUDE_log_of_expression_l562_56203


namespace NUMINAMATH_CALUDE_total_canoes_by_april_l562_56286

def canoes_built (month : Nat) : Nat :=
  match month with
  | 0 => 4  -- February
  | n + 1 => 3 * canoes_built n

theorem total_canoes_by_april : 
  (canoes_built 0) + (canoes_built 1) + (canoes_built 2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_canoes_by_april_l562_56286


namespace NUMINAMATH_CALUDE_parabolas_equal_if_equal_segments_l562_56229

/-- Two non-parallel lines in the plane -/
structure NonParallelLines where
  l₁ : ℝ → ℝ
  l₂ : ℝ → ℝ
  not_parallel : l₁ ≠ l₂

/-- A parabola of the form f(x) = x² + px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- The length of the segment cut by a parabola on a line -/
def segment_length (para : Parabola) (line : ℝ → ℝ) : ℝ := sorry

/-- Two parabolas cut equal segments on two non-parallel lines -/
def equal_segments (f₁ f₂ : Parabola) (lines : NonParallelLines) : Prop :=
  segment_length f₁ lines.l₁ = segment_length f₂ lines.l₁ ∧
  segment_length f₁ lines.l₂ = segment_length f₂ lines.l₂

/-- Main theorem: If two parabolas cut equal segments on two non-parallel lines, 
    then the parabolas are identical -/
theorem parabolas_equal_if_equal_segments (f₁ f₂ : Parabola) (lines : NonParallelLines) :
  equal_segments f₁ f₂ lines → f₁ = f₂ := by sorry

end NUMINAMATH_CALUDE_parabolas_equal_if_equal_segments_l562_56229


namespace NUMINAMATH_CALUDE_triangle_isosceles_from_equation_l562_56271

/-- A triangle with sides a, b, and c is isosceles if it satisfies the equation a^2 - bc = a(b - c) -/
theorem triangle_isosceles_from_equation (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_eq : a^2 - b*c = a*(b - c)) : 
  a = b ∨ b = c ∨ c = a := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_from_equation_l562_56271


namespace NUMINAMATH_CALUDE_right_triangle_area_l562_56268

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 90 →
  a^2 + b^2 + c^2 = 3362 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 180 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l562_56268


namespace NUMINAMATH_CALUDE_cube_root_of_256_l562_56269

theorem cube_root_of_256 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 256) : x = 4 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_256_l562_56269


namespace NUMINAMATH_CALUDE_permutations_formula_l562_56225

def factorial (n : ℕ) : ℕ := Nat.factorial n

def permutations_with_repetition (n : ℕ) (k : List ℕ) : ℚ :=
  (factorial n) / (k.map factorial).prod

theorem permutations_formula (n : ℕ) (k : List ℕ) 
  (h : k.sum = n) : 
  permutations_with_repetition n k = 
    (factorial n) / (k.map factorial).prod := by
  sorry

#eval permutations_with_repetition 5 [5]  -- for "замок"
#eval permutations_with_repetition 5 [1, 2, 2]  -- for "ротор"
#eval permutations_with_repetition 5 [3, 2]  -- for "топор"
#eval permutations_with_repetition 7 [1, 2, 2, 3]  -- for "колокол"

end NUMINAMATH_CALUDE_permutations_formula_l562_56225


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_221_l562_56259

theorem inverse_of_3_mod_221 : ∃ x : ℕ, x < 221 ∧ (3 * x) % 221 = 1 :=
by
  use 74
  sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_221_l562_56259


namespace NUMINAMATH_CALUDE_shade_in_three_folds_l562_56212

/-- Represents a square grid -/
structure Grid :=
  (size : Nat)
  (shaded : Set (Nat × Nat))

/-- Represents a fold along a grid line -/
inductive Fold
  | Vertical (col : Nat)
  | Horizontal (row : Nat)

/-- Apply a fold to a grid -/
def applyFold (g : Grid) (f : Fold) : Grid :=
  sorry

/-- Check if the entire grid is shaded -/
def isFullyShaded (g : Grid) : Prop :=
  sorry

/-- Theorem stating that it's possible to shade the entire grid in 3 or fewer folds -/
theorem shade_in_three_folds (g : Grid) :
  ∃ (folds : List Fold), folds.length ≤ 3 ∧ isFullyShaded (folds.foldl applyFold g) :=
sorry

end NUMINAMATH_CALUDE_shade_in_three_folds_l562_56212


namespace NUMINAMATH_CALUDE_sum_of_fractions_l562_56216

theorem sum_of_fractions : (3 / 30) + (4 / 40) + (5 / 50) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l562_56216


namespace NUMINAMATH_CALUDE_square_perimeter_l562_56260

theorem square_perimeter (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 125)
  (h2 : rectangle_width = 64)
  (h3 : rectangle_length > 0)
  (h4 : rectangle_width > 0) :
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  square_side * 4 = 800 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_l562_56260


namespace NUMINAMATH_CALUDE_least_cans_required_l562_56208

theorem least_cans_required (a b c d e f g h : ℕ+) : 
  a = 139 → b = 223 → c = 179 → d = 199 → e = 173 → f = 211 → g = 131 → h = 257 →
  (∃ (x : ℕ+), x = a + b + c + d + e + f + g + h ∧ 
   x = Nat.gcd a (Nat.gcd b (Nat.gcd c (Nat.gcd d (Nat.gcd e (Nat.gcd f (Nat.gcd g h))))))) :=
by sorry

end NUMINAMATH_CALUDE_least_cans_required_l562_56208


namespace NUMINAMATH_CALUDE_bobby_pancakes_left_l562_56267

/-- The number of pancakes Bobby has left after making and serving breakfast -/
def pancakes_left (standard_batch : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) (friends_ate : ℕ) : ℕ :=
  let total_made := standard_batch + 2 * standard_batch + standard_batch
  let total_eaten := bobby_ate + dog_ate + friends_ate
  total_made - total_eaten

/-- Theorem stating that Bobby has 50 pancakes left -/
theorem bobby_pancakes_left : 
  pancakes_left 21 5 7 22 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bobby_pancakes_left_l562_56267


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l562_56298

theorem max_value_sum_of_roots (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -2/3)
  (z_ge : z ≥ -2) :
  (∀ a b c : ℝ, a + b + c = 3 → a ≥ -1 → b ≥ -2/3 → c ≥ -2 →
    Real.sqrt (3*a + 3) + Real.sqrt (3*b + 2) + Real.sqrt (3*c + 6) ≤ 
    Real.sqrt (3*x + 3) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 6)) ∧
  Real.sqrt (3*x + 3) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 6) = 2 * Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l562_56298


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l562_56200

theorem triangle_trigonometric_identities
  (α β γ : Real) (p r R : Real)
  (h_triangle : α + β + γ = Real.pi)
  (h_positive : 0 < p ∧ 0 < r ∧ 0 < R)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_inradius : r = area / p)
  (h_circumradius : R = (a * b * c) / (4 * area)) :
  (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = (p^2 - r^2 - 4*r*R) / (2*R^2) ∧
  4 * R^2 * Real.cos α * Real.cos β * Real.cos γ = p^2 - (2*R + r)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l562_56200


namespace NUMINAMATH_CALUDE_proper_subsets_of_B_l562_56218

-- Define the sets A and B
def A (b : ℝ) : Set ℝ := {x | x^2 + (b+2)*x + b + 1 = 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

-- State the theorem
theorem proper_subsets_of_B (b : ℝ) (a : ℝ) (h : A b = {a}) :
  {s : Set ℝ | s ⊂ B a b ∧ s ≠ B a b} = {∅, {1}, {0}} := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_of_B_l562_56218


namespace NUMINAMATH_CALUDE_light_reflection_l562_56280

-- Define the incident light ray
def incident_ray (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the reflection line
def reflection_line (x y : ℝ) : Prop := y = x

-- Define the reflected light ray
def reflected_ray (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Theorem statement
theorem light_reflection 
  (x y : ℝ) 
  (h_incident : incident_ray x y) 
  (h_reflection : reflection_line x y) : 
  reflected_ray x y :=
sorry

end NUMINAMATH_CALUDE_light_reflection_l562_56280


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l562_56238

theorem consecutive_integers_average (c : ℕ) (d : ℚ) : 
  (c > 0) →
  (d = (2 * c + (2 * c + 1) + (2 * c + 2) + (2 * c + 3) + (2 * c + 4) + (2 * c + 5) + (2 * c + 6)) / 7) →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = 2 * c + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l562_56238


namespace NUMINAMATH_CALUDE_valid_parameterization_l562_56231

/-- A structure representing a vector parameterization of a line -/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  a : ℝ
  b : ℝ

/-- Checks if a given vector parameterization is valid for the line y = 2x - 4 -/
def isValidParam (p : VectorParam) : Prop :=
  p.y₀ = 2 * p.x₀ - 4 ∧ ∃ k : ℝ, p.a = k * 1 ∧ p.b = k * 2

/-- The theorem stating the conditions for a valid vector parameterization -/
theorem valid_parameterization (p : VectorParam) : 
  isValidParam p ↔ 
  (∀ t : ℝ, (p.x₀ + t * p.a, p.y₀ + t * p.b) ∈ {(x, y) : ℝ × ℝ | y = 2 * x - 4}) :=
sorry

end NUMINAMATH_CALUDE_valid_parameterization_l562_56231


namespace NUMINAMATH_CALUDE_min_value_of_g_l562_56235

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1

-- Define the maximum value M(a) on the interval [1,3]
noncomputable def M (a : ℝ) : ℝ := 
  ⨆ (x : ℝ) (h : x ∈ Set.Icc 1 3), f a x

-- Define the minimum value N(a) on the interval [1,3]
noncomputable def N (a : ℝ) : ℝ := 
  ⨅ (x : ℝ) (h : x ∈ Set.Icc 1 3), f a x

-- Define g(a) as M(a) - N(a)
noncomputable def g (a : ℝ) : ℝ := M a - N a

-- State the theorem
theorem min_value_of_g :
  ∀ a : ℝ, a ∈ Set.Icc (1/3) 1 → 
  ∃ min_g : ℝ, min_g = (⨅ (a : ℝ) (h : a ∈ Set.Icc (1/3) 1), g a) ∧ min_g = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_g_l562_56235


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_one_l562_56226

/-- Two lines in the xy-plane -/
structure ParallelLines where
  /-- The first line equation: x + 2y - 4 = 0 -/
  line1 : ℝ → ℝ → Prop := fun x y => x + 2*y - 4 = 0
  /-- The second line equation: ax + 2y + 6 = 0 -/
  line2 : ℝ → ℝ → ℝ → Prop := fun a x y => a*x + 2*y + 6 = 0
  /-- The lines are parallel -/
  parallel : ∀ (a : ℝ), (∀ x y, line1 x y ↔ ∃ k, line2 a (x + k) (y + k))

/-- If two lines are parallel as defined, then a = 1 -/
theorem parallel_lines_a_equals_one (pl : ParallelLines) : ∃ a, ∀ x y, pl.line2 a x y ↔ pl.line2 1 x y := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_one_l562_56226


namespace NUMINAMATH_CALUDE_fraction_pattern_sum_of_fractions_sum_of_irrational_fractions_l562_56253

-- Part 1: Pattern for positive integers
theorem fraction_pattern (n : ℕ+) : 
  1 / n.val * (1 / (n.val + 1)) = 1 / n.val - 1 / (n.val + 1) := by sorry

-- Part 2: Sum of fractions
theorem sum_of_fractions (x : ℝ) : 
  1 / (x * (x + 1)) + 1 / ((x + 1) * (x + 2)) + 1 / ((x + 2) * (x + 3)) + 1 / ((x + 3) * (x + 4)) = 
  4 / (x^2 + 4*x) := by sorry

-- Part 3: Sum of irrational fractions
theorem sum_of_irrational_fractions : 
  1 / (1 + Real.sqrt 2) + 1 / (Real.sqrt 2 + Real.sqrt 3) + 1 / (Real.sqrt 3 + 2) + 1 / (2 + Real.sqrt 5) +
  1 / (Real.sqrt 5 + Real.sqrt 6) + 1 / (Real.sqrt 6 + 3) + 1 / (3 + Real.sqrt 10) = 
  -1 + Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_fraction_pattern_sum_of_fractions_sum_of_irrational_fractions_l562_56253


namespace NUMINAMATH_CALUDE_sum_of_fours_and_fives_l562_56252

/-- The number of ways to write 1800 as the sum of fours and fives -/
def ways_to_sum_1800 : ℕ :=
  (Finset.range 201).card

/-- Theorem: There are exactly 201 ways to write 1800 as the sum of fours and fives -/
theorem sum_of_fours_and_fives :
  ways_to_sum_1800 = 201 := by
  sorry

#eval ways_to_sum_1800  -- Should output 201

end NUMINAMATH_CALUDE_sum_of_fours_and_fives_l562_56252


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l562_56248

theorem initial_number_of_persons (n : ℕ) 
  (avg_weight_increase : ℝ) 
  (weight_difference : ℝ) : 
  avg_weight_increase = 2.5 →
  weight_difference = 20 →
  weight_difference = n * avg_weight_increase →
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l562_56248


namespace NUMINAMATH_CALUDE_movie_group_composition_l562_56289

-- Define the ticket prices and group information
def adult_price : ℚ := 9.5
def child_price : ℚ := 6.5
def total_people : ℕ := 7
def total_paid : ℚ := 54.5

-- Define the theorem
theorem movie_group_composition :
  ∃ (adults : ℕ) (children : ℕ),
    adults + children = total_people ∧
    (adult_price * adults + child_price * children : ℚ) = total_paid ∧
    adults = 3 := by
  sorry

end NUMINAMATH_CALUDE_movie_group_composition_l562_56289


namespace NUMINAMATH_CALUDE_storm_encounter_average_time_l562_56251

/-- Represents the position of an object in a 2D plane -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents a moving circular storm -/
structure Storm where
  center : Position
  radius : ℝ
  velocity : Position

/-- Represents a car moving in a straight line -/
structure Car where
  position : Position
  velocity : ℝ

/-- Calculates the position of an object after time t -/
def position_at_time (initial : Position) (velocity : Position) (t : ℝ) : Position :=
  { x := initial.x + velocity.x * t
  , y := initial.y + velocity.y * t }

/-- Determines if a point is inside a circle -/
def is_inside_circle (point : Position) (center : Position) (radius : ℝ) : Prop :=
  (point.x - center.x)^2 + (point.y - center.y)^2 ≤ radius^2

/-- The main theorem to be proved -/
theorem storm_encounter_average_time 
  (car : Car)
  (storm : Storm)
  (t₁ t₂ : ℝ) :
  car.position = { x := 0, y := 0 } →
  car.velocity = 3/4 →
  storm.center = { x := 0, y := 150 } →
  storm.radius = 75 →
  storm.velocity = { x := 3/4, y := -3/4 } →
  (is_inside_circle (position_at_time car.position { x := car.velocity, y := 0 } t₁) 
                    (position_at_time storm.center storm.velocity t₁) 
                    storm.radius) →
  (is_inside_circle (position_at_time car.position { x := car.velocity, y := 0 } t₂) 
                    (position_at_time storm.center storm.velocity t₂) 
                    storm.radius) →
  (∀ t, t₁ < t ∧ t < t₂ → 
    is_inside_circle (position_at_time car.position { x := car.velocity, y := 0 } t) 
                     (position_at_time storm.center storm.velocity t) 
                     storm.radius) →
  (t₁ + t₂) / 2 = 400 :=
by sorry

end NUMINAMATH_CALUDE_storm_encounter_average_time_l562_56251


namespace NUMINAMATH_CALUDE_triangles_from_points_l562_56257

/-- Represents a triangular paper with points -/
structure TriangularPaper where
  n : ℕ  -- number of points inside the triangle

/-- Condition that no three points are collinear -/
axiom not_collinear (paper : TriangularPaper) : True

/-- Function to calculate the number of smaller triangles -/
def num_triangles (paper : TriangularPaper) : ℕ :=
  2 * paper.n + 1

/-- Theorem stating the relationship between points and triangles -/
theorem triangles_from_points (paper : TriangularPaper) :
  num_triangles paper = 2 * paper.n + 1 :=
sorry

end NUMINAMATH_CALUDE_triangles_from_points_l562_56257


namespace NUMINAMATH_CALUDE_hyperbola_equation_l562_56277

/-- Given an ellipse and a hyperbola with the same foci, if one asymptote of the hyperbola
    is y = √2 x, then the equation of the hyperbola is 2y^2 - 4x^2 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (a b : ℝ), (4 * x^2 + y^2 = 1) ∧ 
   (∃ (m : ℝ), 0 < m ∧ m < 3/4 ∧ 
     y^2 / m - x^2 / ((3/4) - m) = 1) ∧
   (∃ (k : ℝ), y = k * x ∧ k^2 = 2)) →
  2 * y^2 - 4 * x^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l562_56277


namespace NUMINAMATH_CALUDE_parabola_intersection_point_l562_56283

theorem parabola_intersection_point (n m : ℕ) (x₀ y₀ : ℝ) 
  (hn : n ≥ 2) 
  (hm : m > 0) 
  (h1 : y₀^2 = n * x₀ - 1) 
  (h2 : y₀ = x₀) : 
  ∃ k : ℕ, k ≥ 2 ∧ (x₀^m)^2 = k * (x₀^m) - 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_point_l562_56283


namespace NUMINAMATH_CALUDE_skate_cost_theorem_l562_56232

/-- The cost of a new pair of skates is equal to 26 times the rental fee. -/
theorem skate_cost_theorem (admission_fee : ℝ) (rental_fee : ℝ) (visits : ℕ) 
  (h1 : admission_fee = 5)
  (h2 : rental_fee = 2.5)
  (h3 : visits = 26) :
  visits * rental_fee = 65 := by
  sorry

#check skate_cost_theorem

end NUMINAMATH_CALUDE_skate_cost_theorem_l562_56232


namespace NUMINAMATH_CALUDE_nested_square_root_18_l562_56250

theorem nested_square_root_18 :
  ∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x = (1 + Real.sqrt 73) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_18_l562_56250


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l562_56290

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * (2 - x)

-- Theorem statement
theorem f_strictly_increasing : 
  ∀ x y, 0 < x ∧ x < y ∧ y < 4/3 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l562_56290


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l562_56215

theorem divisible_by_eleven (m : ℕ) : 
  m < 10 →
  (864 * 10^7 + m * 10^6 + 5 * 10^5 + 3 * 10^4 + 7 * 10^3 + 9 * 10^2 + 7 * 10 + 9) % 11 = 0 →
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l562_56215


namespace NUMINAMATH_CALUDE_red_blood_cell_diameter_scientific_notation_l562_56220

/-- Expresses a given decimal number in scientific notation -/
def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem red_blood_cell_diameter_scientific_notation :
  scientific_notation 0.00077 = (7.7, -4) :=
sorry

end NUMINAMATH_CALUDE_red_blood_cell_diameter_scientific_notation_l562_56220


namespace NUMINAMATH_CALUDE_journey_distance_l562_56291

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 20)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) : 
  ∃ (distance : ℝ), distance = 448 ∧ 
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l562_56291


namespace NUMINAMATH_CALUDE_rationalize_denominator_l562_56204

theorem rationalize_denominator : 
  ∃ (a b : ℝ) (h : b ≠ 0), (7 / (2 * Real.sqrt 98)) = a / b ∧ b * Real.sqrt b = b := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l562_56204


namespace NUMINAMATH_CALUDE_solution_product_l562_56262

theorem solution_product (a b : ℝ) : 
  (a - 3) * (3 * a + 7) = a^2 - 16 * a + 55 →
  (b - 3) * (3 * b + 7) = b^2 - 16 * b + 55 →
  a ≠ b →
  (a + 2) * (b + 2) = -54 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l562_56262


namespace NUMINAMATH_CALUDE_contractor_problem_solution_correctness_l562_56234

/-- Represents the number of days required to complete the work -/
def original_days : ℕ := 9

/-- Represents the number of absent laborers -/
def absent_laborers : ℕ := 6

/-- Represents the number of days required to complete the work with absent laborers -/
def new_days : ℕ := 15

/-- Represents the original number of laborers -/
def original_laborers : ℕ := 15

theorem contractor_problem :
  original_laborers * original_days = (original_laborers - absent_laborers) * new_days :=
by sorry

theorem solution_correctness :
  original_laborers = 15 :=
by sorry

end NUMINAMATH_CALUDE_contractor_problem_solution_correctness_l562_56234


namespace NUMINAMATH_CALUDE_certain_number_proof_l562_56270

theorem certain_number_proof : ∃ n : ℕ, n * 240 = 1038 * 40 ∧ n = 173 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l562_56270
