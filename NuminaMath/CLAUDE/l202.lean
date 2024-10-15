import Mathlib

namespace NUMINAMATH_CALUDE_zoo_guide_count_l202_20235

/-- Represents the number of children spoken to by a guide on a specific day -/
structure DailyGuideCount where
  english : ℕ
  french : ℕ
  spanish : ℕ

/-- Represents the count of guides for each language -/
structure GuideCount where
  total : ℕ
  english : ℕ
  french : ℕ

def weekend_count (guides : GuideCount) (friday saturday sunday : DailyGuideCount) : ℕ :=
  let spanish_guides := guides.total - guides.english - guides.french
  let friday_total := guides.english * friday.english + guides.french * friday.french + spanish_guides * friday.spanish
  let saturday_total := guides.english * saturday.english + guides.french * saturday.french + spanish_guides * saturday.spanish
  let sunday_total := guides.english * sunday.english + guides.french * sunday.french + spanish_guides * sunday.spanish
  friday_total + saturday_total + sunday_total

theorem zoo_guide_count :
  let guides : GuideCount := { total := 22, english := 10, french := 6 }
  let friday : DailyGuideCount := { english := 20, french := 25, spanish := 30 }
  let saturday : DailyGuideCount := { english := 22, french := 24, spanish := 32 }
  let sunday : DailyGuideCount := { english := 24, french := 23, spanish := 35 }
  weekend_count guides friday saturday sunday = 1674 := by
  sorry


end NUMINAMATH_CALUDE_zoo_guide_count_l202_20235


namespace NUMINAMATH_CALUDE_smallest_a_inequality_l202_20266

theorem smallest_a_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  (2/9 : ℝ) * (x^2 + y^2 + z^2) + x*y*z ≥ 10/27 ∧
  ∀ a < 2/9, ∃ x' y' z' : ℝ, x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 ∧
    a * (x'^2 + y'^2 + z'^2) + x'*y'*z' < 10/27 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_inequality_l202_20266


namespace NUMINAMATH_CALUDE_train_length_calculation_l202_20297

/-- Conversion factor from km/hr to m/s -/
def kmhr_to_ms : ℚ := 5 / 18

/-- Calculate the length of a train given its speed in km/hr and crossing time in seconds -/
def train_length (speed : ℚ) (time : ℚ) : ℚ :=
  speed * kmhr_to_ms * time

/-- The cumulative length of two trains -/
def cumulative_length (speed1 speed2 time1 time2 : ℚ) : ℚ :=
  train_length speed1 time1 + train_length speed2 time2

theorem train_length_calculation (speed1 speed2 time1 time2 : ℚ) :
  speed1 = 27 ∧ speed2 = 45 ∧ time1 = 20 ∧ time2 = 30 →
  cumulative_length speed1 speed2 time1 time2 = 525 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l202_20297


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l202_20233

/-- Represents a cricket game with given parameters -/
structure CricketGame where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_part_overs
  let first_part_runs := game.first_part_run_rate * game.first_part_overs
  let remaining_runs := game.target_runs - first_part_runs
  remaining_runs / remaining_overs

/-- The main theorem stating the required run rate for the given game parameters -/
theorem cricket_run_rate_theorem (game : CricketGame) 
    (h_total_overs : game.total_overs = 50)
    (h_first_part_overs : game.first_part_overs = 10)
    (h_first_part_run_rate : game.first_part_run_rate = 3.2)
    (h_target_runs : game.target_runs = 242) :
    required_run_rate game = 5.25 := by
  sorry

#eval required_run_rate {
  total_overs := 50,
  first_part_overs := 10,
  first_part_run_rate := 3.2,
  target_runs := 242
}

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l202_20233


namespace NUMINAMATH_CALUDE_polynomial_equality_l202_20229

theorem polynomial_equality (a b c d e : ℝ) :
  (∀ x : ℝ, (x - 3)^4 = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
  b + c + d + e = 15 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l202_20229


namespace NUMINAMATH_CALUDE_remaining_oranges_l202_20258

-- Define the initial number of oranges Mildred collects
def initial_oranges : ℝ := 77.0

-- Define the number of oranges Mildred's father eats
def eaten_oranges : ℝ := 2.0

-- Theorem stating the number of oranges Mildred has after her father eats some
theorem remaining_oranges : initial_oranges - eaten_oranges = 75.0 := by
  sorry

end NUMINAMATH_CALUDE_remaining_oranges_l202_20258


namespace NUMINAMATH_CALUDE_tricycle_wheels_count_l202_20293

theorem tricycle_wheels_count :
  ∀ (tricycle_wheels : ℕ),
    3 * 2 + 4 * tricycle_wheels + 7 * 1 = 25 →
    tricycle_wheels = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tricycle_wheels_count_l202_20293


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l202_20240

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n = 15625) ∧
  (∀ m : ℕ, m < n → m < 10000 ∨ m > 99999 ∨ ¬∃ a : ℕ, m = a^2 ∨ ¬∃ b : ℕ, m = b^3) ∧
  (∃ x : ℕ, n = x^2) ∧
  (∃ y : ℕ, n = y^3) ∧
  (n ≥ 10000) ∧
  (n ≤ 99999) :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l202_20240


namespace NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l202_20206

theorem triangle_sine_sum_inequality (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l202_20206


namespace NUMINAMATH_CALUDE_mechanics_total_charge_l202_20254

/-- Calculates the total charge for two mechanics given their hourly rates, total combined work time, and the second mechanic's work time. -/
theorem mechanics_total_charge 
  (rate1 : ℕ) 
  (rate2 : ℕ) 
  (total_hours : ℕ) 
  (hours2 : ℕ) : 
  rate1 = 45 → 
  rate2 = 85 → 
  total_hours = 20 → 
  hours2 = 5 → 
  rate1 * (total_hours - hours2) + rate2 * hours2 = 1100 := by
sorry

end NUMINAMATH_CALUDE_mechanics_total_charge_l202_20254


namespace NUMINAMATH_CALUDE_angle_c_is_30_degrees_l202_20283

theorem angle_c_is_30_degrees (A B C : ℝ) : 
  3 * Real.sin A + 4 * Real.cos B = 6 →
  4 * Real.sin B + 3 * Real.cos A = 1 →
  A + B + C = π →
  C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_is_30_degrees_l202_20283


namespace NUMINAMATH_CALUDE_cubic_factorization_l202_20257

theorem cubic_factorization (a : ℝ) : a^3 - 9*a = a*(a+3)*(a-3) := by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l202_20257


namespace NUMINAMATH_CALUDE_rectangle_area_l202_20204

/-- A rectangle with length four times its width and perimeter 200 cm has an area of 1600 cm² --/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 → l * w = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l202_20204


namespace NUMINAMATH_CALUDE_hoseok_marbles_l202_20215

theorem hoseok_marbles : ∃ x : ℕ+, x * 80 + 260 = x * 100 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_marbles_l202_20215


namespace NUMINAMATH_CALUDE_hyperbola_equation_l202_20211

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  hpos : 0 < a ∧ 0 < b
  hc : c = Real.sqrt 5
  hasymptote : b / a = 1 / 2

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 / 4 - y^2 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1

theorem hyperbola_equation (h : Hyperbola) : standard_equation h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l202_20211


namespace NUMINAMATH_CALUDE_complex_fraction_equals_negative_i_negative_i_coordinates_l202_20223

theorem complex_fraction_equals_negative_i :
  let z : ℂ := (1 - 2*I) / (2 + I)
  z = -I :=
by sorry

theorem negative_i_coordinates :
  let z : ℂ := -I
  Complex.re z = 0 ∧ Complex.im z = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_negative_i_negative_i_coordinates_l202_20223


namespace NUMINAMATH_CALUDE_kite_area_is_102_l202_20281

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : Int
  y : Int

/-- Represents a kite shape -/
structure Kite where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Calculate the area of a kite given its vertices -/
def kiteArea (k : Kite) : Int :=
  sorry

/-- The kite in the problem -/
def problemKite : Kite := {
  p1 := { x := 0, y := 10 }
  p2 := { x := 6, y := 14 }
  p3 := { x := 12, y := 10 }
  p4 := { x := 6, y := 0 }
}

theorem kite_area_is_102 : kiteArea problemKite = 102 := by
  sorry

end NUMINAMATH_CALUDE_kite_area_is_102_l202_20281


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l202_20269

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 - 5 * x - 2 < 0 ↔ -1/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l202_20269


namespace NUMINAMATH_CALUDE_probability_even_distinct_digits_l202_20248

def is_even (n : ℕ) : Prop := n % 2 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.Nodup digits

def count_valid_numbers : ℕ := sorry

theorem probability_even_distinct_digits :
  (count_valid_numbers : ℚ) / (9999 - 1000 + 1 : ℚ) = 343 / 1125 := by sorry

end NUMINAMATH_CALUDE_probability_even_distinct_digits_l202_20248


namespace NUMINAMATH_CALUDE_vitamin_a_content_l202_20298

/-- The amount of Vitamin A in a single pill, in mg -/
def vitamin_a_per_pill : ℝ := 50

/-- The recommended daily serving of Vitamin A, in mg -/
def daily_recommended : ℝ := 200

/-- The number of pills needed for the weekly recommended amount -/
def pills_per_week : ℕ := 28

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem vitamin_a_content :
  vitamin_a_per_pill = daily_recommended * (days_per_week : ℝ) / (pills_per_week : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vitamin_a_content_l202_20298


namespace NUMINAMATH_CALUDE_rice_wheat_division_l202_20239

/-- Calculates the approximate amount of wheat grains in a large quantity of mixed grains,
    given a sample ratio. -/
def approximate_wheat_amount (total_amount : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) : ℕ :=
  (total_amount * wheat_in_sample) / sample_size

/-- The rice and wheat division problem from "Jiuzhang Suanshu" -/
theorem rice_wheat_division :
  let total_amount : ℕ := 1536
  let sample_size : ℕ := 224
  let wheat_in_sample : ℕ := 28
  approximate_wheat_amount total_amount sample_size wheat_in_sample = 192 := by
  sorry

#eval approximate_wheat_amount 1536 224 28

end NUMINAMATH_CALUDE_rice_wheat_division_l202_20239


namespace NUMINAMATH_CALUDE_exists_even_b_for_odd_n_l202_20288

def operation (p : ℕ × ℕ) : ℕ × ℕ :=
  if p.1 % 2 = 0 then (p.1 / 2, p.2 + p.1 / 2)
  else (p.1 + p.2 / 2, p.2 / 2)

def applyOperationNTimes (p : ℕ × ℕ) (n : ℕ) : ℕ × ℕ :=
  match n with
  | 0 => p
  | m + 1 => operation (applyOperationNTimes p m)

theorem exists_even_b_for_odd_n (n : ℕ) (h_odd : n % 2 = 1) (h_gt_1 : n > 1) :
  ∃ b : ℕ, b % 2 = 0 ∧ b < n ∧ ∃ k : ℕ, applyOperationNTimes (n, b) k = (b, n) := by
  sorry

end NUMINAMATH_CALUDE_exists_even_b_for_odd_n_l202_20288


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l202_20286

theorem opposite_of_negative_2023 : -((-2023) : ℤ) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l202_20286


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l202_20274

theorem power_of_two_divisibility (n : ℕ+) :
  (∀ (n : ℕ+), ∃ (m : ℤ), (2^n.val - 1) ∣ (m^2 + 9)) ↔
  ∃ (s : ℕ), n.val = 2^s :=
sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l202_20274


namespace NUMINAMATH_CALUDE_min_operations_cube_l202_20299

/-- Represents a rhombus configuration --/
structure RhombusConfig :=
  (n : ℕ)
  (rhombuses : ℕ)

/-- Represents a rearrangement operation --/
inductive RearrangementOp
  | insert
  | remove

/-- The minimum number of operations to transform the configuration --/
def min_operations (config : RhombusConfig) : ℕ :=
  config.n^3

/-- Theorem stating that the minimum number of operations is n³ --/
theorem min_operations_cube (config : RhombusConfig) 
  (h1 : config.rhombuses = 3 * config.n^2) :
  min_operations config = config.n^3 := by
  sorry

#check min_operations_cube

end NUMINAMATH_CALUDE_min_operations_cube_l202_20299


namespace NUMINAMATH_CALUDE_empty_proper_subset_implies_nonempty_l202_20272

theorem empty_proper_subset_implies_nonempty (A : Set α) :
  ∅ ⊂ A → A ≠ ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_proper_subset_implies_nonempty_l202_20272


namespace NUMINAMATH_CALUDE_function_product_l202_20262

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem function_product (y z : ℝ) 
  (h1 : -1 < y ∧ y < 1) 
  (h2 : -1 < z ∧ z < 1) 
  (h3 : f ((y + z) / (1 + y * z)) = 1) 
  (h4 : f ((y - z) / (1 - y * z)) = 2) : 
  f y * f z = -3/4 := by
sorry

end NUMINAMATH_CALUDE_function_product_l202_20262


namespace NUMINAMATH_CALUDE_sphere_with_cylindrical_hole_volume_l202_20287

theorem sphere_with_cylindrical_hole_volume :
  let R : ℝ := Real.sqrt 3
  let sphere_volume := (4 / 3) * Real.pi * R^3
  let cylinder_radius := R / 2
  let cylinder_height := R * Real.sqrt 3
  let cylinder_volume := Real.pi * cylinder_radius^2 * cylinder_height
  let spherical_cap_height := R * (2 - Real.sqrt 3) / 2
  let spherical_cap_volume := (Real.pi * spherical_cap_height^2 * (3 * R - spherical_cap_height)) / 3
  let remaining_volume := sphere_volume - cylinder_volume - 2 * spherical_cap_volume
  remaining_volume = (9 * Real.pi) / 2 := by
sorry


end NUMINAMATH_CALUDE_sphere_with_cylindrical_hole_volume_l202_20287


namespace NUMINAMATH_CALUDE_marys_max_earnings_l202_20207

/-- Calculates the maximum weekly earnings for a worker with the given parameters. -/
def maxWeeklyEarnings (maxHours regularHours : ℕ) (regularRate : ℚ) (overtimeRateIncrease : ℚ) : ℚ :=
  let regularEarnings := regularRate * regularHours
  let overtimeRate := regularRate * (1 + overtimeRateIncrease)
  let overtimeHours := maxHours - regularHours
  let overtimeEarnings := overtimeRate * overtimeHours
  regularEarnings + overtimeEarnings

/-- Theorem stating that Mary's maximum weekly earnings are $460 -/
theorem marys_max_earnings :
  maxWeeklyEarnings 50 20 8 (1/4) = 460 := by
  sorry

#eval maxWeeklyEarnings 50 20 8 (1/4)

end NUMINAMATH_CALUDE_marys_max_earnings_l202_20207


namespace NUMINAMATH_CALUDE_pizza_slices_theorem_l202_20291

/-- Represents the types of pizzas available --/
inductive PizzaType
  | Small
  | Medium
  | Large

/-- Returns the number of slices for a given pizza type --/
def slicesPerPizza (pt : PizzaType) : Nat :=
  match pt with
  | .Small => 6
  | .Medium => 8
  | .Large => 12

/-- Calculates the total number of slices for a given number of pizzas of a specific type --/
def totalSlices (pt : PizzaType) (count : Nat) : Nat :=
  (slicesPerPizza pt) * count

/-- Represents the order of pizzas --/
structure PizzaOrder where
  small : Nat
  medium : Nat
  large : Nat
  total : Nat

theorem pizza_slices_theorem (order : PizzaOrder)
  (h1 : order.small = 4)
  (h2 : order.medium = 5)
  (h3 : order.total = 15)
  (h4 : order.large = order.total - order.small - order.medium) :
  totalSlices .Small order.small +
  totalSlices .Medium order.medium +
  totalSlices .Large order.large = 136 := by
    sorry

#check pizza_slices_theorem

end NUMINAMATH_CALUDE_pizza_slices_theorem_l202_20291


namespace NUMINAMATH_CALUDE_positive_integer_M_satisfying_equation_l202_20232

theorem positive_integer_M_satisfying_equation : ∃ M : ℕ+, (12^2 * 60^2 : ℕ) = 30^2 * M^2 ∧ M = 12 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_M_satisfying_equation_l202_20232


namespace NUMINAMATH_CALUDE_lines_intersect_at_one_point_l202_20263

-- Define the basic geometric objects
variable (A B C D E F P Q M O : Point)

-- Define the convex quadrilateral
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the extension relationships
def lies_on_extension (P X Y Z : Point) : Prop := sorry

-- Define the midpoint relationship
def is_midpoint (M X Y : Point) : Prop := sorry

-- Define when a point lies on a line
def point_on_line (P X Y : Point) : Prop := sorry

-- Main theorem
theorem lines_intersect_at_one_point 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_E_ext : lies_on_extension E A B B)
  (h_F_ext : lies_on_extension F C D D)
  (h_M_mid_AD : is_midpoint M A D)
  (h_P_on_BE : point_on_line P B E)
  (h_Q_on_DF : point_on_line Q D F)
  (h_M_mid_PQ : is_midpoint M P Q) :
  ∃ O, point_on_line O A B ∧ point_on_line O C D ∧ point_on_line O P Q :=
sorry

end NUMINAMATH_CALUDE_lines_intersect_at_one_point_l202_20263


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l202_20217

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l202_20217


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l202_20202

theorem square_sum_geq_product_sum {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l202_20202


namespace NUMINAMATH_CALUDE_simplify_expressions_l202_20226

theorem simplify_expressions :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (Real.sqrt (1 / 3) + Real.sqrt 27 * Real.sqrt 9 = 28 * Real.sqrt 3 / 3) ∧
    (Real.sqrt 32 - 3 * Real.sqrt (1 / 2) + Real.sqrt (1 / 8) = 11 * Real.sqrt 2 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l202_20226


namespace NUMINAMATH_CALUDE_composite_function_solution_l202_20250

def δ (x : ℝ) : ℝ := 5 * x + 6
def φ (x : ℝ) : ℝ := 9 * x + 4

theorem composite_function_solution :
  ∀ x : ℝ, δ (φ x) = 14 → x = -4/15 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_solution_l202_20250


namespace NUMINAMATH_CALUDE_vehicle_speed_problem_l202_20242

/-- Proves that the average speed of vehicle X is 36 miles per hour given the conditions -/
theorem vehicle_speed_problem (initial_distance : ℝ) (y_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 22 →
  y_speed = 45 →
  time = 5 →
  final_distance = 23 →
  let x_distance := y_speed * time - (initial_distance + final_distance)
  let x_speed := x_distance / time
  x_speed = 36 := by sorry

end NUMINAMATH_CALUDE_vehicle_speed_problem_l202_20242


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l202_20264

theorem multiplication_addition_equality : 15 * 35 + 45 * 15 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l202_20264


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l202_20261

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x + y + z = 3) :
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) > 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l202_20261


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l202_20277

/-- Represents a polynomial of degree 4 with rational coefficients -/
structure Polynomial4 where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a given number is a root of the polynomial -/
def isRoot (p : Polynomial4) (x : ℝ) : Prop :=
  x^4 + p.a * x^2 + p.b * x + p.c = 0

theorem integer_roots_of_polynomial (p : Polynomial4) :
  isRoot p (2 - Real.sqrt 5) →
  (∃ (r₁ r₂ : ℤ), isRoot p (r₁ : ℝ) ∧ isRoot p (r₂ : ℝ)) →
  ∃ (r : ℤ), isRoot p (r : ℝ) ∧ r = -2 :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l202_20277


namespace NUMINAMATH_CALUDE_no_integer_solution_l202_20245

theorem no_integer_solution : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l202_20245


namespace NUMINAMATH_CALUDE_class_vision_only_comprehensive_l202_20259

-- Define the concept of a survey
structure Survey where
  population : Type
  data_collection : population → Bool

-- Define what makes a survey comprehensive
def is_comprehensive (s : Survey) : Prop :=
  ∀ x : s.population, s.data_collection x

-- Define the specific surveys
def bulb_survey : Survey := sorry
def class_vision_survey : Survey := sorry
def food_preservative_survey : Survey := sorry
def river_water_quality_survey : Survey := sorry

-- State the theorem
theorem class_vision_only_comprehensive :
  is_comprehensive class_vision_survey ∧
  ¬is_comprehensive bulb_survey ∧
  ¬is_comprehensive food_preservative_survey ∧
  ¬is_comprehensive river_water_quality_survey :=
sorry

end NUMINAMATH_CALUDE_class_vision_only_comprehensive_l202_20259


namespace NUMINAMATH_CALUDE_pauls_toy_boxes_l202_20253

theorem pauls_toy_boxes (toys_per_box : ℕ) (total_toys : ℕ) (h1 : toys_per_box = 8) (h2 : total_toys = 32) :
  total_toys / toys_per_box = 4 := by
sorry

end NUMINAMATH_CALUDE_pauls_toy_boxes_l202_20253


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocals_l202_20228

theorem min_sum_of_reciprocals (x y z : ℕ+) (h : (1 : ℚ) / x + 4 / y + 9 / z = 1) :
  36 ≤ (x : ℚ) + y + z :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocals_l202_20228


namespace NUMINAMATH_CALUDE_business_value_l202_20271

/-- Given a man who owns 2/3 of a business and sells 3/4 of his shares for 45,000 Rs,
    prove that the value of the entire business is 90,000 Rs. -/
theorem business_value (man_share : ℚ) (sold_portion : ℚ) (sold_value : ℕ) :
  man_share = 2/3 →
  sold_portion = 3/4 →
  sold_value = 45000 →
  ∃ (total_value : ℕ), total_value = 90000 ∧
    (total_value : ℚ) = sold_value / (man_share * sold_portion) :=
by sorry

end NUMINAMATH_CALUDE_business_value_l202_20271


namespace NUMINAMATH_CALUDE_unique_contributions_exist_l202_20224

/-- Represents the contributions of five friends to a project -/
structure Contributions where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (c : Contributions) : Prop :=
  c.A = 1.1 * c.B ∧
  c.C = 0.8 * c.A ∧
  c.D = 2 * c.B ∧
  c.E = c.D - 200 ∧
  c.A + c.B + c.C + c.D + c.E = 1500

/-- Theorem stating that there exists a unique set of contributions satisfying the conditions -/
theorem unique_contributions_exist : ∃! c : Contributions, satisfies_conditions c :=
sorry

end NUMINAMATH_CALUDE_unique_contributions_exist_l202_20224


namespace NUMINAMATH_CALUDE_min_sum_squares_l202_20220

theorem min_sum_squares (a b c : ℝ) (h : a^3 + b^3 + c^3 - 3*a*b*c = 8) :
  ∃ (m : ℝ), (∀ x y z : ℝ, x^3 + y^3 + z^3 - 3*x*y*z = 8 → x^2 + y^2 + z^2 ≥ m) ∧
             (a^2 + b^2 + c^2 ≥ m) ∧
             (m = 4) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l202_20220


namespace NUMINAMATH_CALUDE_ratio_equality_l202_20241

theorem ratio_equality {a₁ a₂ a₃ b₁ b₂ b₃ p₁ p₂ p₃ : ℝ} (h1 : a₁ / b₁ = a₂ / b₂) (h2 : a₁ / b₁ = a₃ / b₃)
    (h3 : ¬(p₁ = 0 ∧ p₂ = 0 ∧ p₃ = 0)) :
  ∀ n : ℕ, (a₁ / b₁) ^ n = (p₁ * a₁^n + p₂ * a₂^n + p₃ * a₃^n) / (p₁ * b₁^n + p₂ * b₂^n + p₃ * b₃^n) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l202_20241


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l202_20200

/-- The value of m for which the circle x^2 + y^2 = m^2 is tangent to the line x + y = m -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = m^2 ∧ x + y = m) → m = 0 := by
  sorry


end NUMINAMATH_CALUDE_circle_tangent_to_line_l202_20200


namespace NUMINAMATH_CALUDE_condition1_condition2_man_work_twice_boy_work_l202_20294

/-- The daily work done by a man -/
def M : ℝ := sorry

/-- The daily work done by a boy -/
def B : ℝ := sorry

/-- The total work to be done -/
def total_work : ℝ := sorry

/-- First condition: 12 men and 16 boys can do the work in 5 days -/
theorem condition1 : 5 * (12 * M + 16 * B) = total_work := sorry

/-- Second condition: 13 men and 24 boys can do the work in 4 days -/
theorem condition2 : 4 * (13 * M + 24 * B) = total_work := sorry

/-- Theorem to prove: The daily work done by a man is twice that of a boy -/
theorem man_work_twice_boy_work : M = 2 * B := by sorry

end NUMINAMATH_CALUDE_condition1_condition2_man_work_twice_boy_work_l202_20294


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_3063_l202_20219

theorem smallest_prime_factor_of_3063 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3063 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3063 → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_3063_l202_20219


namespace NUMINAMATH_CALUDE_cakes_destroyed_or_stolen_proof_l202_20222

def total_cakes : ℕ := 36
def num_stacks : ℕ := 2

def cakes_per_stack : ℕ := total_cakes / num_stacks

def crow_knocked_percentage : ℚ := 60 / 100
def mischievous_squirrel_stole_fraction : ℚ := 1 / 3
def red_squirrel_took_percentage : ℚ := 25 / 100
def red_squirrel_dropped_fraction : ℚ := 1 / 2
def dog_ate : ℕ := 4

def cakes_destroyed_or_stolen : ℕ := 19

theorem cakes_destroyed_or_stolen_proof :
  let crow_knocked := (crow_knocked_percentage * cakes_per_stack).floor
  let mischievous_squirrel_stole := (mischievous_squirrel_stole_fraction * crow_knocked).floor
  let red_squirrel_took := (red_squirrel_took_percentage * cakes_per_stack).floor
  let red_squirrel_destroyed := (red_squirrel_dropped_fraction * red_squirrel_took).floor
  crow_knocked + mischievous_squirrel_stole + red_squirrel_destroyed + dog_ate = cakes_destroyed_or_stolen :=
by sorry

end NUMINAMATH_CALUDE_cakes_destroyed_or_stolen_proof_l202_20222


namespace NUMINAMATH_CALUDE_square_root_meaningful_range_l202_20234

theorem square_root_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 6 * x + 12) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_meaningful_range_l202_20234


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a10_l202_20255

/-- An arithmetic sequence with a_2 = 2 and a_3 = 4 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 3 = 4 ∧ ∀ n : ℕ, a (n + 1) - a n = a 3 - a 2

theorem arithmetic_sequence_a10 (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a10_l202_20255


namespace NUMINAMATH_CALUDE_subtraction_problem_l202_20273

theorem subtraction_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l202_20273


namespace NUMINAMATH_CALUDE_paper_folding_ratio_l202_20218

theorem paper_folding_ratio : 
  let square_side : ℝ := 1
  let large_rect_length : ℝ := square_side
  let large_rect_width : ℝ := square_side / 2
  let small_rect_length : ℝ := square_side
  let small_rect_width : ℝ := square_side / 4
  let large_rect_perimeter : ℝ := 2 * (large_rect_length + large_rect_width)
  let small_rect_perimeter : ℝ := 2 * (small_rect_length + small_rect_width)
  small_rect_perimeter / large_rect_perimeter = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_paper_folding_ratio_l202_20218


namespace NUMINAMATH_CALUDE_schedule_arrangements_l202_20256

/-- Represents the number of subjects to be scheduled -/
def num_subjects : ℕ := 6

/-- Represents the number of periods in a day -/
def num_periods : ℕ := 6

/-- Calculates the number of arrangements for scheduling subjects with given constraints -/
def num_arrangements : ℕ :=
  5 * 4 * (Finset.range 4).prod (λ i => i + 1)

/-- Theorem stating the number of different arrangements -/
theorem schedule_arrangements :
  num_arrangements = 480 := by sorry

end NUMINAMATH_CALUDE_schedule_arrangements_l202_20256


namespace NUMINAMATH_CALUDE_simplify_exponents_l202_20296

theorem simplify_exponents (a b : ℝ) : (a^4 * a^3) * (b^2 * b^5) = a^7 * b^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponents_l202_20296


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l202_20292

/-- Two lines intersecting on the y-axis -/
structure IntersectingLines where
  k : ℝ
  line1 : ℝ → ℝ → ℝ := fun x y => 2*x + 3*y - k
  line2 : ℝ → ℝ → ℝ := fun x y => x - k*y + 12
  intersect_on_y_axis : ∃ y, line1 0 y = 0 ∧ line2 0 y = 0

/-- The value of k for intersecting lines -/
theorem intersecting_lines_k_value (l : IntersectingLines) : l.k = 6 ∨ l.k = -6 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l202_20292


namespace NUMINAMATH_CALUDE_x_squared_mod_24_l202_20267

theorem x_squared_mod_24 (x : ℤ) 
  (h1 : 6 * x ≡ 12 [ZMOD 24])
  (h2 : 4 * x ≡ 20 [ZMOD 24]) : 
  x^2 ≡ 12 [ZMOD 24] := by
sorry

end NUMINAMATH_CALUDE_x_squared_mod_24_l202_20267


namespace NUMINAMATH_CALUDE_system_solution_l202_20285

def solution_set := {x : ℝ | 0 < x ∧ x < 1}

theorem system_solution : 
  {x : ℝ | x * (x + 2) > 0 ∧ |x| < 1} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l202_20285


namespace NUMINAMATH_CALUDE_wage_difference_l202_20252

/-- The total pay for the research project -/
def total_pay : ℝ := 360

/-- Candidate P's hourly wage -/
def wage_p : ℝ := 18

/-- Candidate Q's hourly wage -/
def wage_q : ℝ := 12

/-- The number of hours candidate P needs to complete the job -/
def hours_p : ℝ := 20

/-- The number of hours candidate Q needs to complete the job -/
def hours_q : ℝ := 30

theorem wage_difference : 
  (wage_p = 1.5 * wage_q) ∧ 
  (hours_q = hours_p + 10) ∧ 
  (wage_p * hours_p = total_pay) ∧ 
  (wage_q * hours_q = total_pay) → 
  wage_p - wage_q = 6 := by
  sorry

end NUMINAMATH_CALUDE_wage_difference_l202_20252


namespace NUMINAMATH_CALUDE_probability_not_raining_l202_20227

theorem probability_not_raining (p : ℚ) (h : p = 4/9) : 1 - p = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_raining_l202_20227


namespace NUMINAMATH_CALUDE_student_count_problem_l202_20243

theorem student_count_problem : ∃! n : ℕ, n < 500 ∧ 
  n % 17 = 15 ∧ 
  n % 19 = 18 ∧ 
  n % 16 = 7 ∧ 
  n = 417 := by
sorry

end NUMINAMATH_CALUDE_student_count_problem_l202_20243


namespace NUMINAMATH_CALUDE_daily_increase_calculation_l202_20208

def squats_sequence (initial : ℕ) (increase : ℕ) (day : ℕ) : ℕ :=
  initial + (day - 1) * increase

theorem daily_increase_calculation (initial : ℕ) (increase : ℕ) :
  initial = 30 →
  squats_sequence initial increase 4 = 45 →
  increase = 5 := by
  sorry

end NUMINAMATH_CALUDE_daily_increase_calculation_l202_20208


namespace NUMINAMATH_CALUDE_trig_expression_equality_l202_20284

theorem trig_expression_equality (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π)  -- α is in the second quadrant
  (h2 : Real.sin (π/2 + α) = -Real.sqrt 5 / 5) :
  (Real.cos α ^ 3 + Real.sin α) / Real.cos (α - π/4) = 9 * Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l202_20284


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l202_20216

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let c := 3 * b
  let e := c / a
  (c + b/2) / (c - b/2) = 7/5 → e = 3 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l202_20216


namespace NUMINAMATH_CALUDE_juan_tire_count_l202_20225

/-- The number of tires on the vehicles Juan saw --/
def total_tires (cars trucks bicycles tricycles : ℕ) : ℕ :=
  4 * (cars + trucks) + 2 * bicycles + 3 * tricycles

/-- Theorem stating the total number of tires Juan saw --/
theorem juan_tire_count : total_tires 15 8 3 1 = 101 := by
  sorry

end NUMINAMATH_CALUDE_juan_tire_count_l202_20225


namespace NUMINAMATH_CALUDE_game_download_time_l202_20236

/-- Calculates the remaining download time for a game -/
theorem game_download_time (total_size : ℕ) (downloaded : ℕ) (speed : ℕ) : 
  total_size = 880 ∧ downloaded = 310 ∧ speed = 3 → 
  (total_size - downloaded) / speed = 190 := by
  sorry

end NUMINAMATH_CALUDE_game_download_time_l202_20236


namespace NUMINAMATH_CALUDE_range_of_m_l202_20231

-- Define the propositions p and q
def p (m : ℝ) : Prop := (1 + 1 - 2*m + 2*m + 2*m^2 - 4) < 0

def q (m : ℝ) : Prop := m ≥ 0 ∧ 2*m + 1 ≥ 0

-- Define the theorem
theorem range_of_m :
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (∀ m : ℝ, (-1 < m ∧ m < 0) ∨ m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l202_20231


namespace NUMINAMATH_CALUDE_smallest_number_l202_20260

theorem smallest_number : 
  ∀ (a b c : ℝ), a = -Real.sqrt 2 ∧ b = 3.14 ∧ c = 2021 → 
    a < 0 ∧ a < b ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l202_20260


namespace NUMINAMATH_CALUDE_units_digit_of_100_factorial_l202_20295

theorem units_digit_of_100_factorial (n : ℕ) : n = 100 → n.factorial % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_100_factorial_l202_20295


namespace NUMINAMATH_CALUDE_sequence_3078th_term_l202_20203

/-- Calculates the sum of cubes of digits of a natural number -/
def sumOfCubesOfDigits (n : ℕ) : ℕ := sorry

/-- Generates the next term in the sequence -/
def nextTerm (n : ℕ) : ℕ := sumOfCubesOfDigits n

/-- Generates the nth term of the sequence starting with the given initial term -/
def nthTerm (initial : ℕ) (n : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem sequence_3078th_term (initial : ℕ) (h : initial = 3078) : 
  nthTerm initial 3078 = 153 := by sorry

end NUMINAMATH_CALUDE_sequence_3078th_term_l202_20203


namespace NUMINAMATH_CALUDE_trackball_mice_count_l202_20290

theorem trackball_mice_count (total : ℕ) (wireless_ratio optical_ratio : ℚ) : 
  total = 80 →
  wireless_ratio = 1/2 →
  optical_ratio = 1/4 →
  (wireless_ratio + optical_ratio + (1 - wireless_ratio - optical_ratio)) = 1 →
  ↑total * (1 - wireless_ratio - optical_ratio) = 20 :=
by sorry

end NUMINAMATH_CALUDE_trackball_mice_count_l202_20290


namespace NUMINAMATH_CALUDE_cookie_accident_l202_20270

/-- Problem: Cookie Baking Accident -/
theorem cookie_accident (alice_initial bob_initial alice_additional bob_additional final_edible : ℕ) :
  alice_initial = 74 →
  bob_initial = 7 →
  alice_additional = 5 →
  bob_additional = 36 →
  final_edible = 93 →
  (alice_initial + bob_initial + alice_additional + bob_additional) - final_edible = 29 :=
by sorry

end NUMINAMATH_CALUDE_cookie_accident_l202_20270


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l202_20268

/-- Given a point (2, -2), its symmetric point with respect to the origin has coordinates (-2, 2) -/
theorem symmetric_point_wrt_origin :
  let original_point : ℝ × ℝ := (2, -2)
  let symmetric_point : ℝ × ℝ := (-2, 2)
  (∀ (x y : ℝ), (x, y) = original_point → (-x, -y) = symmetric_point) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l202_20268


namespace NUMINAMATH_CALUDE_not_equivalent_polar_point_l202_20237

def is_equivalent_polar (r : ℝ) (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem not_equivalent_polar_point :
  ¬ is_equivalent_polar 2 (π/6) (11*π/6) := by
  sorry

end NUMINAMATH_CALUDE_not_equivalent_polar_point_l202_20237


namespace NUMINAMATH_CALUDE_olivia_coins_left_l202_20214

/-- The number of coins Olivia has left after buying a soda -/
def coins_left (initial_quarters : ℕ) (spent_quarters : ℕ) : ℕ :=
  initial_quarters - spent_quarters

/-- Theorem: Olivia has 7 coins left after buying a soda -/
theorem olivia_coins_left : coins_left 11 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_olivia_coins_left_l202_20214


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l202_20289

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

theorem intersection_implies_a_equals_one :
  ∀ a : ℝ, (A ∩ B a = {3}) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l202_20289


namespace NUMINAMATH_CALUDE_zias_club_size_l202_20280

/-- Represents the number of people with one coin -/
def one_coin_people : ℕ := 7

/-- Represents the angle of the smallest sector in degrees -/
def smallest_sector : ℕ := 35

/-- Represents the angle increment between sectors in degrees -/
def angle_increment : ℕ := 10

/-- Calculates the total number of sectors in the pie chart -/
def total_sectors : ℕ := 6

/-- Represents the total angle of a full circle in degrees -/
def full_circle : ℕ := 360

/-- Theorem: The number of people in Zia's club is 72 -/
theorem zias_club_size : 
  (full_circle / (smallest_sector / one_coin_people) : ℕ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_zias_club_size_l202_20280


namespace NUMINAMATH_CALUDE_increasing_cubic_function_parameter_negative_l202_20238

/-- Given a function y = a(x^3 - 3x) that is increasing on the interval (-1, 1), prove that a < 0 --/
theorem increasing_cubic_function_parameter_negative
  (a : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = a * (x^3 - 3*x))
  (h2 : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, StrictMono y):
  a < 0 :=
sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_parameter_negative_l202_20238


namespace NUMINAMATH_CALUDE_raisin_distribution_l202_20251

theorem raisin_distribution (total_raisins total_boxes box1_raisins box2_raisins : ℕ) 
  (h1 : total_raisins = 437)
  (h2 : total_boxes = 5)
  (h3 : box1_raisins = 72)
  (h4 : box2_raisins = 74)
  (h5 : ∃ (equal_box_raisins : ℕ), 
    total_raisins = box1_raisins + box2_raisins + 3 * equal_box_raisins) :
  ∃ (equal_box_raisins : ℕ), equal_box_raisins = 97 ∧ 
    total_raisins = box1_raisins + box2_raisins + 3 * equal_box_raisins :=
by sorry

end NUMINAMATH_CALUDE_raisin_distribution_l202_20251


namespace NUMINAMATH_CALUDE_total_selected_in_survey_l202_20210

/-- The number of residents aged 21 to 35 -/
def residents_21_35 : ℕ := 840

/-- The number of residents aged 36 to 50 -/
def residents_36_50 : ℕ := 700

/-- The number of residents aged 51 to 65 -/
def residents_51_65 : ℕ := 560

/-- The number of people selected from the 36 to 50 age group -/
def selected_36_50 : ℕ := 100

/-- The total number of residents -/
def total_residents : ℕ := residents_21_35 + residents_36_50 + residents_51_65

/-- The theorem stating the total number of people selected in the survey -/
theorem total_selected_in_survey : 
  (selected_36_50 : ℚ) * (total_residents : ℚ) / (residents_36_50 : ℚ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_selected_in_survey_l202_20210


namespace NUMINAMATH_CALUDE_pauls_remaining_books_l202_20230

/-- Calculates the number of books remaining after a sale -/
def books_remaining (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

/-- Theorem: Paul's remaining books after the sale -/
theorem pauls_remaining_books :
  let initial_books : ℕ := 115
  let books_sold : ℕ := 78
  books_remaining initial_books books_sold = 37 := by
  sorry

end NUMINAMATH_CALUDE_pauls_remaining_books_l202_20230


namespace NUMINAMATH_CALUDE_eggs_per_basket_l202_20212

theorem eggs_per_basket (purple_eggs teal_eggs min_eggs : ℕ) 
  (h1 : purple_eggs = 30)
  (h2 : teal_eggs = 42)
  (h3 : min_eggs = 5) :
  ∃ (n : ℕ), n ≥ min_eggs ∧ purple_eggs % n = 0 ∧ teal_eggs % n = 0 ∧ 
  ∀ (m : ℕ), m ≥ min_eggs ∧ purple_eggs % m = 0 ∧ teal_eggs % m = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l202_20212


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l202_20213

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let original_area := L * W
  let new_area := (1.2 * L) * (1.2 * W)
  (new_area - original_area) / original_area * 100 = 44 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l202_20213


namespace NUMINAMATH_CALUDE_complex_magnitude_two_thirds_minus_four_fifths_i_l202_20249

theorem complex_magnitude_two_thirds_minus_four_fifths_i :
  Complex.abs (⟨2/3, -4/5⟩ : ℂ) = Real.sqrt 244 / 15 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_two_thirds_minus_four_fifths_i_l202_20249


namespace NUMINAMATH_CALUDE_quadratic_coefficient_positive_l202_20265

theorem quadratic_coefficient_positive (a c : ℝ) (h₁ : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 2*a*x + c
  f (-1) = 1 ∧ f (-5) = 5 → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_positive_l202_20265


namespace NUMINAMATH_CALUDE_book_cost_price_l202_20282

theorem book_cost_price (selling_price_1 : ℝ) (selling_price_2 : ℝ) : 
  (selling_price_1 = 1.10 * 1800) → 
  (selling_price_2 = 1.15 * 1800) → 
  (selling_price_2 - selling_price_1 = 90) → 
  1800 = 1800 := by
sorry

end NUMINAMATH_CALUDE_book_cost_price_l202_20282


namespace NUMINAMATH_CALUDE_salad_dressing_composition_l202_20276

theorem salad_dressing_composition (vinegar_p : ℝ) (oil_p : ℝ) (vinegar_q : ℝ) (oil_q : ℝ) 
  (ratio_p : ℝ) (ratio_q : ℝ) (vinegar_new : ℝ) :
  vinegar_p = 0.3 →
  vinegar_p + oil_p = 1 →
  vinegar_q = 0.1 →
  oil_q = 0.9 →
  ratio_p = 0.1 →
  ratio_q = 0.9 →
  ratio_p + ratio_q = 1 →
  vinegar_new = 0.12 →
  ratio_p * vinegar_p + ratio_q * vinegar_q = vinegar_new →
  oil_p = 0.7 := by
sorry

end NUMINAMATH_CALUDE_salad_dressing_composition_l202_20276


namespace NUMINAMATH_CALUDE_estimate_overweight_students_l202_20209

def sample_size : ℕ := 100
def total_population : ℕ := 2000
def frequencies : List ℝ := [0.04, 0.035, 0.015]

theorem estimate_overweight_students :
  let total_frequency := (List.sum frequencies) * (total_population / sample_size)
  let estimated_students := total_population * total_frequency
  estimated_students = 360 := by sorry

end NUMINAMATH_CALUDE_estimate_overweight_students_l202_20209


namespace NUMINAMATH_CALUDE_resource_sum_theorem_l202_20205

/-- Converts a base 6 number to base 10 -/
def base6_to_base10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The amount of mineral X in base 6 -/
def mineral_x : List Nat := [2, 3, 4, 1]

/-- The amount of mineral Y in base 6 -/
def mineral_y : List Nat := [4, 1, 2, 3]

/-- The amount of water in base 6 -/
def water : List Nat := [4, 1, 2]

theorem resource_sum_theorem :
  base6_to_base10 mineral_x + base6_to_base10 mineral_y + base6_to_base10 water = 868 := by
  sorry

end NUMINAMATH_CALUDE_resource_sum_theorem_l202_20205


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l202_20275

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the property of being a positive sequence
def IsPositive (a : Sequence) : Prop :=
  ∀ n : ℕ, a n > 0

-- Define the recurrence relation
def SatisfiesRecurrence (a : Sequence) : Prop :=
  ∀ n : ℕ, a (n + 2) = a n * a (n + 1)

-- Define the property of being a geometric sequence
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_condition (a : Sequence) 
  (h_pos : IsPositive a) (h_rec : SatisfiesRecurrence a) :
  IsGeometric a ↔ a 1 = 1 ∧ a 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l202_20275


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l202_20201

/-- Proves that given a boat with speed 18 kmph in still water and a stream with speed 6 kmph,
    if the boat can cover 48 km downstream or a certain distance upstream in the same time,
    then the distance the boat can cover upstream is 24 km. -/
theorem boat_upstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) :
  boat_speed = 18 →
  stream_speed = 6 →
  downstream_distance = 48 →
  (downstream_distance / (boat_speed + stream_speed) = 
   (boat_speed - stream_speed) * (downstream_distance / (boat_speed + stream_speed))) →
  (boat_speed - stream_speed) * (downstream_distance / (boat_speed + stream_speed)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_boat_upstream_distance_l202_20201


namespace NUMINAMATH_CALUDE_father_son_age_difference_l202_20279

/-- Proves that a father is 25 years older than his son given the problem conditions -/
theorem father_son_age_difference :
  ∀ (father_age son_age : ℕ),
    father_age > son_age →
    son_age = 23 →
    father_age + 2 = 2 * (son_age + 2) →
    father_age - son_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_difference_l202_20279


namespace NUMINAMATH_CALUDE_train_speed_problem_l202_20246

theorem train_speed_problem (faster_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 44 →
  passing_time = 36 →
  train_length = 40 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * (5/18) * passing_time = 2 * train_length :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l202_20246


namespace NUMINAMATH_CALUDE_weight_sum_proof_l202_20221

/-- Given the weights of four people in pairs, prove that the sum of two specific people's weights can be determined. -/
theorem weight_sum_proof (e f g h : ℝ) 
  (ef_sum : e + f = 280)
  (fg_sum : f + g = 230)
  (gh_sum : g + h = 260) :
  e + h = 310 := by sorry

end NUMINAMATH_CALUDE_weight_sum_proof_l202_20221


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l202_20278

variable (a b c : ℤ)
variable (ω : ℂ)

theorem min_value_complex_expression (h1 : a * b * c = 60)
                                     (h2 : ω ≠ 1)
                                     (h3 : ω^3 = 1) :
  ∃ (min : ℝ), min = Real.sqrt 3 ∧
    ∀ (x y z : ℤ), x * y * z = 60 →
      Complex.abs (↑x + ↑y * ω + ↑z * ω^2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l202_20278


namespace NUMINAMATH_CALUDE_set_operation_result_l202_20247

def U : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem set_operation_result :
  A ∪ (U \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l202_20247


namespace NUMINAMATH_CALUDE_ellipse_iff_k_range_l202_20244

/-- The curve equation -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 / (4 - k) + y^2 / (k - 1) = 1

/-- Conditions for the curve to be an ellipse -/
def is_ellipse (k : ℝ) : Prop :=
  4 - k > 0 ∧ k - 1 > 0 ∧ 4 - k ≠ k - 1

/-- The range of k for which the curve is an ellipse -/
def k_range (k : ℝ) : Prop :=
  1 < k ∧ k < 4 ∧ k ≠ 5/2

/-- Theorem: The curve is an ellipse if and only if k is in the specified range -/
theorem ellipse_iff_k_range (k : ℝ) :
  is_ellipse k ↔ k_range k :=
sorry

end NUMINAMATH_CALUDE_ellipse_iff_k_range_l202_20244
