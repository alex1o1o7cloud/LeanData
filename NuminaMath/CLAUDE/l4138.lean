import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_condition_l4138_413800

theorem unique_solution_condition (j : ℝ) : 
  (∃! x : ℝ, (2*x + 7)*(x - 5) = -43 + j*x) ↔ (j = 5 ∨ j = -11) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l4138_413800


namespace NUMINAMATH_CALUDE_circle_centers_distance_l4138_413827

theorem circle_centers_distance (r₁ r₂ : ℝ) (angle : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 95) (h₃ : angle = 60) :
  let distance := 2 * r₂ - 2 * r₁
  distance = 160 :=
sorry

end NUMINAMATH_CALUDE_circle_centers_distance_l4138_413827


namespace NUMINAMATH_CALUDE_rectangle_width_is_fifteen_l4138_413805

/-- Represents a rectangle with length and width in centimeters. -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: For a rectangle where the width is 3 cm longer than the length
    and the perimeter is 54 cm, the width is 15 cm. -/
theorem rectangle_width_is_fifteen (r : Rectangle)
    (h1 : r.width = r.length + 3)
    (h2 : perimeter r = 54) :
    r.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_is_fifteen_l4138_413805


namespace NUMINAMATH_CALUDE_consecutive_square_roots_l4138_413880

theorem consecutive_square_roots (n : ℕ) (h : Real.sqrt n = 3) :
  Real.sqrt (n + 1) = 3 + Real.sqrt 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_square_roots_l4138_413880


namespace NUMINAMATH_CALUDE_candidates_calculation_l4138_413815

theorem candidates_calculation (candidates : ℕ) : 
  (candidates * 7 / 100 = candidates * 6 / 100 + 83) → 
  candidates = 8300 := by
sorry

end NUMINAMATH_CALUDE_candidates_calculation_l4138_413815


namespace NUMINAMATH_CALUDE_subtraction_to_addition_division_to_multiplication_problem_1_problem_2_l4138_413850

-- Problem 1
theorem subtraction_to_addition (a b : ℤ) : a - b = a + (-b) := by sorry

-- Problem 2
theorem division_to_multiplication (a : ℚ) (b : ℚ) (h : b ≠ 0) :
  a / b = a * (1 / b) := by sorry

-- Specific instances
theorem problem_1 : -8 - 5 = -8 + (-5) := by sorry

theorem problem_2 : (1 : ℚ) / 2 / (-2) = (1 : ℚ) / 2 * (-1 / 2) := by sorry

end NUMINAMATH_CALUDE_subtraction_to_addition_division_to_multiplication_problem_1_problem_2_l4138_413850


namespace NUMINAMATH_CALUDE_store_purchase_combinations_l4138_413890

theorem store_purchase_combinations (headphones : ℕ) (mice : ℕ) (keyboards : ℕ) 
  (keyboard_mouse_sets : ℕ) (headphones_mouse_sets : ℕ) : 
  headphones = 9 → mice = 13 → keyboards = 5 → 
  keyboard_mouse_sets = 4 → headphones_mouse_sets = 5 → 
  keyboard_mouse_sets * headphones + 
  headphones_mouse_sets * keyboards + 
  headphones * mice * keyboards = 646 := by
  sorry

end NUMINAMATH_CALUDE_store_purchase_combinations_l4138_413890


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l4138_413878

/-- Calculates the total wet surface area of a rectangular cistern -/
def cisternWetSurfaceArea (length width depth : Real) : Real :=
  let bottomArea := length * width
  let longerSidesArea := 2 * length * depth
  let shorterSidesArea := 2 * width * depth
  bottomArea + longerSidesArea + shorterSidesArea

/-- Theorem: The wet surface area of a cistern with given dimensions is 68.6 square meters -/
theorem cistern_wet_surface_area :
  cisternWetSurfaceArea 7 5 1.40 = 68.6 := by
  sorry

#eval cisternWetSurfaceArea 7 5 1.40

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l4138_413878


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l4138_413841

theorem cylinder_surface_area (r h V : ℝ) : 
  r = 1 → V = 4 * Real.pi → V = Real.pi * r^2 * h → 
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 10 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l4138_413841


namespace NUMINAMATH_CALUDE_inequality_proof_l4138_413830

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4138_413830


namespace NUMINAMATH_CALUDE_mile_to_rod_l4138_413870

-- Define the units
def mile : ℕ := 1
def furlong : ℕ := 1
def rod : ℕ := 1

-- Define the conversion rates
axiom mile_to_furlong : mile = 6 * furlong
axiom furlong_to_rod : furlong = 60 * rod

-- Theorem to prove
theorem mile_to_rod : mile = 360 * rod := by
  sorry

end NUMINAMATH_CALUDE_mile_to_rod_l4138_413870


namespace NUMINAMATH_CALUDE_triangle_sum_equality_l4138_413823

theorem triangle_sum_equality 
  (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = a^2)
  (h2 : y^2 + y*z + z^2 = b^2)
  (h3 : x^2 + x*z + z^2 = c^2) :
  let p := (a + b + c) / 2
  x*y + y*z + x*z = 4 * Real.sqrt ((p*(p-a)*(p-b)*(p-c))/3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_equality_l4138_413823


namespace NUMINAMATH_CALUDE_number_of_divisors_720_l4138_413864

theorem number_of_divisors_720 : Finset.card (Nat.divisors 720) = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_720_l4138_413864


namespace NUMINAMATH_CALUDE_modular_congruence_l4138_413855

theorem modular_congruence (a b n : ℤ) : 
  a % 48 = 25 →
  b % 48 = 80 →
  150 ≤ n →
  n ≤ 191 →
  (a - b) % 48 = n % 48 ↔ n = 185 := by
sorry

end NUMINAMATH_CALUDE_modular_congruence_l4138_413855


namespace NUMINAMATH_CALUDE_city_population_problem_l4138_413859

theorem city_population_problem (p : ℕ) : 
  (p + 800) * 85 / 100 = p - 48 → p = 4853 := by sorry

end NUMINAMATH_CALUDE_city_population_problem_l4138_413859


namespace NUMINAMATH_CALUDE_recordedLineLengthApprox_l4138_413857

/-- Represents the parameters of a record turntable --/
structure TurntableParams where
  revPerMinute : ℝ
  playTime : ℝ
  initialDiameter : ℝ
  finalDiameter : ℝ

/-- Calculates the length of the recorded line on a turntable --/
def recordedLineLength (params : TurntableParams) : ℝ :=
  sorry

/-- The main theorem stating the length of the recorded line --/
theorem recordedLineLengthApprox (params : TurntableParams) 
  (h1 : params.revPerMinute = 100)
  (h2 : params.playTime = 24.5)
  (h3 : params.initialDiameter = 29)
  (h4 : params.finalDiameter = 11.5) :
  abs (recordedLineLength params - 155862.265789099) < 1e-6 := by
  sorry

end NUMINAMATH_CALUDE_recordedLineLengthApprox_l4138_413857


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l4138_413816

theorem mod_equivalence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 21 ∧ 47635 % 21 = n ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l4138_413816


namespace NUMINAMATH_CALUDE_last_two_digits_product_l4138_413851

def last_two_digits (n : ℤ) : ℤ × ℤ :=
  let tens := (n / 10) % 10
  let ones := n % 10
  (tens, ones)

def sum_last_two_digits (n : ℤ) : ℤ :=
  let (tens, ones) := last_two_digits n
  tens + ones

def product_last_two_digits (n : ℤ) : ℤ :=
  let (tens, ones) := last_two_digits n
  tens * ones

theorem last_two_digits_product (n : ℤ) :
  n % 5 = 0 → sum_last_two_digits n = 14 → product_last_two_digits n = 45 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l4138_413851


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4138_413853

/-- Given a geometric sequence with common ratio 2 and sum of first 4 terms equal to 1,
    the sum of the first 8 terms is 17 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 2 * a n) 
    (h2 : a 1 + a 2 + a 3 + a 4 = 1) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4138_413853


namespace NUMINAMATH_CALUDE_no_integer_solution_l4138_413839

theorem no_integer_solution : ∀ (x y : ℤ), x^2 + 4*x - 11 ≠ 8*y := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l4138_413839


namespace NUMINAMATH_CALUDE_original_number_l4138_413846

theorem original_number : ∃ x : ℝ, 3 * (2 * x + 6) = 72 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l4138_413846


namespace NUMINAMATH_CALUDE_percent_calculation_l4138_413860

theorem percent_calculation (a b c d e : ℝ) 
  (h1 : c = 0.25 * a)
  (h2 : c = 0.1 * b)
  (h3 : d = 0.5 * b)
  (h4 : d = 0.2 * e)
  (h5 : e = 0.15 * a)
  (h6 : e = 0.05 * c)
  (h7 : a ≠ 0)
  (h8 : c ≠ 0) :
  (d * b + c * e) / (a * c) = 12.65 := by
  sorry

#check percent_calculation

end NUMINAMATH_CALUDE_percent_calculation_l4138_413860


namespace NUMINAMATH_CALUDE_sum_of_special_integers_l4138_413879

theorem sum_of_special_integers (a b c d e : ℤ) : 
  (a + 1 = b) ∧ (c + 1 = d) ∧ (d + 1 = e) ∧ (a * b = 272) ∧ (c * d * e = 336) →
  a + b + c + d + e = 54 := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_integers_l4138_413879


namespace NUMINAMATH_CALUDE_tunnel_construction_equation_l4138_413865

/-- Represents the tunnel construction scenario -/
def tunnel_construction (x : ℝ) : Prop :=
  let total_length : ℝ := 1280
  let increased_speed : ℝ := 1.4 * x
  let weeks_saved : ℝ := 2
  (total_length - x) / x = (total_length - x) / increased_speed + weeks_saved

theorem tunnel_construction_equation :
  ∀ x : ℝ, x > 0 → tunnel_construction x :=
by
  sorry

end NUMINAMATH_CALUDE_tunnel_construction_equation_l4138_413865


namespace NUMINAMATH_CALUDE_square_not_sum_of_periodic_l4138_413804

-- Define a periodic function
def Periodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem square_not_sum_of_periodic :
  ¬ ∃ (g h : ℝ → ℝ), (Periodic g ∧ Periodic h) ∧ (∀ x : ℝ, x^2 = g x + h x) := by
  sorry

end NUMINAMATH_CALUDE_square_not_sum_of_periodic_l4138_413804


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l4138_413898

theorem quadratic_roots_sum_product (p q : ℝ) :
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ r₁ r₂ : ℝ, r₁ + r₂ = 10 ∧ r₁ * r₂ = 15 ∧ 
      3 * r₁^2 - p * r₁ + q = 0 ∧ 
      3 * r₂^2 - p * r₂ + q = 0)) →
  p = 30 ∧ q = 45 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l4138_413898


namespace NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l4138_413828

/-- Given a geometric sequence of positive numbers where the fifth term is 16 and the eleventh term is 2, 
    prove that the eighth term is 4√2. -/
theorem geometric_sequence_eighth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a > 0) 
  (h2 : r > 0) 
  (h3 : a * r^4 = 16) 
  (h4 : a * r^10 = 2) : 
  a * r^7 = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l4138_413828


namespace NUMINAMATH_CALUDE_abs_equality_implies_geq_one_l4138_413807

theorem abs_equality_implies_geq_one (m : ℝ) : |m - 1| = m - 1 → m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_equality_implies_geq_one_l4138_413807


namespace NUMINAMATH_CALUDE_stairs_problem_l4138_413852

/-- Calculates the number of steps climbed given the number of flights, height per flight, and step height. -/
def steps_climbed (flights : ℕ) (flight_height : ℕ) (step_height : ℕ) : ℕ :=
  (flights * flight_height * 12) / step_height

/-- Theorem: Given 9 flights of stairs, with each flight being 10 feet, and each step being 18 inches, 
    the total number of steps climbed is 60. -/
theorem stairs_problem : steps_climbed 9 10 18 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stairs_problem_l4138_413852


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4138_413836

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (((n : ℝ) - 2) * 180 = n * 160) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l4138_413836


namespace NUMINAMATH_CALUDE_problem_statement_l4138_413813

theorem problem_statement (a b c : ℝ) 
  (h1 : a^2 + a*b = c) 
  (h2 : a*b + b^2 = c + 5) : 
  (2*c + 5 ≥ 0) ∧ 
  (a^2 - b^2 = -5) ∧ 
  (a ≠ b ∧ a ≠ -b) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4138_413813


namespace NUMINAMATH_CALUDE_min_games_prediction_l4138_413874

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  white_rook : ℕ  -- Number of students from "White Rook" school
  black_elephant : ℕ  -- Number of students from "Black Elephant" school
  total_games : ℕ  -- Total number of games to be played

/-- Predicate to check if a tournament setup is valid -/
def valid_tournament (t : ChessTournament) : Prop :=
  t.white_rook * t.black_elephant = t.total_games

/-- The minimum number of games after which one can definitely name a participant -/
def min_games_to_predict (t : ChessTournament) : ℕ :=
  t.total_games - t.black_elephant

/-- Theorem stating the minimum number of games for prediction in the given tournament -/
theorem min_games_prediction (t : ChessTournament) 
  (h_valid : valid_tournament t) 
  (h_white : t.white_rook = 15) 
  (h_black : t.black_elephant = 20) 
  (h_total : t.total_games = 300) : 
  min_games_to_predict t = 280 := by
  sorry

#eval min_games_to_predict { white_rook := 15, black_elephant := 20, total_games := 300 }

end NUMINAMATH_CALUDE_min_games_prediction_l4138_413874


namespace NUMINAMATH_CALUDE_exists_a_b_counterexample_l4138_413821

theorem exists_a_b_counterexample : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by sorry

end NUMINAMATH_CALUDE_exists_a_b_counterexample_l4138_413821


namespace NUMINAMATH_CALUDE_floor_x_eq_1994_minus_n_l4138_413891

def x : ℕ → ℚ
  | 0 => 1994
  | n + 1 => (x n)^2 / (x n + 1)

theorem floor_x_eq_1994_minus_n (n : ℕ) (h : n ≤ 998) :
  ⌊x n⌋ = 1994 - n :=
by sorry

end NUMINAMATH_CALUDE_floor_x_eq_1994_minus_n_l4138_413891


namespace NUMINAMATH_CALUDE_speed_at_40_degrees_l4138_413861

-- Define the relationship between temperature and speed
def temperature (s : ℝ) : ℝ := 5 * s^2 + 20 * s + 15

-- Theorem statement
theorem speed_at_40_degrees : 
  ∃ (s₁ s₂ : ℝ), s₁ ≠ s₂ ∧ temperature s₁ = 40 ∧ temperature s₂ = 40 ∧ 
  ((s₁ = 1 ∧ s₂ = -5) ∨ (s₁ = -5 ∧ s₂ = 1)) := by
  sorry

end NUMINAMATH_CALUDE_speed_at_40_degrees_l4138_413861


namespace NUMINAMATH_CALUDE_decimal_comparisons_l4138_413808

theorem decimal_comparisons : 
  (0.839 < 0.9) ∧ (6.7 > 6.07) ∧ (5.45 = 5.450) := by
  sorry

end NUMINAMATH_CALUDE_decimal_comparisons_l4138_413808


namespace NUMINAMATH_CALUDE_area_ratio_is_three_fiftieths_l4138_413810

/-- A large square subdivided into 25 equal smaller squares -/
structure LargeSquare :=
  (side_length : ℝ)
  (num_subdivisions : ℕ)
  (h_subdivisions : num_subdivisions = 25)

/-- A shaded region formed by connecting midpoints of sides of five smaller squares -/
structure ShadedRegion :=
  (large_square : LargeSquare)
  (num_squares : ℕ)
  (h_num_squares : num_squares = 5)

/-- The ratio of the area of the shaded region to the area of the large square -/
def area_ratio (sr : ShadedRegion) : ℚ :=
  3 / 50

/-- Theorem stating that the area ratio is 3/50 -/
theorem area_ratio_is_three_fiftieths (sr : ShadedRegion) :
  area_ratio sr = 3 / 50 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_three_fiftieths_l4138_413810


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l4138_413838

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 9*x^2 - 16*x > 0 ↔ x < -1 ∨ x > 16 := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l4138_413838


namespace NUMINAMATH_CALUDE_not_power_function_l4138_413848

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x^α

-- Define the specific function
def f (x : ℝ) : ℝ := 2 * x^(1/2)

-- Theorem statement
theorem not_power_function : ¬ isPowerFunction f := by
  sorry

end NUMINAMATH_CALUDE_not_power_function_l4138_413848


namespace NUMINAMATH_CALUDE_parabola_equation_l4138_413856

/-- The standard equation of a parabola with focus (3, 0) and vertex (0, 0) is y² = 12x -/
theorem parabola_equation (x y : ℝ) :
  let focus : ℝ × ℝ := (3, 0)
  let vertex : ℝ × ℝ := (0, 0)
  (x - vertex.1) ^ 2 + (y - vertex.2) ^ 2 = (x - focus.1) ^ 2 + (y - focus.2) ^ 2 →
  y ^ 2 = 12 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l4138_413856


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l4138_413884

/-- The line (a-1)x + ay + 3 = 0 passes through the point (3, -3) for any real a -/
theorem fixed_point_on_line (a : ℝ) : (a - 1) * 3 + a * (-3) + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l4138_413884


namespace NUMINAMATH_CALUDE_max_x_minus_y_l4138_413886

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l4138_413886


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l4138_413899

theorem min_value_of_function (x : ℝ) (h : x > 5/4) :
  (4 * x + 1 / (4 * x - 5)) ≥ 7 := by
  sorry

theorem min_value_attained (x : ℝ) (h : x > 5/4) :
  ∃ x₀ > 5/4, 4 * x₀ + 1 / (4 * x₀ - 5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l4138_413899


namespace NUMINAMATH_CALUDE_simplify_expression_l4138_413844

theorem simplify_expression : (7^3 * (2^5)^3) / ((7^2) * 2^(3*3)) = 448 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4138_413844


namespace NUMINAMATH_CALUDE_rio_persimmon_picking_l4138_413809

/-- Given the conditions of Rio's persimmon picking, calculate the average number of persimmons
    she must pick from each of the last 5 trees to achieve her desired overall average. -/
theorem rio_persimmon_picking (first_pick : ℕ) (first_trees : ℕ) (remaining_trees : ℕ) (desired_avg : ℚ) :
  first_pick = 12 →
  first_trees = 5 →
  remaining_trees = 5 →
  desired_avg = 4 →
  (desired_avg * (first_trees + remaining_trees) - first_pick) / remaining_trees = 28/5 := by
  sorry

end NUMINAMATH_CALUDE_rio_persimmon_picking_l4138_413809


namespace NUMINAMATH_CALUDE_balloon_tank_capacity_l4138_413847

theorem balloon_tank_capacity 
  (num_balloons : ℕ) 
  (air_per_balloon : ℕ) 
  (num_tanks : ℕ) 
  (h1 : num_balloons = 1000)
  (h2 : air_per_balloon = 10)
  (h3 : num_tanks = 20) :
  (num_balloons * air_per_balloon) / num_tanks = 500 := by
  sorry

end NUMINAMATH_CALUDE_balloon_tank_capacity_l4138_413847


namespace NUMINAMATH_CALUDE_rivertown_marching_band_max_members_l4138_413824

theorem rivertown_marching_band_max_members :
  ∀ n : ℕ, 
    (20 * n ≡ 11 [MOD 31]) → 
    (20 * n < 1200) → 
    (∀ m : ℕ, (20 * m ≡ 11 [MOD 31]) → (20 * m < 1200) → (20 * m ≤ 20 * n)) →
    20 * n = 1100 :=
by sorry

end NUMINAMATH_CALUDE_rivertown_marching_band_max_members_l4138_413824


namespace NUMINAMATH_CALUDE_num_paths_equals_1960_l4138_413840

/-- The number of paths from A to D passing through C in a 7x9 grid -/
def num_paths_A_to_D_via_C : ℕ :=
  let grid_width := 7
  let grid_height := 9
  let C_right := 4
  let C_down := 3
  let paths_A_to_C := Nat.choose (C_right + C_down) C_right
  let paths_C_to_D := Nat.choose ((grid_width - C_right) + (grid_height - C_down)) (grid_height - C_down)
  paths_A_to_C * paths_C_to_D

/-- Theorem stating that the number of 15-step paths from A to D passing through C is 1960 -/
theorem num_paths_equals_1960 : num_paths_A_to_D_via_C = 1960 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_equals_1960_l4138_413840


namespace NUMINAMATH_CALUDE_equivalent_annual_rate_l4138_413834

/-- Given an annual interest rate of 8% compounded quarterly, 
    the equivalent constant annual compounding rate is approximately 8.24% -/
theorem equivalent_annual_rate (quarterly_rate : ℝ) (annual_rate : ℝ) (r : ℝ) : 
  quarterly_rate = 0.08 / 4 →
  annual_rate = 0.08 →
  (1 + quarterly_rate)^4 = 1 + r →
  ∀ ε > 0, |r - 0.0824| < ε :=
sorry

end NUMINAMATH_CALUDE_equivalent_annual_rate_l4138_413834


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l4138_413858

/-- The coordinates of a point with respect to the origin are equal to its Cartesian coordinates. -/
theorem point_coordinates_wrt_origin (x y : ℝ) :
  let p : ℝ × ℝ := (x, y)
  p = p := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l4138_413858


namespace NUMINAMATH_CALUDE_smallest_valid_n_l4138_413869

def is_valid (n : ℕ) : Prop :=
  ∃ k : ℕ, 17 * n - 1 = 11 * k

theorem smallest_valid_n :
  ∃ n : ℕ, n > 0 ∧ is_valid n ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬is_valid m :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l4138_413869


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4138_413843

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ x => |x^2 - 4*x + 4| - (3 - x)
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 5) / 2 ∧
              x₂ = (3 - Real.sqrt 5) / 2 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4138_413843


namespace NUMINAMATH_CALUDE_tangent_ellipse_solution_l4138_413820

/-- An ellipse with semi-major axis a and semi-minor axis b that is tangent to a rectangle with area 48 -/
structure TangentEllipse where
  a : ℝ
  b : ℝ
  area_eq : a * b = 12
  a_pos : a > 0
  b_pos : b > 0

/-- The theorem stating that the ellipse with a = 4 and b = 3 satisfies the conditions -/
theorem tangent_ellipse_solution :
  ∃ (e : TangentEllipse), e.a = 4 ∧ e.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ellipse_solution_l4138_413820


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4138_413895

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 9 ∧ x ≠ -7 →
  (5 * x - 3) / (x^2 - 2*x - 63) = (21/8) / (x - 9) + (19/8) / (x + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4138_413895


namespace NUMINAMATH_CALUDE_u_2008_eq_225_l4138_413817

/-- Defines the sequence u_n as described in the problem -/
def u : ℕ → ℕ := sorry

/-- The 2008th term of the sequence is 225 -/
theorem u_2008_eq_225 : u 2008 = 225 := by sorry

end NUMINAMATH_CALUDE_u_2008_eq_225_l4138_413817


namespace NUMINAMATH_CALUDE_females_chose_malt_cheerleader_malt_choice_l4138_413876

/-- Represents the group of cheerleaders -/
structure CheerleaderGroup where
  total : Nat
  males : Nat
  females : Nat
  malt_choosers : Nat
  coke_choosers : Nat
  male_malt_choosers : Nat

/-- The theorem to prove -/
theorem females_chose_malt (group : CheerleaderGroup) : Nat :=
  let female_malt_choosers := group.malt_choosers - group.male_malt_choosers
  female_malt_choosers

/-- The main theorem stating the conditions and the result to prove -/
theorem cheerleader_malt_choice : ∃ (group : CheerleaderGroup), 
  group.total = 26 ∧
  group.males = 10 ∧
  group.females = 16 ∧
  group.malt_choosers = 2 * group.coke_choosers ∧
  group.male_malt_choosers = 6 ∧
  females_chose_malt group = 10 := by
  sorry


end NUMINAMATH_CALUDE_females_chose_malt_cheerleader_malt_choice_l4138_413876


namespace NUMINAMATH_CALUDE_month_treat_cost_l4138_413832

/-- The number of treats given per day -/
def treats_per_day : ℕ := 2

/-- The cost of each treat in dollars -/
def cost_per_treat : ℚ := 1/10

/-- The number of days in the month -/
def days_in_month : ℕ := 30

/-- The total cost of treats for the month -/
def total_cost : ℚ := treats_per_day * days_in_month * cost_per_treat

theorem month_treat_cost : total_cost = 6 := by sorry

end NUMINAMATH_CALUDE_month_treat_cost_l4138_413832


namespace NUMINAMATH_CALUDE_inequality_proof_l4138_413837

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) ≤ 1 / 2 ∧
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) = 1 / 2 ↔ 
   a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4138_413837


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4138_413802

theorem complex_equation_solution (a : ℝ) : 
  (a * Complex.I) / (2 - Complex.I) = 1 - 2 * Complex.I → a = -5 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4138_413802


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4138_413896

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∀ z : ℂ, (1 - i) * z = 1 + i → z = i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4138_413896


namespace NUMINAMATH_CALUDE_players_quit_l4138_413866

def video_game_problem (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  initial_players - (total_lives / lives_per_player)

theorem players_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ)
  (h1 : initial_players = 8)
  (h2 : lives_per_player = 5)
  (h3 : total_lives = 15) :
  video_game_problem initial_players lives_per_player total_lives = 5 := by
  sorry

#eval video_game_problem 8 5 15

end NUMINAMATH_CALUDE_players_quit_l4138_413866


namespace NUMINAMATH_CALUDE_max_a_value_l4138_413822

/-- The quadratic function f(x) = ax^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

/-- The maximum possible value of a for the quadratic function f(x) = ax^2 - ax + 1
    such that |f(x)| ≤ 1 for all x in [0, 1] is 8 -/
theorem max_a_value :
  ∃ (a_max : ℝ), a_max = 8 ∧
  (∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |f a x| ≤ 1) →
               a ≤ a_max) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |f a_max x| ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l4138_413822


namespace NUMINAMATH_CALUDE_lcm_gcd_relation_l4138_413868

theorem lcm_gcd_relation (a b : ℕ) : 
  (Nat.lcm a b = Nat.gcd a b + 19) ↔ 
  ((a = 1 ∧ b = 20) ∨ (a = 20 ∧ b = 1) ∨ 
   (a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨ 
   (a = 19 ∧ b = 38) ∨ (a = 38 ∧ b = 19)) :=
by sorry

end NUMINAMATH_CALUDE_lcm_gcd_relation_l4138_413868


namespace NUMINAMATH_CALUDE_multiple_properties_l4138_413862

theorem multiple_properties (a b : ℤ) 
  (ha : 4 ∣ a) (hb : 8 ∣ b) : 
  (4 ∣ b) ∧ (4 ∣ (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_multiple_properties_l4138_413862


namespace NUMINAMATH_CALUDE_lamp_marked_price_l4138_413863

/-- The marked price of a lamp given initial price, purchase discount, desired gain, and sales discount -/
def marked_price (initial_price : ℚ) (purchase_discount : ℚ) (desired_gain : ℚ) (sales_discount : ℚ) : ℚ :=
  let cost_price := initial_price * (1 - purchase_discount)
  let selling_price := cost_price * (1 + desired_gain)
  selling_price / (1 - sales_discount)

theorem lamp_marked_price :
  marked_price 40 (1/5) (1/4) (3/20) = 800/17 := by
  sorry

end NUMINAMATH_CALUDE_lamp_marked_price_l4138_413863


namespace NUMINAMATH_CALUDE_cookie_bringers_l4138_413829

theorem cookie_bringers (num_brownie_students : ℕ) (brownies_per_student : ℕ)
                        (num_donut_students : ℕ) (donuts_per_student : ℕ)
                        (cookies_per_student : ℕ) (price_per_item : ℚ)
                        (total_raised : ℚ) :
  num_brownie_students = 30 →
  brownies_per_student = 12 →
  num_donut_students = 15 →
  donuts_per_student = 12 →
  cookies_per_student = 24 →
  price_per_item = 2 →
  total_raised = 2040 →
  ∃ (num_cookie_students : ℕ),
    num_cookie_students = 20 ∧
    total_raised = price_per_item * (num_brownie_students * brownies_per_student +
                                     num_cookie_students * cookies_per_student +
                                     num_donut_students * donuts_per_student) :=
by sorry

end NUMINAMATH_CALUDE_cookie_bringers_l4138_413829


namespace NUMINAMATH_CALUDE_train_crossing_time_l4138_413867

/-- Time for a train to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 50 → 
  train_speed_kmh = 360 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l4138_413867


namespace NUMINAMATH_CALUDE_unpainted_area_calculation_l4138_413883

theorem unpainted_area_calculation (board_width1 board_width2 : ℝ) 
  (angle : ℝ) (h1 : board_width1 = 5) (h2 : board_width2 = 8) 
  (h3 : angle = 45 * π / 180) : 
  board_width1 * (board_width2 * Real.sin angle) = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_area_calculation_l4138_413883


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l4138_413831

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![2, 3; 0, -1]) : 
  (B^2)⁻¹ = !![4, 3; 0, 1] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l4138_413831


namespace NUMINAMATH_CALUDE_range_of_shifted_and_translated_function_l4138_413803

/-- Given a function f: ℝ → ℝ with range [1,2], 
    prove that the range of g(x) = f(x+1)-2 is [-1,0] -/
theorem range_of_shifted_and_translated_function 
  (f : ℝ → ℝ) (h : Set.range f = Set.Icc 1 2) :
  Set.range (fun x ↦ f (x + 1) - 2) = Set.Icc (-1) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_shifted_and_translated_function_l4138_413803


namespace NUMINAMATH_CALUDE_circle_radius_in_ellipse_l4138_413854

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 + 6 * y^2 = 8

-- Define the condition of two circles being externally tangent
def externally_tangent_circles (r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 * r^2

-- Define the condition of a circle being internally tangent to the ellipse
def internally_tangent_to_ellipse (r : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_equation x y ∧ (x - r)^2 + y^2 = r^2

-- State the theorem
theorem circle_radius_in_ellipse (r : ℝ) :
  externally_tangent_circles r →
  internally_tangent_to_ellipse r →
  r = Real.sqrt 10 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_radius_in_ellipse_l4138_413854


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l4138_413881

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Theorem statement
theorem fifteenth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 8) (h₃ : a₃ = 13) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 73 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l4138_413881


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l4138_413811

def f (x : ℝ) : ℝ := x^4 - 3*x^3 - 2*x^2 + 6*x + 9

theorem real_roots_of_polynomial :
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l4138_413811


namespace NUMINAMATH_CALUDE_average_student_height_l4138_413849

/-- Calculates the average height of all students given the average heights of males and females and the ratio of males to females. -/
theorem average_student_height
  (avg_female_height : ℝ)
  (avg_male_height : ℝ)
  (male_to_female_ratio : ℝ)
  (h1 : avg_female_height = 170)
  (h2 : avg_male_height = 185)
  (h3 : male_to_female_ratio = 2) :
  (male_to_female_ratio * avg_male_height + avg_female_height) / (male_to_female_ratio + 1) = 180 :=
by sorry

end NUMINAMATH_CALUDE_average_student_height_l4138_413849


namespace NUMINAMATH_CALUDE_five_is_integer_l4138_413814

-- Define the set of natural numbers
def NaturalNumber : Type := ℕ

-- Define the set of integers
def Integer : Type := ℤ

-- Define the property that all natural numbers are integers
axiom all_naturals_are_integers : ∀ (n : NaturalNumber), Integer

-- Define that 5 is a natural number
axiom five_is_natural : NaturalNumber

-- Theorem to prove
theorem five_is_integer : Integer :=
sorry

end NUMINAMATH_CALUDE_five_is_integer_l4138_413814


namespace NUMINAMATH_CALUDE_unique_five_digit_number_exists_l4138_413875

/-- Represents a 5-digit number with different non-zero digits -/
structure FiveDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  d_nonzero : d ≠ 0
  e_nonzero : e ≠ 0
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- Checks if the sum of the shifted additions equals a 7-digit number with all digits A -/
def isValidSum (n : FiveDigitNumber) : Prop :=
  let sum := n.a * 1000000 + n.b * 100000 + n.c * 10000 + n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.b * 100000 + n.c * 10000 + n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.c * 10000 + n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.e * 100 + n.d * 10 + n.b +
             n.d * 10 + n.b +
             n.b
  sum = n.a * 1111111

theorem unique_five_digit_number_exists : ∃! n : FiveDigitNumber, isValidSum n ∧ n.a = 8 ∧ n.b = 4 ∧ n.c = 2 ∧ n.d = 6 ∧ n.e = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_exists_l4138_413875


namespace NUMINAMATH_CALUDE_remainder_333_power_333_mod_11_l4138_413842

theorem remainder_333_power_333_mod_11 : 333^333 % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_333_power_333_mod_11_l4138_413842


namespace NUMINAMATH_CALUDE_fraction_equation_solutions_l4138_413871

theorem fraction_equation_solutions (x : ℝ) : 
  1 / (x^2 + 17*x - 8) + 1 / (x^2 + 4*x - 8) + 1 / (x^2 - 9*x - 8) = 0 ↔ 
  x = 1 ∨ x = -8 ∨ x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solutions_l4138_413871


namespace NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_wire_l4138_413873

theorem min_sum_of_squares (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ 8 := by sorry

theorem min_sum_of_squares_wire :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 4 ∧ x^2 + y^2 = 8 := by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_wire_l4138_413873


namespace NUMINAMATH_CALUDE_last_nonzero_digit_50_factorial_l4138_413812

theorem last_nonzero_digit_50_factorial (n : ℕ) : n = 50 →
  ∃ k : ℕ, (n.factorial : ℤ) ≡ 2 [ZMOD 10^k] ∧ (n.factorial : ℤ) % 10^(k+1) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_last_nonzero_digit_50_factorial_l4138_413812


namespace NUMINAMATH_CALUDE_prob_through_c_is_three_sevenths_l4138_413897

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Calculates the number of paths between two points -/
def numPaths (start finish : Point) : Nat :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The probability of passing through a point when moving from start to finish -/
def probThroughPoint (start mid finish : Point) : Rat :=
  (numPaths start mid * numPaths mid finish : Rat) / numPaths start finish

theorem prob_through_c_is_three_sevenths : 
  let a := Point.mk 0 0
  let b := Point.mk 4 4
  let c := Point.mk 3 2
  probThroughPoint a c b = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_through_c_is_three_sevenths_l4138_413897


namespace NUMINAMATH_CALUDE_five_mondays_in_march_l4138_413818

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a leap year with five Sundays in February -/
structure LeapYearWithFiveSundaysInFebruary :=
  (isLeapYear : Bool)
  (februaryHasFiveSundays : Bool)

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

theorem five_mondays_in_march 
  (year : LeapYearWithFiveSundaysInFebruary) : 
  ∃ (mondayCount : Nat), mondayCount = 5 ∧ 
  (∀ (d : DayOfWeek), d ≠ DayOfWeek.Monday → 
    ∃ (otherCount : Nat), otherCount < 5) :=
by sorry

end NUMINAMATH_CALUDE_five_mondays_in_march_l4138_413818


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l4138_413826

def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + (2*m - 1)*x + m^2 = 0

def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation m x

def range_of_m : Set ℝ :=
  {m : ℝ | m ≤ 1/4}

def roots_relation (m : ℝ) (α β : ℝ) : Prop :=
  quadratic_equation m α ∧ quadratic_equation m β ∧ α ≠ β

theorem quadratic_equation_properties :
  (∀ m : ℝ, has_real_roots m → m ∈ range_of_m) ∧
  (∃ m : ℝ, m = -1 ∧ 
    ∃ α β : ℝ, roots_relation m α β ∧ α^2 + β^2 - α*β = 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l4138_413826


namespace NUMINAMATH_CALUDE_no_common_root_for_specific_quadratics_l4138_413892

theorem no_common_root_for_specific_quadratics
  (a b c d : ℝ)
  (h_order : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ (x : ℝ), (x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_common_root_for_specific_quadratics_l4138_413892


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l4138_413894

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  a = Real.sqrt 6 →
  b = 2 →
  B = 45 * π / 180 →
  Real.tan A * Real.tan C > 1 →
  A + B + C = π →
  (a / Real.sin A = b / Real.sin B) →
  C = 75 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l4138_413894


namespace NUMINAMATH_CALUDE_johns_cloth_cost_l4138_413888

/-- The total cost of cloth for John, given the length and price per metre -/
def total_cost (length : ℝ) (price_per_metre : ℝ) : ℝ :=
  length * price_per_metre

/-- Theorem stating that John's total cost for cloth is $444 -/
theorem johns_cloth_cost : 
  total_cost 9.25 48 = 444 := by
  sorry

end NUMINAMATH_CALUDE_johns_cloth_cost_l4138_413888


namespace NUMINAMATH_CALUDE_square_order_l4138_413833

theorem square_order (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_order_l4138_413833


namespace NUMINAMATH_CALUDE_bill_denomination_proof_l4138_413872

/-- Given the cost of berries, cost of peaches, and the amount of change received,
    prove that the denomination of the bill used equals the sum of these three amounts. -/
theorem bill_denomination_proof 
  (cost_berries : ℚ) 
  (cost_peaches : ℚ) 
  (change_received : ℚ) 
  (h1 : cost_berries = 719/100)
  (h2 : cost_peaches = 683/100)
  (h3 : change_received = 598/100) :
  cost_berries + cost_peaches + change_received = 20 := by
  sorry

end NUMINAMATH_CALUDE_bill_denomination_proof_l4138_413872


namespace NUMINAMATH_CALUDE_stratified_sample_second_year_l4138_413887

theorem stratified_sample_second_year (total_students : ℕ) (second_year_students : ℕ) (sample_size : ℕ) : 
  total_students = 1000 →
  second_year_students = 320 →
  sample_size = 200 →
  (second_year_students * sample_size) / total_students = 64 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_second_year_l4138_413887


namespace NUMINAMATH_CALUDE_sixteen_factorial_digit_sum_l4138_413893

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem sixteen_factorial_digit_sum :
  ∃ (X Y : ℕ),
    X < 10 ∧ Y < 10 ∧
    factorial 16 = 2092200000000 + X * 100000000 + 208960000 + Y * 1000000 ∧
    X + Y = 7 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_factorial_digit_sum_l4138_413893


namespace NUMINAMATH_CALUDE_triangle_height_l4138_413835

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 10 → area = 50 → area = (base * height) / 2 → height = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l4138_413835


namespace NUMINAMATH_CALUDE_great_wall_soldiers_l4138_413882

/-- Calculates the total number of soldiers in beacon towers along a wall -/
def total_soldiers (wall_length : ℕ) (tower_interval : ℕ) (soldiers_per_tower : ℕ) : ℕ :=
  (wall_length / tower_interval) * soldiers_per_tower

/-- Theorem stating that for a wall of 7300 km with towers every 5 km and 2 soldiers per tower, 
    the total number of soldiers is 2920 -/
theorem great_wall_soldiers : 
  total_soldiers 7300 5 2 = 2920 := by
  sorry

end NUMINAMATH_CALUDE_great_wall_soldiers_l4138_413882


namespace NUMINAMATH_CALUDE_c_range_l4138_413845

def P (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) : Prop := ∀ x y : ℝ, 1/2 < x ∧ x < y → (x^2 - 2*c*x + 1) < (y^2 - 2*c*y + 1)

theorem c_range (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) :
  (P c ∨ q c) ∧ ¬(P c ∧ q c) → 1/2 < c ∧ c < 1 := by
  sorry

end NUMINAMATH_CALUDE_c_range_l4138_413845


namespace NUMINAMATH_CALUDE_min_vertical_distance_l4138_413885

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 + 2*x - 1

-- Define the vertical distance between the two functions
def vertical_distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x₀ : ℝ), vertical_distance x₀ = 3/4 ∧
  ∀ (x : ℝ), vertical_distance x ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l4138_413885


namespace NUMINAMATH_CALUDE_boys_percentage_in_class_l4138_413806

theorem boys_percentage_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (boys_ratio * total_students : ℚ) / ((boys_ratio + girls_ratio) * total_students) * 100 = 42857 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_boys_percentage_in_class_l4138_413806


namespace NUMINAMATH_CALUDE_number_of_polynomials_l4138_413825

/-- A function to determine if an expression is a polynomial -/
def isPolynomial (expr : String) : Bool :=
  match expr with
  | "x^2+2" => true
  | "1/a+4" => false
  | "3ab^2/7" => true
  | "ab/c" => false
  | "-5x" => true
  | _ => false

/-- The list of expressions to check -/
def expressions : List String :=
  ["x^2+2", "1/a+4", "3ab^2/7", "ab/c", "-5x"]

/-- Theorem stating that the number of polynomials in the given list is 3 -/
theorem number_of_polynomials :
  (expressions.filter isPolynomial).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_polynomials_l4138_413825


namespace NUMINAMATH_CALUDE_identify_six_genuine_coins_l4138_413819

/-- Represents the result of a weighing on a balance scale -/
inductive WeighResult
| Equal : WeighResult
| LeftHeavier : WeighResult
| RightHeavier : WeighResult

/-- Represents a group of coins -/
structure CoinGroup where
  total : Nat
  genuine : Nat
  counterfeit : Nat

/-- Represents a weighing action on the balance scale -/
def weigh (left right : CoinGroup) : WeighResult :=
  sorry

/-- Represents the process of identifying genuine coins -/
def identifyGenuineCoins (coins : CoinGroup) (maxWeighings : Nat) : Option (Fin 6 → Bool) :=
  sorry

theorem identify_six_genuine_coins :
  ∀ (coins : CoinGroup),
    coins.total = 25 →
    coins.genuine = 22 →
    coins.counterfeit = 3 →
    ∃ (result : Fin 6 → Bool),
      identifyGenuineCoins coins 2 = some result ∧
      (∀ i, result i = true → i.val < 6) :=
by
  sorry

end NUMINAMATH_CALUDE_identify_six_genuine_coins_l4138_413819


namespace NUMINAMATH_CALUDE_winston_initial_gas_l4138_413889

/-- The amount of gas in gallons used for a trip -/
structure Trip where
  gas_used : ℝ

/-- The gas tank of a car -/
structure GasTank where
  capacity : ℝ
  initial_amount : ℝ
  remaining_amount : ℝ

/-- Winston's car trips and gas tank -/
def winston_scenario (store_trip doctor_trip : Trip) (tank : GasTank) : Prop :=
  store_trip.gas_used = 6 ∧
  doctor_trip.gas_used = 2 ∧
  tank.capacity = 12 ∧
  tank.initial_amount = tank.remaining_amount + store_trip.gas_used + doctor_trip.gas_used ∧
  tank.remaining_amount > 0 ∧
  tank.initial_amount ≤ tank.capacity

theorem winston_initial_gas 
  (store_trip doctor_trip : Trip) (tank : GasTank) 
  (h : winston_scenario store_trip doctor_trip tank) : 
  tank.initial_amount = 12 :=
sorry

end NUMINAMATH_CALUDE_winston_initial_gas_l4138_413889


namespace NUMINAMATH_CALUDE_sum_of_evens_l4138_413877

theorem sum_of_evens (n : ℕ) (sum_first_n : ℕ) (first_term : ℕ) (last_term : ℕ) : 
  n = 50 → 
  sum_first_n = 2550 → 
  first_term = 102 → 
  last_term = 200 → 
  (n : ℕ) * (first_term + last_term) / 2 = 7550 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_evens_l4138_413877


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l4138_413801

theorem smallest_lcm_with_gcd_5 :
  ∃ (k l : ℕ), 
    1000 ≤ k ∧ k < 10000 ∧
    1000 ≤ l ∧ l < 10000 ∧
    Nat.gcd k l = 5 ∧
    Nat.lcm k l = 201000 ∧
    ∀ (m n : ℕ), 
      1000 ≤ m ∧ m < 10000 ∧
      1000 ≤ n ∧ n < 10000 ∧
      Nat.gcd m n = 5 →
      Nat.lcm m n ≥ 201000 :=
sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l4138_413801
