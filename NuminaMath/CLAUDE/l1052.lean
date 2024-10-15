import Mathlib

namespace NUMINAMATH_CALUDE_f_neg_two_eq_nine_l1052_105259

/-- The function f(x) = x^5 + ax^3 + x^2 + bx + 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + x^2 + b*x + 2

/-- Theorem: If f(2) = 3, then f(-2) = 9 -/
theorem f_neg_two_eq_nine {a b : ℝ} (h : f a b 2 = 3) : f a b (-2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_nine_l1052_105259


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1052_105286

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := sorry

theorem basketball_team_selection :
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let starters : ℕ := 7
  let quadruplets_in_lineup : ℕ := 3
  
  (choose quadruplets quadruplets_in_lineup) * 
  (choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 1980 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1052_105286


namespace NUMINAMATH_CALUDE_one_mile_in_yards_l1052_105291

-- Define the conversion rates
def mile_to_furlong : ℚ := 5
def furlong_to_rod : ℚ := 50
def rod_to_yard : ℚ := 5

-- Theorem statement
theorem one_mile_in_yards :
  mile_to_furlong * furlong_to_rod * rod_to_yard = 1250 := by
  sorry

end NUMINAMATH_CALUDE_one_mile_in_yards_l1052_105291


namespace NUMINAMATH_CALUDE_part_one_part_two_l1052_105297

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (x : ℝ) :
  let a : ℝ := 1
  (f a x ≥ 4 - |x - 1|) ↔ (x ≤ -1 ∨ x ≥ 3) :=
sorry

-- Part II
theorem part_two (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : ℝ := 1
  (∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  (1/m + 1/(2*n) = a) →
  (∀ k l, k > 0 → l > 0 → 1/k + 1/(2*l) = a → m*n ≤ k*l) →
  m*n = 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1052_105297


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l1052_105270

/-- Proves that given specific conditions, the speed of a car in the first hour is 10 km/h -/
theorem car_speed_first_hour 
  (total_time : ℝ) 
  (second_hour_speed : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_time = 2)
  (h2 : second_hour_speed = 60)
  (h3 : average_speed = 35) : 
  ∃ (first_hour_speed : ℝ), first_hour_speed = 10 ∧ 
    average_speed = (first_hour_speed + second_hour_speed) / total_time :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l1052_105270


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1052_105267

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 60) : 
  2 * a 9 - a 10 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1052_105267


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1052_105206

/-- Given a line passing through points (-3, 8) and (0, -1), prove that the sum of its slope and y-intercept is -4 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, (x = -3 ∧ y = 8) ∨ (x = 0 ∧ y = -1) → y = m * x + b) → 
  m + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1052_105206


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1052_105264

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), (Real.log (x + 5) + Real.log (x - 2) = Real.log (x^2 - 3*x - 10)) ∧ 
              (x + 5 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 3*x - 10 > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1052_105264


namespace NUMINAMATH_CALUDE_base_10_89_equals_base_4_1121_l1052_105258

/-- Converts a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n < 4 then [n]
  else (n % 4) :: toBase4 (n / 4)

/-- Converts a list of base-4 digits to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 4 * acc) 0

/-- Theorem stating that 89 in base 10 is equal to 1121 in base 4 -/
theorem base_10_89_equals_base_4_1121 :
  fromBase4 [1, 2, 1, 1] = 89 := by
  sorry

#eval toBase4 89  -- Should output [1, 2, 1, 1]
#eval fromBase4 [1, 2, 1, 1]  -- Should output 89

end NUMINAMATH_CALUDE_base_10_89_equals_base_4_1121_l1052_105258


namespace NUMINAMATH_CALUDE_special_geometric_sequence_q_values_l1052_105225

/-- A geometric sequence with special properties -/
structure SpecialGeometricSequence where
  a : ℕ+ → ℕ+
  q : ℕ+
  first_term : a 1 = 2^81
  geometric : ∀ n : ℕ+, a (n + 1) = a n * q
  product_closure : ∀ m n : ℕ+, ∃ p : ℕ+, a m * a n = a p

/-- The set of all possible values for the common ratio q -/
def possible_q_values : Set ℕ+ :=
  {2^81, 2^27, 2^9, 2^3, 2}

/-- Main theorem: The set of all possible values of q for a SpecialGeometricSequence -/
theorem special_geometric_sequence_q_values (seq : SpecialGeometricSequence) :
  seq.q ∈ possible_q_values := by
  sorry

end NUMINAMATH_CALUDE_special_geometric_sequence_q_values_l1052_105225


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1052_105292

theorem sin_330_degrees : 
  Real.sin (330 * Real.pi / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1052_105292


namespace NUMINAMATH_CALUDE_right_triangle_area_l1052_105249

theorem right_triangle_area (base height : ℝ) (h1 : base = 15) (h2 : height = 10) :
  (1 / 2) * base * height = 75 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1052_105249


namespace NUMINAMATH_CALUDE_circle_selection_theorem_l1052_105201

/-- A figure with circles arranged in a specific pattern -/
structure CircleFigure where
  total_circles : ℕ
  horizontal_lines : ℕ
  diagonal_directions : ℕ

/-- The number of ways to choose three consecutive circles in a given direction -/
def consecutive_choices (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of ways to choose three consecutive circles in the figure -/
def total_choices (fig : CircleFigure) : ℕ :=
  consecutive_choices fig.horizontal_lines +
  fig.diagonal_directions * consecutive_choices (fig.horizontal_lines - 1)

/-- The main theorem stating the number of ways to choose three consecutive circles -/
theorem circle_selection_theorem (fig : CircleFigure) 
  (h1 : fig.total_circles = 33)
  (h2 : fig.horizontal_lines = 7)
  (h3 : fig.diagonal_directions = 2) :
  total_choices fig = 57 := by
  sorry

end NUMINAMATH_CALUDE_circle_selection_theorem_l1052_105201


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1052_105252

theorem two_numbers_difference (x y : ℝ) : 
  x < y ∧ x + y = 34 ∧ y = 22 → y - x = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1052_105252


namespace NUMINAMATH_CALUDE_line_and_circle_problem_l1052_105209

/-- Line l: x - y + m = 0 -/
def line_l (m : ℝ) (x y : ℝ) : Prop := x - y + m = 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on a line -/
def point_on_line (p : Point) (m : ℝ) : Prop := line_l m p.x p.y

/-- Rotate a line by 90 degrees counterclockwise around its x-axis intersection -/
def rotate_line (m : ℝ) (x y : ℝ) : Prop := y + x + m = 0

/-- Circle equation -/
def circle_equation (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.x)^2 + (y - center.y)^2 = radius^2

theorem line_and_circle_problem (m : ℝ) :
  (∃ (x y : ℝ), rotate_line m x y ∧ x = 2 ∧ y = -3) →
  (∃ (center : Point) (radius : ℝ),
    point_on_line center m ∧
    circle_equation center radius 1 1 ∧
    circle_equation center radius 2 (-2)) →
  m = 1 ∧
  (∃ (center : Point),
    point_on_line center 1 ∧
    circle_equation center 5 1 1 ∧
    circle_equation center 5 2 (-2) ∧
    center.x = -3 ∧
    center.y = -2) := by
  sorry

end NUMINAMATH_CALUDE_line_and_circle_problem_l1052_105209


namespace NUMINAMATH_CALUDE_angle_sum_proof_l1052_105283

theorem angle_sum_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : (1 - Real.tan α) * (1 - Real.tan β) = 2) : α + β = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l1052_105283


namespace NUMINAMATH_CALUDE_tray_height_is_seven_l1052_105260

/-- Represents the dimensions of the rectangular paper --/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Represents the parameters of the cuts made on the paper --/
structure CutParameters where
  distance_from_corner : ℝ
  angle : ℝ

/-- Calculates the height of the tray formed from the paper --/
def tray_height (paper : PaperDimensions) (cut : CutParameters) : ℝ :=
  sorry

/-- Theorem stating that the height of the tray is 7 for the given parameters --/
theorem tray_height_is_seven :
  let paper := PaperDimensions.mk 150 100
  let cut := CutParameters.mk 7 (π / 4)
  tray_height paper cut = 7 := by
  sorry

end NUMINAMATH_CALUDE_tray_height_is_seven_l1052_105260


namespace NUMINAMATH_CALUDE_shyne_plants_l1052_105289

/-- The number of eggplants that can be grown from one seed packet -/
def eggplants_per_packet : ℕ := 14

/-- The number of sunflowers that can be grown from one seed packet -/
def sunflowers_per_packet : ℕ := 10

/-- The number of eggplant seed packets Shyne bought -/
def eggplant_packets : ℕ := 4

/-- The number of sunflower seed packets Shyne bought -/
def sunflower_packets : ℕ := 6

/-- The total number of plants Shyne can grow -/
def total_plants : ℕ := eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets

theorem shyne_plants : total_plants = 116 := by
  sorry

end NUMINAMATH_CALUDE_shyne_plants_l1052_105289


namespace NUMINAMATH_CALUDE_sphere_in_cube_untouchable_area_l1052_105215

/-- The area of a cube's inner surface that a sphere can't touch -/
def untouchableArea (cubeEdge : ℝ) (sphereRadius : ℝ) : ℝ :=
  12 * cubeEdge * sphereRadius - 24 * sphereRadius^2

theorem sphere_in_cube_untouchable_area :
  untouchableArea 5 1 = 96 := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cube_untouchable_area_l1052_105215


namespace NUMINAMATH_CALUDE_nine_possible_values_for_E_l1052_105230

def is_digit (n : ℕ) : Prop := n < 10

theorem nine_possible_values_for_E :
  ∀ (A B C D E : ℕ),
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ is_digit E →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E →
    A + B = E →
    (C + D = E ∨ C + D = E + 10) →
    ∃! (count : ℕ), count = 9 ∧ 
      ∃ (possible_E : Finset ℕ), 
        possible_E.card = count ∧
        (∀ e, e ∈ possible_E ↔ 
          ∃ (A' B' C' D' : ℕ),
            is_digit A' ∧ is_digit B' ∧ is_digit C' ∧ is_digit D' ∧ is_digit e ∧
            A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ A' ≠ e ∧
            B' ≠ C' ∧ B' ≠ D' ∧ B' ≠ e ∧
            C' ≠ D' ∧ C' ≠ e ∧
            D' ≠ e ∧
            A' + B' = e ∧
            (C' + D' = e ∨ C' + D' = e + 10)) :=
by
  sorry

end NUMINAMATH_CALUDE_nine_possible_values_for_E_l1052_105230


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1052_105280

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/4 is 4/3 -/
theorem geometric_series_sum : 
  let a : ℚ := 1
  let r : ℚ := 1/4
  let S : ℚ := ∑' n, a * r^n
  S = 4/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1052_105280


namespace NUMINAMATH_CALUDE_warship_path_safe_l1052_105269

/-- Represents the distance of the reefs from the island in nautical miles -/
def reef_distance : ℝ := 3.8

/-- Represents the distance the warship travels from A to C in nautical miles -/
def travel_distance : ℝ := 8

/-- Represents the angle at which the island is seen from point A (in degrees) -/
def angle_at_A : ℝ := 75

/-- Represents the angle at which the island is seen from point C (in degrees) -/
def angle_at_C : ℝ := 60

/-- Theorem stating that the warship's path is safe from the reefs -/
theorem warship_path_safe :
  ∃ (distance_to_island : ℝ),
    distance_to_island > reef_distance ∧
    distance_to_island = travel_distance * Real.sin ((angle_at_A - angle_at_C) / 2 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_warship_path_safe_l1052_105269


namespace NUMINAMATH_CALUDE_cone_base_radius_l1052_105216

/-- Given a cone with surface area 24π cm² and its lateral surface unfolded is a semicircle,
    the radius of the base circle is 2√2 cm. -/
theorem cone_base_radius (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  (π * r^2 + π * r * l = 24 * π) → 
  (π * l = 2 * π * r) → 
  r = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1052_105216


namespace NUMINAMATH_CALUDE_highest_percentage_increase_city_H_l1052_105253

structure City where
  name : String
  pop1990 : ℕ
  pop2000 : ℕ

def percentageIncrease (c : City) : ℚ :=
  ((c.pop2000 - c.pop1990) : ℚ) / (c.pop1990 : ℚ) * 100

def cities : List City := [
  ⟨"F", 60000, 78000⟩,
  ⟨"G", 80000, 96000⟩,
  ⟨"H", 70000, 91000⟩,
  ⟨"I", 85000, 94500⟩,
  ⟨"J", 95000, 114000⟩
]

theorem highest_percentage_increase_city_H :
  ∃ (c : City), c ∈ cities ∧ c.name = "H" ∧
  ∀ (other : City), other ∈ cities → percentageIncrease c ≥ percentageIncrease other :=
by sorry

end NUMINAMATH_CALUDE_highest_percentage_increase_city_H_l1052_105253


namespace NUMINAMATH_CALUDE_jeff_tennis_games_l1052_105228

/-- Calculates the number of games Jeff wins in tennis given the playing time, scoring rate, and points needed to win a match. -/
theorem jeff_tennis_games (playing_time : ℕ) (scoring_rate : ℕ) (points_per_match : ℕ) : 
  playing_time = 120 ∧ scoring_rate = 5 ∧ points_per_match = 8 → 
  (playing_time / scoring_rate) / points_per_match = 3 := by sorry

end NUMINAMATH_CALUDE_jeff_tennis_games_l1052_105228


namespace NUMINAMATH_CALUDE_lcm_problem_l1052_105299

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : a * b = 2560) :
  Nat.lcm a b = 128 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1052_105299


namespace NUMINAMATH_CALUDE_fibonacci_ratio_difference_bound_l1052_105243

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_ratio_difference_bound (n k : ℕ) (hn : n ≥ 1) (hk : k ≥ 1) :
  |((fibonacci (n + 1) : ℝ) / fibonacci n) - ((fibonacci (k + 1) : ℝ) / fibonacci k)| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_ratio_difference_bound_l1052_105243


namespace NUMINAMATH_CALUDE_nora_fundraiser_solution_l1052_105237

/-- Represents the fundraising problem for Nora's school trip -/
def muffin_fundraiser (target : ℕ) (muffins_per_pack : ℕ) (packs_per_case : ℕ) (price_per_muffin : ℕ) : Prop :=
  ∃ (cases : ℕ),
    cases * (packs_per_case * muffins_per_pack * price_per_muffin) = target

/-- Theorem stating the solution to Nora's fundraising problem -/
theorem nora_fundraiser_solution :
  muffin_fundraiser 120 4 3 2 → (5 : ℕ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_nora_fundraiser_solution_l1052_105237


namespace NUMINAMATH_CALUDE_jacks_initial_dollars_l1052_105271

theorem jacks_initial_dollars (x : ℕ) : 
  x + 36 * 2 = 117 → x = 45 := by sorry

end NUMINAMATH_CALUDE_jacks_initial_dollars_l1052_105271


namespace NUMINAMATH_CALUDE_equal_roots_equation_l1052_105298

theorem equal_roots_equation : ∃ x : ℝ, (x - 1) * (x - 1) = 0 ∧ 
  (∀ y : ℝ, (y - 1) * (y - 1) = 0 → y = x) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_equation_l1052_105298


namespace NUMINAMATH_CALUDE_circle_properties_l1052_105218

noncomputable def circle_equation (x y : ℝ) : ℝ := (x - Real.sqrt 3)^2 + (y - 1)^2

theorem circle_properties :
  ∃ (c : ℝ × ℝ),
    (∀ x y : ℝ, circle_equation x y = 1 → ‖(x, y) - c‖ = 1) ∧
    (∃ x : ℝ, circle_equation x 0 = 1) ∧
    (∃ x y : ℝ, circle_equation x y = 1 ∧ y = Real.sqrt 3 * x) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l1052_105218


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1052_105241

theorem min_value_of_expression (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1052_105241


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1052_105256

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_cond : x + y + z = 2)
  (x_bound : x ≥ -1/2)
  (y_bound : y ≥ -2)
  (z_bound : z ≥ -3)
  (xy_cond : 2*x + y = 1) :
  ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ + y₀ + z₀ = 2 ∧ 
    2*x₀ + y₀ = 1 ∧
    x₀ ≥ -1/2 ∧ 
    y₀ ≥ -2 ∧ 
    z₀ ≥ -3 ∧
    ∀ x y z, 
      x + y + z = 2 → 
      2*x + y = 1 → 
      x ≥ -1/2 → 
      y ≥ -2 → 
      z ≥ -3 →
      Real.sqrt (4*x + 2) + Real.sqrt (3*y + 6) + Real.sqrt (4*z + 12) ≤ 
      Real.sqrt (4*x₀ + 2) + Real.sqrt (3*y₀ + 6) + Real.sqrt (4*z₀ + 12) ∧
      Real.sqrt (4*x₀ + 2) + Real.sqrt (3*y₀ + 6) + Real.sqrt (4*z₀ + 12) = Real.sqrt 68 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1052_105256


namespace NUMINAMATH_CALUDE_solve_system_l1052_105255

theorem solve_system (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1052_105255


namespace NUMINAMATH_CALUDE_percentage_of_three_digit_numbers_with_repeated_digit_l1052_105210

theorem percentage_of_three_digit_numbers_with_repeated_digit : 
  let total_three_digit_numbers : ℕ := 900
  let three_digit_numbers_without_repeat : ℕ := 9 * 9 * 8
  let three_digit_numbers_with_repeat : ℕ := total_three_digit_numbers - three_digit_numbers_without_repeat
  let percentage : ℚ := three_digit_numbers_with_repeat / total_three_digit_numbers
  ⌊percentage * 1000 + 5⌋ / 10 = 28 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_three_digit_numbers_with_repeated_digit_l1052_105210


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l1052_105275

theorem cos_squared_minus_sin_squared_15_deg :
  Real.cos (15 * Real.pi / 180) ^ 2 - Real.sin (15 * Real.pi / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l1052_105275


namespace NUMINAMATH_CALUDE_probability_in_standard_deck_l1052_105288

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- The probability of drawing a red card first and then a black card -/
def probability_red_then_black (d : Deck) : Rat :=
  (d.red_cards * d.black_cards) / (d.cards * (d.cards - 1))

/-- Theorem statement for the probability in a standard 52-card deck -/
theorem probability_in_standard_deck :
  let d : Deck := ⟨52, 26, 26⟩
  probability_red_then_black d = 13 / 51 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_standard_deck_l1052_105288


namespace NUMINAMATH_CALUDE_hannahs_age_l1052_105214

/-- Given the ages of Eliza, Felipe, Gideon, and Hannah, prove Hannah's age -/
theorem hannahs_age 
  (eliza felipe gideon hannah : ℕ)
  (h1 : eliza = felipe - 4)
  (h2 : felipe = gideon + 6)
  (h3 : hannah = gideon + 2)
  (h4 : eliza = 15) :
  hannah = 15 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_age_l1052_105214


namespace NUMINAMATH_CALUDE_video_game_points_l1052_105239

/-- The number of points earned in a video game level --/
def points_earned (total_enemies : ℕ) (enemies_left : ℕ) (points_per_enemy : ℕ) : ℕ :=
  (total_enemies - enemies_left) * points_per_enemy

/-- Theorem: In the given scenario, the player earns 40 points --/
theorem video_game_points : points_earned 7 2 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_video_game_points_l1052_105239


namespace NUMINAMATH_CALUDE_sum_of_digits_base7_squared_expectation_l1052_105246

/-- Sum of digits in base 7 -/
def sum_of_digits_base7 (n : ℕ) : ℕ :=
  sorry

/-- Expected value of a function over a finite range -/
def expected_value {α : Type*} (f : α → ℝ) (range : Finset α) : ℝ :=
  sorry

theorem sum_of_digits_base7_squared_expectation :
  expected_value (λ n => (sum_of_digits_base7 n)^2) (Finset.range (7^20)) = 3680 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_base7_squared_expectation_l1052_105246


namespace NUMINAMATH_CALUDE_prob_at_least_3_hits_l1052_105236

-- Define the probability of hitting the target on a single shot
def p_hit : ℝ := 0.8

-- Define the number of shots
def n_shots : ℕ := 4

-- Define the probability of hitting the target at least 3 times out of 4 shots
def p_at_least_3 : ℝ := 
  (Nat.choose n_shots 3 : ℝ) * p_hit^3 * (1 - p_hit) + p_hit^4

-- Theorem statement
theorem prob_at_least_3_hits : p_at_least_3 = 0.8192 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_3_hits_l1052_105236


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1052_105232

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1052_105232


namespace NUMINAMATH_CALUDE_tangent_slope_perpendicular_lines_l1052_105222

/-- The function f(x) = x^3 + 3ax --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x

/-- The derivative of f(x) --/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 3*a

theorem tangent_slope_perpendicular_lines (a : ℝ) :
  (f_derivative a 1 = 6) ↔ ((-1 : ℝ) * (-a) = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_perpendicular_lines_l1052_105222


namespace NUMINAMATH_CALUDE_highest_score_can_be_less_than_15_l1052_105274

/-- Represents a team in the tournament -/
structure Team :=
  (score : ℕ)

/-- Represents the tournament -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : ℕ)
  (total_games : ℕ)
  (total_points : ℕ)

/-- The tournament satisfies the given conditions -/
def valid_tournament (t : Tournament) : Prop :=
  t.num_teams = 10 ∧
  t.total_games = t.num_teams * (t.num_teams - 1) / 2 ∧
  t.total_points = 3 * t.total_games ∧
  t.teams.card = t.num_teams

/-- The theorem to be proved -/
theorem highest_score_can_be_less_than_15 :
  ∃ (t : Tournament), valid_tournament t ∧
    (∀ team ∈ t.teams, team.score < 15) :=
  sorry

end NUMINAMATH_CALUDE_highest_score_can_be_less_than_15_l1052_105274


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l1052_105290

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (s : ℝ) (r : ℝ) 
  (h_s : s = 10) (h_r : r = 3) : 
  let square_area := s^2
  let triangle_area := 8 * (1/2 * s/2 * (r * Real.sqrt 3))
  let sector_area := 4 * (1/12 * Real.pi * r^2)
  square_area - triangle_area - sector_area = 100 - 60 * Real.sqrt 3 - 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l1052_105290


namespace NUMINAMATH_CALUDE_coffee_savings_l1052_105240

/-- Calculates the savings in daily coffee expenditure after a price increase and consumption reduction -/
theorem coffee_savings (original_coffees : ℕ) (original_price : ℚ) (price_increase : ℚ) : 
  let new_price := original_price * (1 + price_increase)
  let new_coffees := original_coffees / 2
  let original_spending := original_coffees * original_price
  let new_spending := new_coffees * new_price
  original_spending - new_spending = 2 :=
by
  sorry

#check coffee_savings 4 2 (1/2)

end NUMINAMATH_CALUDE_coffee_savings_l1052_105240


namespace NUMINAMATH_CALUDE_rancher_problem_l1052_105268

theorem rancher_problem :
  ∃! (b h : ℕ), b > 0 ∧ h > 0 ∧ 30 * b + 32 * h = 1200 ∧ b > h := by
  sorry

end NUMINAMATH_CALUDE_rancher_problem_l1052_105268


namespace NUMINAMATH_CALUDE_furniture_production_max_profit_l1052_105234

/-- Represents the problem of maximizing profit in furniture production --/
theorem furniture_production_max_profit :
  let x : ℝ := 16  -- Number of sets of type A furniture
  let y : ℝ := -0.3 * x + 80  -- Total profit function
  let time_constraint : Prop := (5/4) * x + (5/3) * (100 - x) ≤ 160  -- Time constraint
  let total_sets : Prop := x + (100 - x) = 100  -- Total number of sets
  let profit_decreasing : Prop := ∀ x₁ x₂, x₁ < x₂ → (-0.3 * x₁ + 80) > (-0.3 * x₂ + 80)  -- Profit decreases as x increases
  
  -- The following conditions hold:
  time_constraint ∧
  total_sets ∧
  profit_decreasing ∧
  (∀ x' : ℝ, x' ≥ 0 → x' ≤ 100 → (5/4) * x' + (5/3) * (100 - x') ≤ 160 → y ≥ -0.3 * x' + 80) →
  
  -- Then the maximum profit is achieved:
  y = 75.2 := by sorry

end NUMINAMATH_CALUDE_furniture_production_max_profit_l1052_105234


namespace NUMINAMATH_CALUDE_first_positive_term_is_26_l1052_105226

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 4 * n - 102

-- Define the property of being the first positive term
def is_first_positive (k : ℕ) : Prop :=
  a k > 0 ∧ ∀ m : ℕ, m < k → a m ≤ 0

-- Theorem statement
theorem first_positive_term_is_26 : is_first_positive 26 := by
  sorry

end NUMINAMATH_CALUDE_first_positive_term_is_26_l1052_105226


namespace NUMINAMATH_CALUDE_hex_B1F4_equals_45556_l1052_105278

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Convert a HexDigit to its decimal value --/
def hexToDecimal (h : HexDigit) : Nat :=
  match h with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Convert a list of HexDigits to its decimal value --/
def hexListToDecimal (hexList : List HexDigit) : Nat :=
  hexList.foldr (fun digit acc => hexToDecimal digit + 16 * acc) 0

theorem hex_B1F4_equals_45556 :
  hexListToDecimal [HexDigit.B, HexDigit.D1, HexDigit.F, HexDigit.D4] = 45556 := by
  sorry

#eval hexListToDecimal [HexDigit.B, HexDigit.D1, HexDigit.F, HexDigit.D4]

end NUMINAMATH_CALUDE_hex_B1F4_equals_45556_l1052_105278


namespace NUMINAMATH_CALUDE_product_congruence_l1052_105212

theorem product_congruence : 66 * 77 * 88 ≡ 16 [ZMOD 25] := by sorry

end NUMINAMATH_CALUDE_product_congruence_l1052_105212


namespace NUMINAMATH_CALUDE_angle_A_measure_l1052_105277

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem angle_A_measure (t : Triangle) (h1 : t.a = 7) (h2 : t.b = 8) (h3 : Real.cos t.B = 1/7) :
  t.A = π/3 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l1052_105277


namespace NUMINAMATH_CALUDE_degree_of_g_given_f_plus_g_l1052_105220

/-- Given two polynomials f and g, where f(x) = -3x^5 + 2x^4 + x^2 - 6 and the degree of f + g is 2, the degree of g is 5. -/
theorem degree_of_g_given_f_plus_g (f g : Polynomial ℝ) : 
  f = -3 * X^5 + 2 * X^4 + X^2 - 6 →
  Polynomial.degree (f + g) = 2 →
  Polynomial.degree g = 5 := by sorry

end NUMINAMATH_CALUDE_degree_of_g_given_f_plus_g_l1052_105220


namespace NUMINAMATH_CALUDE_bacteria_at_8_20_am_l1052_105284

/-- Calculates the bacterial population after a given time period -/
def bacterial_population (initial_population : ℕ) (doubling_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  initial_population * (2 ^ (elapsed_time / doubling_time))

/-- Theorem stating the bacterial population at 8:20 AM -/
theorem bacteria_at_8_20_am : 
  let initial_population : ℕ := 30
  let doubling_time : ℕ := 4  -- in minutes
  let elapsed_time : ℕ := 20  -- in minutes
  bacterial_population initial_population doubling_time elapsed_time = 960 :=
by
  sorry


end NUMINAMATH_CALUDE_bacteria_at_8_20_am_l1052_105284


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1052_105223

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 36)
  (area2 : w * h = 18)
  (area3 : l * h = 12) :
  l * w * h = 36 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1052_105223


namespace NUMINAMATH_CALUDE_total_ways_is_81_l1052_105296

/-- The number of base options available to each student -/
def num_bases : ℕ := 3

/-- The number of students choosing bases -/
def num_students : ℕ := 4

/-- The total number of ways for students to choose bases -/
def total_ways : ℕ := num_bases ^ num_students

/-- Theorem stating that the total number of ways is 81 -/
theorem total_ways_is_81 : total_ways = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_is_81_l1052_105296


namespace NUMINAMATH_CALUDE_section_b_average_weight_l1052_105251

/-- Proves that the average weight of section B is 30 kg given the class composition and weight information -/
theorem section_b_average_weight 
  (num_students_a : ℕ) 
  (num_students_b : ℕ) 
  (avg_weight_a : ℝ) 
  (avg_weight_total : ℝ) :
  num_students_a = 26 →
  num_students_b = 34 →
  avg_weight_a = 50 →
  avg_weight_total = 38.67 →
  (num_students_a * avg_weight_a + num_students_b * 30) / (num_students_a + num_students_b) = avg_weight_total :=
by
  sorry

#eval (26 * 50 + 34 * 30) / (26 + 34) -- Should output approximately 38.67

end NUMINAMATH_CALUDE_section_b_average_weight_l1052_105251


namespace NUMINAMATH_CALUDE_unique_valid_f_l1052_105261

def is_valid_f (f : ℕ → ℕ) : Prop :=
  (∀ m, f m = 1 ↔ m = 1) ∧
  (∀ m n, f (m * n) = f m * f n / f (Nat.gcd m n)) ∧
  (∀ m, (f^[2000]) m = f m)

theorem unique_valid_f :
  ∃! f : ℕ → ℕ, is_valid_f f ∧ ∀ n, f n = n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_f_l1052_105261


namespace NUMINAMATH_CALUDE_smallest_slope_tangent_line_l1052_105294

/-- The function f(x) = x^3 + 3x^2 + 6x - 10 --/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem smallest_slope_tangent_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = f x → (a*x + b*y + c = 0 ↔ y - f x = f' x * (x - x)))  -- Tangent line equation
    ∧ (∀ x₀ : ℝ, f' x₀ ≥ f' (-1))  -- Slope at x = -1 is the smallest
    ∧ a = 3 ∧ b = -1 ∧ c = -11  -- Coefficients of the tangent line equation
:= by sorry

end NUMINAMATH_CALUDE_smallest_slope_tangent_line_l1052_105294


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l1052_105244

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 13*n + 36 ≤ 0 ∧
  n = 9 ∧
  ∀ (m : ℤ), m^2 - 13*m + 36 ≤ 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l1052_105244


namespace NUMINAMATH_CALUDE_abs_f_properties_l1052_105273

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the absolute value function of f
def abs_f (x : ℝ) : ℝ := |f x|

-- Theorem stating the properties of |f(x)|
theorem abs_f_properties :
  (∀ x, abs_f f x ≥ 0) ∧ 
  (∀ x, f x ≥ 0 → abs_f f x = f x) ∧
  (∀ x, f x < 0 → abs_f f x = -f x) :=
by sorry

end NUMINAMATH_CALUDE_abs_f_properties_l1052_105273


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1052_105202

theorem imaginary_part_of_z : Complex.im ((1 + 2 * Complex.I) / (3 - Complex.I)) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1052_105202


namespace NUMINAMATH_CALUDE_sin_cos_power_sum_l1052_105208

theorem sin_cos_power_sum (x : ℝ) (h : 3 * Real.sin x ^ 3 + Real.cos x ^ 3 = 3) :
  Real.sin x ^ 2018 + Real.cos x ^ 2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_power_sum_l1052_105208


namespace NUMINAMATH_CALUDE_line_segment_length_l1052_105235

/-- Given points A, B, C, and D on a line in that order, prove that AC = 1 cm -/
theorem line_segment_length (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order on the line
  (B - A = 2) →                  -- AB = 2 cm
  (D - B = 6) →                  -- BD = 6 cm
  (D - C = 3) →                  -- CD = 3 cm
  (C - A = 1) :=                 -- AC = 1 cm
by sorry

end NUMINAMATH_CALUDE_line_segment_length_l1052_105235


namespace NUMINAMATH_CALUDE_triangle_problem_l1052_105227

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Given conditions
  t.c = 2 ∧ 
  t.A = π / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 / 2 →
  -- Conclusion
  t.a = Real.sqrt 3 ∧ 
  t.b = 1 ∧ 
  t.C = π / 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l1052_105227


namespace NUMINAMATH_CALUDE_elevator_exit_theorem_l1052_105263

/-- The number of ways passengers can exit an elevator -/
def elevator_exit_ways (num_passengers : ℕ) (total_floors : ℕ) (start_floor : ℕ) : ℕ :=
  (total_floors - start_floor + 1) ^ num_passengers

/-- Theorem: 6 passengers exiting an elevator in a 12-story building starting from the 3rd floor -/
theorem elevator_exit_theorem :
  elevator_exit_ways 6 12 3 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_elevator_exit_theorem_l1052_105263


namespace NUMINAMATH_CALUDE_points_on_line_l1052_105254

theorem points_on_line (t : ℝ) :
  let x := Real.sin t ^ 2
  let y := Real.cos t ^ 2
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_points_on_line_l1052_105254


namespace NUMINAMATH_CALUDE_workers_total_earning_l1052_105203

/-- Represents the daily wages and work days of three workers -/
structure Workers where
  a_days : ℕ
  b_days : ℕ
  c_days : ℕ
  c_wage : ℕ
  wage_ratio : Fin 3 → ℕ

/-- Calculates the total earnings of the workers -/
def total_earning (w : Workers) : ℕ :=
  let unit := w.c_wage / w.wage_ratio 2
  let a_wage := unit * w.wage_ratio 0
  let b_wage := unit * w.wage_ratio 1
  a_wage * w.a_days + b_wage * w.b_days + w.c_wage * w.c_days

/-- The main theorem stating the total earning of the workers -/
theorem workers_total_earning : ∃ (w : Workers), 
  w.a_days = 6 ∧ 
  w.b_days = 9 ∧ 
  w.c_days = 4 ∧ 
  w.c_wage = 105 ∧ 
  w.wage_ratio = ![3, 4, 5] ∧
  total_earning w = 1554 := by
  sorry

end NUMINAMATH_CALUDE_workers_total_earning_l1052_105203


namespace NUMINAMATH_CALUDE_positive_difference_54_and_y_l1052_105282

theorem positive_difference_54_and_y (y : ℝ) (h : (54 + y) / 2 = 32) :
  |54 - y| = 44 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_54_and_y_l1052_105282


namespace NUMINAMATH_CALUDE_anns_shopping_cost_l1052_105238

theorem anns_shopping_cost (shorts_count : ℕ) (shorts_price : ℚ)
                            (shoes_count : ℕ) (shoes_price : ℚ)
                            (tops_count : ℕ) (total_cost : ℚ) :
  shorts_count = 5 →
  shorts_price = 7 →
  shoes_count = 2 →
  shoes_price = 10 →
  tops_count = 4 →
  total_cost = 75 →
  ∃ (top_price : ℚ), top_price = 5 ∧
    total_cost = shorts_count * shorts_price + shoes_count * shoes_price + tops_count * top_price :=
by
  sorry


end NUMINAMATH_CALUDE_anns_shopping_cost_l1052_105238


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1052_105279

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 1 - 2 * Real.sin (2 * α) = Real.cos (2 * α)) : 
  Real.cos α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1052_105279


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1052_105217

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    and eccentricity 5/3, its asymptotes are y = ±(4/3)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2) / a^2 = 25 / 9 →
  ∃ k : ℝ, k = 4/3 ∧ (∀ x y : ℝ, y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1052_105217


namespace NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_value_l1052_105287

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  j : ℝ
  k : ℝ
  zeros : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → zeros i ≠ zeros j
  arithmetic_progression : ∃ (b d : ℝ), ∀ i, zeros i = b + d * i.val
  is_zero : ∀ x, x^4 + j * x^2 + k * x + 256 = (x - zeros 0) * (x - zeros 1) * (x - zeros 2) * (x - zeros 3)

/-- The value of j in an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_j_value (p : ArithmeticProgressionPolynomial) : p.j = -40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_value_l1052_105287


namespace NUMINAMATH_CALUDE_digit2012_is_zero_l1052_105200

/-- The sequence of digits obtained by writing positive integers in order -/
def digitSequence : ℕ → ℕ :=
  sorry

/-- The 2012th digit in the sequence -/
def digit2012 : ℕ := digitSequence 2012

theorem digit2012_is_zero : digit2012 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit2012_is_zero_l1052_105200


namespace NUMINAMATH_CALUDE_group_size_problem_l1052_105231

theorem group_size_problem (x : ℕ) : 
  (5 * x + 45 = 7 * x + 3) → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l1052_105231


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1052_105281

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (X^5 - X^3 + X - 1) * (X^3 - X^2 + 1) = (X^2 + X + 1) * q + (-7 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1052_105281


namespace NUMINAMATH_CALUDE_stamp_difference_l1052_105219

theorem stamp_difference (p q : ℕ) (h1 : p * 4 = q * 7) 
  (h2 : (p - 8) * 5 = (q + 8) * 6) : p - q = 8 := by
  sorry

end NUMINAMATH_CALUDE_stamp_difference_l1052_105219


namespace NUMINAMATH_CALUDE_shaded_area_between_squares_l1052_105266

/-- Given a larger square with area 10 cm² and a smaller square with area 4 cm²,
    where the diagonals of the larger square contain the diagonals of the smaller square,
    prove that the area of one of the four identical regions formed between the squares is 1.5 cm². -/
theorem shaded_area_between_squares (larger_square_area smaller_square_area : ℝ)
  (h1 : larger_square_area = 10)
  (h2 : smaller_square_area = 4)
  (h3 : larger_square_area > smaller_square_area)
  (h4 : ∃ (n : ℕ), n = 4 ∧ n * (larger_square_area - smaller_square_area) / n = 1.5) :
  ∃ (shaded_area : ℝ), shaded_area = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_squares_l1052_105266


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1052_105224

theorem quadratic_factorization :
  ∀ x : ℝ, 2 * x^2 + 4 * x - 6 = 2 * (x - 1) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1052_105224


namespace NUMINAMATH_CALUDE_solve_pocket_money_problem_l1052_105207

def pocket_money_problem (initial_money : ℕ) : Prop :=
  let remaining_money := initial_money / 2
  let total_money := remaining_money + 550
  total_money = 1000 ∧ initial_money = 900

theorem solve_pocket_money_problem :
  ∃ (initial_money : ℕ), pocket_money_problem initial_money :=
sorry

end NUMINAMATH_CALUDE_solve_pocket_money_problem_l1052_105207


namespace NUMINAMATH_CALUDE_taco_cost_l1052_105221

-- Define the cost of a taco and an enchilada
variable (T E : ℚ)

-- Define the conditions from the problem
def condition1 : Prop := 2 * T + 3 * E = 390 / 50
def condition2 : Prop := 3 * T + 5 * E = 635 / 50

-- Theorem to prove
theorem taco_cost (h1 : condition1 T E) (h2 : condition2 T E) : T = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_taco_cost_l1052_105221


namespace NUMINAMATH_CALUDE_three_times_x_greater_than_four_l1052_105257

theorem three_times_x_greater_than_four (x : ℝ) : 
  (3 * x > 4) ↔ (∀ y : ℝ, y = 3 * x → y > 4) :=
by sorry

end NUMINAMATH_CALUDE_three_times_x_greater_than_four_l1052_105257


namespace NUMINAMATH_CALUDE_salary_comparison_l1052_105229

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) :
  b = 1.25 * a := by sorry

end NUMINAMATH_CALUDE_salary_comparison_l1052_105229


namespace NUMINAMATH_CALUDE_train_length_l1052_105247

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : Real) (time : Real) : 
  speed = 72 → time = 4.499640028797696 → 
  ∃ (length : Real), abs (length - 89.99280057595392) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1052_105247


namespace NUMINAMATH_CALUDE_one_face_colored_cubes_125_l1052_105276

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  edge_divisions : ℕ
  total_small_cubes : ℕ
  colored_faces : ℕ

/-- The number of small cubes with exactly one colored face -/
def one_face_colored_cubes (c : CutCube) : ℕ :=
  c.colored_faces * (c.edge_divisions - 2) ^ 2

/-- Theorem stating the number of cubes with one colored face for a specific case -/
theorem one_face_colored_cubes_125 :
  ∀ c : CutCube,
    c.edge_divisions = 5 →
    c.total_small_cubes = 125 →
    c.colored_faces = 6 →
    one_face_colored_cubes c = 54 := by
  sorry

end NUMINAMATH_CALUDE_one_face_colored_cubes_125_l1052_105276


namespace NUMINAMATH_CALUDE_unique_solution_system_l1052_105248

theorem unique_solution_system (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  x^2 + y^2 + x*y = 7 →
  x^2 + z^2 + x*z = 13 →
  y^2 + z^2 + y*z = 19 →
  x = 1 ∧ y = 2 ∧ z = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1052_105248


namespace NUMINAMATH_CALUDE_chips_left_uneaten_l1052_105205

def cookies_per_dozen : ℕ := 12
def dozens_made : ℕ := 4
def chips_per_cookie : ℕ := 7
def fraction_eaten : ℚ := 1/2

theorem chips_left_uneaten : 
  (dozens_made * cookies_per_dozen * chips_per_cookie) * (1 - fraction_eaten) = 168 := by
  sorry

end NUMINAMATH_CALUDE_chips_left_uneaten_l1052_105205


namespace NUMINAMATH_CALUDE_xiaogangMathScore_l1052_105211

theorem xiaogangMathScore (chineseScore englishScore averageScore : ℕ) (mathScore : ℕ) :
  chineseScore = 88 →
  englishScore = 91 →
  averageScore = 90 →
  (chineseScore + mathScore + englishScore) / 3 = averageScore →
  mathScore = 91 := by
  sorry

end NUMINAMATH_CALUDE_xiaogangMathScore_l1052_105211


namespace NUMINAMATH_CALUDE_min_distance_intersection_points_l1052_105242

open Real

theorem min_distance_intersection_points (a : ℝ) :
  let f (x : ℝ) := (x - exp x - 3) / 2
  ∃ (x₁ x₂ : ℝ), a = 2 * x₁ - 3 ∧ a = x₂ + exp x₂ ∧ 
    ∀ (y₁ y₂ : ℝ), a = 2 * y₁ - 3 → a = y₂ + exp y₂ → 
      |x₂ - x₁| ≤ |y₂ - y₁| ∧ |x₂ - x₁| = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_intersection_points_l1052_105242


namespace NUMINAMATH_CALUDE_find_c_l1052_105295

theorem find_c (a b c : ℝ) : 
  (∀ x, (x + 3) * (x + b) = x^2 + c*x + 15) → c = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_c_l1052_105295


namespace NUMINAMATH_CALUDE_circumcircle_equation_l1052_105245

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define point P
def P : ℝ × ℝ := (4, 2)

-- Define that P is outside C
axiom P_outside_C : P ∉ C

-- Define that there are two tangent points A and B
axiom tangent_points_exist : ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ A ≠ B

-- Define the circumcircle of triangle ABP
def circumcircle (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 5}

-- Theorem statement
theorem circumcircle_equation (A B : ℝ × ℝ) 
  (h1 : A ∈ C) (h2 : B ∈ C) (h3 : A ≠ B) :
  circumcircle A B = {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 5} :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l1052_105245


namespace NUMINAMATH_CALUDE_honor_students_count_l1052_105293

theorem honor_students_count 
  (total_students : ℕ) 
  (girls : ℕ) 
  (boys : ℕ) 
  (honor_girls : ℕ) 
  (honor_boys : ℕ) : 
  total_students < 30 →
  total_students = girls + boys →
  (honor_girls : ℚ) / girls = 3 / 13 →
  (honor_boys : ℚ) / boys = 4 / 11 →
  honor_girls + honor_boys = 7 :=
by sorry

end NUMINAMATH_CALUDE_honor_students_count_l1052_105293


namespace NUMINAMATH_CALUDE_race_finish_times_l1052_105262

/-- Represents the time difference at the finish line between two runners -/
def time_difference (distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  distance * (speed2 - speed1)

theorem race_finish_times (malcolm_speed joshua_speed alice_speed : ℝ) 
  (h1 : malcolm_speed = 5)
  (h2 : joshua_speed = 7)
  (h3 : alice_speed = 6)
  (race_distance : ℝ)
  (h4 : race_distance = 12) :
  time_difference race_distance malcolm_speed joshua_speed = 24 ∧
  time_difference race_distance malcolm_speed alice_speed = 12 :=
by sorry

end NUMINAMATH_CALUDE_race_finish_times_l1052_105262


namespace NUMINAMATH_CALUDE_inverse_proportion_m_value_l1052_105285

theorem inverse_proportion_m_value : 
  ∃! m : ℝ, m^2 - 5 = -1 ∧ m + 2 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_m_value_l1052_105285


namespace NUMINAMATH_CALUDE_grandmas_apples_l1052_105204

/-- The problem of Grandma's apple purchase --/
theorem grandmas_apples :
  ∀ (tuesday_price : ℝ) (tuesday_kg : ℝ) (saturday_kg : ℝ),
    tuesday_kg > 0 →
    tuesday_price > 0 →
    tuesday_price * tuesday_kg = 20 →
    saturday_kg = 1.5 * tuesday_kg →
    (tuesday_price - 1) * saturday_kg = 24 →
    saturday_kg = 6 := by
  sorry


end NUMINAMATH_CALUDE_grandmas_apples_l1052_105204


namespace NUMINAMATH_CALUDE_volume_of_one_gram_l1052_105233

/-- Given a substance with the following properties:
  - The mass of 1 cubic meter of the substance is 200 kg.
  - 1 kg = 1,000 grams
  - 1 cubic meter = 1,000,000 cubic centimeters
  
  This theorem proves that the volume of 1 gram of this substance is 5 cubic centimeters. -/
theorem volume_of_one_gram (mass_per_cubic_meter : ℝ) (kg_to_g : ℝ) (cubic_meter_to_cubic_cm : ℝ) :
  mass_per_cubic_meter = 200 →
  kg_to_g = 1000 →
  cubic_meter_to_cubic_cm = 1000000 →
  (1 : ℝ) / (mass_per_cubic_meter * kg_to_g / cubic_meter_to_cubic_cm) = 5 := by
  sorry

#check volume_of_one_gram

end NUMINAMATH_CALUDE_volume_of_one_gram_l1052_105233


namespace NUMINAMATH_CALUDE_problem_solution_l1052_105250

theorem problem_solution (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7) 
  (h2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1052_105250


namespace NUMINAMATH_CALUDE_doctors_visit_cost_l1052_105265

theorem doctors_visit_cost (cast_cost insurance_coverage out_of_pocket : ℝ) :
  cast_cost = 200 →
  insurance_coverage = 0.6 →
  out_of_pocket = 200 →
  ∃ (visit_cost : ℝ),
    visit_cost = 300 ∧
    out_of_pocket = (1 - insurance_coverage) * (visit_cost + cast_cost) :=
by sorry

end NUMINAMATH_CALUDE_doctors_visit_cost_l1052_105265


namespace NUMINAMATH_CALUDE_f_composition_negative_one_l1052_105213

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_composition_negative_one : f (f (-1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_one_l1052_105213


namespace NUMINAMATH_CALUDE_maggie_candy_count_l1052_105272

/-- Given the Halloween candy collection scenario, prove that Maggie collected 50 pieces of candy. -/
theorem maggie_candy_count :
  -- Harper collected 30% more candy than Maggie
  ∀ (maggie harper : ℕ), harper = (13 * maggie) / 10 →
  -- Neil collected 40% more candy than Harper
  ∀ (neil : ℕ), neil = (14 * harper) / 10 →
  -- Neil got 91 pieces of candy
  neil = 91 →
  -- Maggie collected 50 pieces of candy
  maggie = 50 := by
sorry

end NUMINAMATH_CALUDE_maggie_candy_count_l1052_105272
