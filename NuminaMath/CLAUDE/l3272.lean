import Mathlib

namespace NUMINAMATH_CALUDE_trees_not_replanted_l3272_327201

/-- 
Given a track with trees planted every 4 meters along its 48-meter length,
prove that when replanting trees every 6 meters, 5 trees do not need to be replanted.
-/
theorem trees_not_replanted (track_length : ℕ) (initial_spacing : ℕ) (new_spacing : ℕ) : 
  track_length = 48 → initial_spacing = 4 → new_spacing = 6 → 
  (track_length / Nat.lcm initial_spacing new_spacing) + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_trees_not_replanted_l3272_327201


namespace NUMINAMATH_CALUDE_inequality_preservation_l3272_327225

theorem inequality_preservation (a b c : ℝ) (h : a > b) : c - a < c - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3272_327225


namespace NUMINAMATH_CALUDE_f_of_3_equals_41_l3272_327230

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

theorem f_of_3_equals_41 : f 3 = 41 := by sorry

end NUMINAMATH_CALUDE_f_of_3_equals_41_l3272_327230


namespace NUMINAMATH_CALUDE_investment_profit_l3272_327287

/-- The daily price increase rate of the shares -/
def daily_increase : ℝ := 1.1

/-- The amount spent on shares each day in rubles -/
def daily_investment : ℝ := 1000

/-- The number of days the businessman buys shares -/
def investment_days : ℕ := 3

/-- The number of days until the shares are sold -/
def total_days : ℕ := 4

/-- Calculate the total profit from the share investment -/
def calculate_profit : ℝ :=
  let total_investment := daily_investment * investment_days
  let total_value := daily_investment * (daily_increase^3 + daily_increase^2 + daily_increase)
  total_value - total_investment

theorem investment_profit :
  calculate_profit = 641 := by sorry

end NUMINAMATH_CALUDE_investment_profit_l3272_327287


namespace NUMINAMATH_CALUDE_smithtown_left_handed_women_percentage_l3272_327292

/-- Represents the population of Smithtown -/
structure Population where
  right_handed : ℕ  -- Number of right-handed people
  left_handed : ℕ   -- Number of left-handed people
  men : ℕ           -- Number of men
  women : ℕ         -- Number of women

/-- The conditions of the Smithtown population problem -/
def smithtown_conditions (p : Population) : Prop :=
  -- Ratio of right-handed to left-handed is 3:1
  p.right_handed = 3 * p.left_handed ∧
  -- Ratio of men to women is 3:2
  3 * p.women = 2 * p.men ∧
  -- Number of right-handed men is maximized (all men are right-handed)
  p.men = p.right_handed

/-- The theorem stating that 25% of the population are left-handed women -/
theorem smithtown_left_handed_women_percentage (p : Population)
  (h : smithtown_conditions p) :
  (p.left_handed : ℚ) / (p.right_handed + p.left_handed : ℚ) = 1/4 := by
  sorry

#check smithtown_left_handed_women_percentage

end NUMINAMATH_CALUDE_smithtown_left_handed_women_percentage_l3272_327292


namespace NUMINAMATH_CALUDE_equation_solution_l3272_327253

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => 1/3 + 1/x + 1/(x^2)
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 33) / 4 ∧ 
              x₂ = (3 - Real.sqrt 33) / 4 ∧ 
              f x₁ = 1 ∧ 
              f x₂ = 1 ∧ 
              ∀ x : ℝ, f x = 1 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3272_327253


namespace NUMINAMATH_CALUDE_f_properties_l3272_327272

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), 0 < x → x ≤ π / 3 → 2 ≤ f x ∧ f x ≤ 3) ∧
  (∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ ≤ π / 3 ∧ 0 < x₂ ∧ x₂ ≤ π / 3 ∧ f x₁ = 2 ∧ f x₂ = 3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3272_327272


namespace NUMINAMATH_CALUDE_cookfire_logs_proof_l3272_327293

/-- The number of logs burned per hour -/
def logs_burned_per_hour : ℕ := 3

/-- The number of logs added at the end of each hour -/
def logs_added_per_hour : ℕ := 2

/-- The number of hours the cookfire burns -/
def burn_duration : ℕ := 3

/-- The number of logs left after the burn duration -/
def logs_remaining : ℕ := 3

/-- The initial number of logs in the cookfire -/
def initial_logs : ℕ := 6

theorem cookfire_logs_proof :
  initial_logs - burn_duration * logs_burned_per_hour + (burn_duration - 1) * logs_added_per_hour = logs_remaining :=
by
  sorry

end NUMINAMATH_CALUDE_cookfire_logs_proof_l3272_327293


namespace NUMINAMATH_CALUDE_alpo4_molecular_weight_l3272_327277

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Phosphorus in g/mol -/
def atomic_weight_P : ℝ := 30.97

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of AlPO4 in g/mol -/
def molecular_weight_AlPO4 : ℝ :=
  atomic_weight_Al + atomic_weight_P + 4 * atomic_weight_O

/-- Theorem stating that the molecular weight of AlPO4 is 121.95 g/mol -/
theorem alpo4_molecular_weight :
  molecular_weight_AlPO4 = 121.95 := by sorry

end NUMINAMATH_CALUDE_alpo4_molecular_weight_l3272_327277


namespace NUMINAMATH_CALUDE_three_digit_numbers_theorem_l3272_327296

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digits (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (a, b, c)

def satisfies_condition (n : ℕ) : Prop :=
  let (a, b, c) := digits n
  a ≠ 0 ∧ (26 ∣ (a^2 + b^2 + c^2))

def valid_numbers : Finset ℕ :=
  {100, 110, 101, 320, 302, 230, 203, 510, 501, 150, 105}

theorem three_digit_numbers_theorem :
  ∀ n : ℕ, is_three_digit n → (satisfies_condition n ↔ n ∈ valid_numbers) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_theorem_l3272_327296


namespace NUMINAMATH_CALUDE_tan_roots_problem_l3272_327205

open Real

theorem tan_roots_problem (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : (tan α)^2 - 5*(tan α) + 6 = 0) (h4 : (tan β)^2 - 5*(tan β) + 6 = 0) :
  (α + β = 3*π/4) ∧ ¬∃(x : Real), tan (2*(α + β)) = x := by
  sorry

end NUMINAMATH_CALUDE_tan_roots_problem_l3272_327205


namespace NUMINAMATH_CALUDE_transform_second_to_third_l3272_327212

/-- A point in the 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the second quadrant. -/
def isInSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Transforms a point according to the given rule. -/
def transformPoint (p : Point2D) : Point2D :=
  ⟨3 * p.x - 2, -p.y⟩

/-- Determines if a point is in the third quadrant. -/
def isInThirdQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- 
Theorem: If a point is in the second quadrant, 
then its transformed point is in the third quadrant.
-/
theorem transform_second_to_third (p : Point2D) :
  isInSecondQuadrant p → isInThirdQuadrant (transformPoint p) := by
  sorry

end NUMINAMATH_CALUDE_transform_second_to_third_l3272_327212


namespace NUMINAMATH_CALUDE_car_travel_inequality_l3272_327239

/-- Represents the daily distance traveled by a car -/
def daily_distance : ℝ → ℝ
| x => x + 19

/-- Represents the total distance traveled in 8 days -/
def total_distance (x : ℝ) : ℝ := 8 * (daily_distance x)

/-- Theorem stating the inequality representing the car's travel -/
theorem car_travel_inequality (x : ℝ) :
  total_distance x > 2200 ↔ 8 * (x + 19) > 2200 := by sorry

end NUMINAMATH_CALUDE_car_travel_inequality_l3272_327239


namespace NUMINAMATH_CALUDE_max_quarters_is_thirteen_l3272_327263

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The total amount Carlos has in dollars -/
def total_amount : ℚ := 545 / 100

/-- 
Given $5.45 in U.S. coins, with an equal number of quarters, dimes, and nickels,
prove that the maximum number of quarters (and thus dimes and nickels) is 13.
-/
theorem max_quarters_is_thirteen :
  ∃ (n : ℕ), n * (quarter_value + dime_value + nickel_value) ≤ total_amount ∧
             ∀ (m : ℕ), m * (quarter_value + dime_value + nickel_value) ≤ total_amount → m ≤ n ∧
             n = 13 :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_is_thirteen_l3272_327263


namespace NUMINAMATH_CALUDE_sum_distinct_f_values_l3272_327290

def f (x : ℤ) : ℤ := x^2 - 4*x + 100

def sum_distinct_values : ℕ → ℤ
  | 0 => 0
  | n + 1 => sum_distinct_values n + f (n + 1)

theorem sum_distinct_f_values : 
  sum_distinct_values 100 - f 1 = 328053 := by sorry

end NUMINAMATH_CALUDE_sum_distinct_f_values_l3272_327290


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3272_327268

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geometric_mean : 3 = Real.sqrt (3^a * 3^b)) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h_xy : 3 = Real.sqrt (3^x * 3^y)), 1/x + 1/y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3272_327268


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l3272_327265

/-- The remainder when the sum of an arithmetic sequence is divided by 8 -/
theorem arithmetic_sequence_sum_remainder (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) : 
  a₁ = 3 → d = 6 → aₙ = 309 → n * (a₁ + aₙ) % 16 = 8 → 
  (n * (a₁ + aₙ) / 2) % 8 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l3272_327265


namespace NUMINAMATH_CALUDE_exists_valid_configuration_l3272_327283

/-- Represents a point on the chessboard -/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- A configuration of 16 points on the chessboard -/
def Configuration := Fin 16 → Point

/-- Checks if a configuration is valid (no three points are collinear) -/
def valid_configuration (config : Configuration) : Prop :=
  ∀ i j k, i < j → j < k → ¬collinear (config i) (config j) (config k)

/-- Theorem: There exists a valid configuration of 16 points on an 8x8 chessboard -/
theorem exists_valid_configuration : ∃ (config : Configuration), valid_configuration config := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_configuration_l3272_327283


namespace NUMINAMATH_CALUDE_line_equation_through_two_points_l3272_327284

/-- The equation of a line passing through two points -/
theorem line_equation_through_two_points 
  (x₁ y₁ x₂ y₂ x y : ℝ) : 
  (x - x₁) * (y₂ - y₁) = (y - y₁) * (x₂ - x₁) ↔ 
  (x₁ = x₂ ∧ y₁ = y₂) ∨ 
  (∃ (t : ℝ), x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_two_points_l3272_327284


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l3272_327245

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let sector_arc_length := π * r
  let cone_base_radius := sector_arc_length / (2 * π)
  let cone_slant_height := r
  let cone_height := Real.sqrt (cone_slant_height^2 - cone_base_radius^2)
  let cone_volume := (1/3) * π * cone_base_radius^2 * cone_height
  cone_volume = 3 * π * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l3272_327245


namespace NUMINAMATH_CALUDE_boys_passed_exam_l3272_327264

theorem boys_passed_exam (total_boys : ℕ) (avg_all : ℚ) (avg_passed : ℚ) (avg_failed : ℚ) 
  (h_total : total_boys = 120)
  (h_avg_all : avg_all = 37)
  (h_avg_passed : avg_passed = 39)
  (h_avg_failed : avg_failed = 15) :
  ∃ (passed_boys : ℕ), 
    passed_boys = 110 ∧ 
    (passed_boys : ℚ) * avg_passed + (total_boys - passed_boys : ℚ) * avg_failed = 
      (total_boys : ℚ) * avg_all :=
by
  sorry

end NUMINAMATH_CALUDE_boys_passed_exam_l3272_327264


namespace NUMINAMATH_CALUDE_paperclip_production_l3272_327260

/-- Given that 8 identical machines can produce 560 paperclips per minute,
    prove that 12 machines running at the same rate will produce 5040 paperclips in 6 minutes. -/
theorem paperclip_production 
  (rate : ℕ → ℕ → ℕ) -- rate function: number of machines → minutes → number of paperclips
  (h1 : rate 8 1 = 560) -- 8 machines produce 560 paperclips in 1 minute
  (h2 : ∀ n m, rate n m = n * rate 1 m) -- machines work at the same rate
  (h3 : ∀ n m k, rate n (m * k) = k * rate n m) -- linear scaling with time
  : rate 12 6 = 5040 :=
by sorry

end NUMINAMATH_CALUDE_paperclip_production_l3272_327260


namespace NUMINAMATH_CALUDE_transform_negative_expression_l3272_327252

theorem transform_negative_expression (a b c : ℝ) :
  -(a - b + c) = -a + b - c := by sorry

end NUMINAMATH_CALUDE_transform_negative_expression_l3272_327252


namespace NUMINAMATH_CALUDE_exactly_one_root_l3272_327280

-- Define the function f(x) = -x^3 - x
def f (x : ℝ) : ℝ := -x^3 - x

-- State the theorem
theorem exactly_one_root (m n : ℝ) (h_interval : m ≤ n) (h_product : f m * f n < 0) :
  ∃! x, m ≤ x ∧ x ≤ n ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_root_l3272_327280


namespace NUMINAMATH_CALUDE_circular_garden_radius_l3272_327203

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 3) * π * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l3272_327203


namespace NUMINAMATH_CALUDE_parallelogram_properties_l3272_327206

/-- Represents a parallelogram with specific properties -/
structure Parallelogram where
  /-- Length of the shorter side -/
  side_short : ℝ
  /-- Length of the longer side -/
  side_long : ℝ
  /-- Length of the first diagonal -/
  diag1 : ℝ
  /-- Length of the second diagonal -/
  diag2 : ℝ
  /-- The difference between the lengths of the sides is 7 -/
  side_diff : side_long - side_short = 7
  /-- A perpendicular from a vertex divides a diagonal into segments of 6 and 15 -/
  diag_segments : diag1 = 6 + 15

/-- Theorem stating the properties of the specific parallelogram -/
theorem parallelogram_properties : 
  ∃ (p : Parallelogram), 
    p.side_short = 10 ∧ 
    p.side_long = 17 ∧ 
    p.diag1 = 21 ∧ 
    p.diag2 = Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_properties_l3272_327206


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3272_327286

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3272_327286


namespace NUMINAMATH_CALUDE_root_implies_a_range_l3272_327237

def f (a x : ℝ) : ℝ := 2 * a * x^2 + 2 * x - 3 - a

theorem root_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 0) →
  a ≥ 1 ∨ a ≤ -(3 + Real.sqrt 7) / 2 :=
by sorry

end NUMINAMATH_CALUDE_root_implies_a_range_l3272_327237


namespace NUMINAMATH_CALUDE_full_spots_count_l3272_327270

/-- Represents a parking garage with information about its levels and parking spots. -/
structure ParkingGarage where
  levels : Nat
  spots_per_level : Nat
  open_spots_first : Nat
  open_spots_second : Nat
  open_spots_third : Nat
  open_spots_fourth : Nat

/-- Calculates the number of full parking spots in the garage. -/
def full_spots (garage : ParkingGarage) : Nat :=
  garage.levels * garage.spots_per_level -
  (garage.open_spots_first + garage.open_spots_second + garage.open_spots_third + garage.open_spots_fourth)

/-- Theorem stating the number of full parking spots in the given garage configuration. -/
theorem full_spots_count (garage : ParkingGarage) 
  (h1 : garage.levels = 4)
  (h2 : garage.spots_per_level = 100)
  (h3 : garage.open_spots_first = 58)
  (h4 : garage.open_spots_second = garage.open_spots_first + 2)
  (h5 : garage.open_spots_third = garage.open_spots_second + 5)
  (h6 : garage.open_spots_fourth = 31) :
  full_spots garage = 186 := by
  sorry

end NUMINAMATH_CALUDE_full_spots_count_l3272_327270


namespace NUMINAMATH_CALUDE_population_average_age_l3272_327295

theorem population_average_age
  (ratio_women_men : ℚ)
  (avg_age_women : ℚ)
  (avg_age_men : ℚ)
  (h_ratio : ratio_women_men = 7 / 5)
  (h_women_age : avg_age_women = 40)
  (h_men_age : avg_age_men = 30) :
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 215 / 6 :=
by sorry

end NUMINAMATH_CALUDE_population_average_age_l3272_327295


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3272_327216

/-- The focal length of a hyperbola with equation x²/a² - y² = 1,
    where one of its asymptotes is perpendicular to the line 3x + y + 1 = 0,
    is equal to 2√10. -/
theorem hyperbola_focal_length (a : ℝ) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 = 1) →
  (∃ (m : ℝ), m * (-1/3) = -1 ∧ y = m * x) →
  2 * Real.sqrt (1 + a^2) = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3272_327216


namespace NUMINAMATH_CALUDE_smallest_n_with_75_divisors_l3272_327227

def is_multiple_of_75 (n : ℕ) : Prop := ∃ k : ℕ, n = 75 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_n_with_75_divisors :
  ∃ n : ℕ, 
    is_multiple_of_75 n ∧ 
    count_divisors n = 75 ∧ 
    (∀ m : ℕ, m < n → ¬(is_multiple_of_75 m ∧ count_divisors m = 75)) ∧
    n / 75 = 432 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_75_divisors_l3272_327227


namespace NUMINAMATH_CALUDE_probability_open_path_correct_l3272_327278

/-- The probability of being able to go from the first to the last floor using only open doors -/
def probability_open_path (n : ℕ) : ℚ :=
  (2 ^ (n - 1 : ℕ)) / (Nat.choose (2 * (n - 1)) (n - 1))

/-- Theorem stating the probability of an open path in a building with n floors -/
theorem probability_open_path_correct (n : ℕ) (h : n > 1) :
  probability_open_path n = (2 ^ (n - 1 : ℕ)) / (Nat.choose (2 * (n - 1)) (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_probability_open_path_correct_l3272_327278


namespace NUMINAMATH_CALUDE_distance_between_foci_l3272_327238

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (4, 5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem stating the distance between foci
theorem distance_between_foci :
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l3272_327238


namespace NUMINAMATH_CALUDE_composite_number_division_l3272_327229

def first_seven_composite_product : ℕ := 4 * 6 * 8 * 9 * 10 * 12 * 14
def next_eight_composite_product : ℕ := 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25

theorem composite_number_division :
  (first_seven_composite_product : ℚ) / next_eight_composite_product = 1 / 2475 := by
  sorry

end NUMINAMATH_CALUDE_composite_number_division_l3272_327229


namespace NUMINAMATH_CALUDE_inequality_proof_l3272_327219

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3272_327219


namespace NUMINAMATH_CALUDE_waiter_remaining_customers_l3272_327255

theorem waiter_remaining_customers 
  (initial_customers : Real) 
  (first_group_left : Real) 
  (second_group_left : Real) 
  (h1 : initial_customers = 36.0)
  (h2 : first_group_left = 19.0)
  (h3 : second_group_left = 14.0) : 
  initial_customers - first_group_left - second_group_left = 3.0 := by
sorry

end NUMINAMATH_CALUDE_waiter_remaining_customers_l3272_327255


namespace NUMINAMATH_CALUDE_set_equality_l3272_327269

theorem set_equality (M P : Set α) (h : M ∩ P = P) : M ∪ P = M := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3272_327269


namespace NUMINAMATH_CALUDE_max_value_x_plus_sqrt_one_minus_x_squared_l3272_327214

theorem max_value_x_plus_sqrt_one_minus_x_squared :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x + Real.sqrt (1 - x^2) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_sqrt_one_minus_x_squared_l3272_327214


namespace NUMINAMATH_CALUDE_limit_fraction_three_n_l3272_327233

/-- The limit of (3^n - 1) / (3^(n+1) + 1) as n approaches infinity is 1/3 -/
theorem limit_fraction_three_n (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |((3^n - 1) / (3^(n+1) + 1)) - 1/3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_fraction_three_n_l3272_327233


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3272_327285

/-- The trajectory of the midpoint of a line segment PQ, where P moves on the unit circle and Q is fixed at (3,0) -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x_p y_p : ℝ, x_p^2 + y_p^2 = 1 ∧ x = (x_p + 3)/2 ∧ y = y_p/2) → 
  (2*x - 3)^2 + 4*y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3272_327285


namespace NUMINAMATH_CALUDE_max_d_is_one_l3272_327258

/-- The sequence a_n defined as (10^n - 1) / 9 -/
def a (n : ℕ) : ℕ := (10^n - 1) / 9

/-- The greatest common divisor of a_n and a_{n+1} -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- Theorem: The maximum value of d_n is 1 -/
theorem max_d_is_one : ∀ n : ℕ, d n = 1 := by sorry

end NUMINAMATH_CALUDE_max_d_is_one_l3272_327258


namespace NUMINAMATH_CALUDE_board_number_theorem_l3272_327297

/-- Represents the state of the numbers on the board -/
structure BoardState where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The operation described in the problem -/
def applyOperation (state : BoardState) : BoardState :=
  ⟨state.a, state.b, state.a + state.b - state.c⟩

/-- Checks if the numbers form an arithmetic sequence with difference 6 -/
def isArithmeticSequence (state : BoardState) : Prop :=
  state.b - state.a = 6 ∧ state.c - state.b = 6

/-- The main theorem to be proved -/
theorem board_number_theorem :
  ∃ (n : ℕ) (finalState : BoardState),
    finalState = (applyOperation^[n] ⟨3, 9, 15⟩) ∧
    isArithmeticSequence finalState ∧
    finalState.a = 2013 ∧
    finalState.b = 2019 ∧
    finalState.c = 2025 := by
  sorry

end NUMINAMATH_CALUDE_board_number_theorem_l3272_327297


namespace NUMINAMATH_CALUDE_parallel_line_condition_perpendicular_line_condition_l3272_327231

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (3, 0)

-- Define the given lines
def line1 (x y : ℝ) : Prop := 4 * x + y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 4 * x + y - 14 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 3 = 0

-- Theorem for the first condition
theorem parallel_line_condition :
  parallel_line A.1 A.2 ∧
  ∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), line1 (x + k) (y - 4 * k) :=
sorry

-- Theorem for the second condition
theorem perpendicular_line_condition :
  perpendicular_line B.1 B.2 ∧
  ∀ (x y : ℝ), perpendicular_line x y ↔ ∃ (k : ℝ), line2 (x + 2 * k) (y + k) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_condition_perpendicular_line_condition_l3272_327231


namespace NUMINAMATH_CALUDE_range_of_x_l3272_327271

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x y, x ≤ y → y ≤ 0 → f y ≤ f x
axiom f_at_neg_one : f (-1) = 1/2

-- State the theorem
theorem range_of_x (x : ℝ) : 2 * f (2*x - 1) - 1 < 0 ↔ 0 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_range_of_x_l3272_327271


namespace NUMINAMATH_CALUDE_french_fries_price_l3272_327274

/-- The cost of french fries at the burger hut -/
def french_fries_cost : ℝ := 1.50

/-- The cost of a burger -/
def burger_cost : ℝ := 5

/-- The cost of a soft drink -/
def soft_drink_cost : ℝ := 3

/-- The cost of a special burger meal -/
def special_meal_cost : ℝ := 9.50

theorem french_fries_price : 
  french_fries_cost = special_meal_cost - (burger_cost + soft_drink_cost) := by
  sorry

end NUMINAMATH_CALUDE_french_fries_price_l3272_327274


namespace NUMINAMATH_CALUDE_exponential_inequality_range_l3272_327281

theorem exponential_inequality_range (x : ℝ) : 
  (2 : ℝ) ^ (2 * x - 7) < (2 : ℝ) ^ (x - 3) → x < 4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_range_l3272_327281


namespace NUMINAMATH_CALUDE_matrix_not_invertible_sum_l3272_327217

/-- Given a 3x3 matrix with real entries x, y, z in the form
    [[x, y, z], [y, z, x], [z, x, y]],
    if the matrix is not invertible, then the sum
    x/(y+z) + y/(z+x) + z/(x+y) is equal to either -3 or 3/2 -/
theorem matrix_not_invertible_sum (x y z : ℝ) :
  let M := ![![x, y, z], ![y, z, x], ![z, x, y]]
  ¬ IsUnit (Matrix.det M) →
  (x / (y + z) + y / (z + x) + z / (x + y) = -3) ∨
  (x / (y + z) + y / (z + x) + z / (x + y) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_sum_l3272_327217


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_l3272_327226

theorem least_positive_integer_multiple (x : ℕ) : x = 47 ↔ 
  (x > 0 ∧ ∀ y : ℕ, y > 0 → y < x → ¬((2*y)^2 + 2*47*2*y + 47^2) % 47 = 0) ∧
  ((2*x)^2 + 2*47*2*x + 47^2) % 47 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_l3272_327226


namespace NUMINAMATH_CALUDE_angle_value_l3272_327256

theorem angle_value (θ : Real) (h1 : θ ∈ Set.Ioo 0 (2 * Real.pi)) 
  (h2 : (Real.sin 2, Real.cos 2) ∈ Set.range (λ t => (Real.sin t, Real.cos t))) :
  θ = 5 * Real.pi / 2 - 2 := by sorry

end NUMINAMATH_CALUDE_angle_value_l3272_327256


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3272_327200

/-- A square with perimeter 36 cm has an area of 81 cm² -/
theorem square_area_from_perimeter : 
  ∀ (s : ℝ), s > 0 → 4 * s = 36 → s^2 = 81 :=
by
  sorry


end NUMINAMATH_CALUDE_square_area_from_perimeter_l3272_327200


namespace NUMINAMATH_CALUDE_shaded_area_circle_configuration_l3272_327273

/-- The area of the shaded region in a circle configuration --/
theorem shaded_area_circle_configuration (R : ℝ) (h : R = 8) : 
  R^2 * Real.pi - 3 * (R/2)^2 * Real.pi = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_circle_configuration_l3272_327273


namespace NUMINAMATH_CALUDE_pet_sitting_earnings_l3272_327242

def hourly_rate : ℕ := 5
def hours_week1 : ℕ := 20
def hours_week2 : ℕ := 30

theorem pet_sitting_earnings : 
  hourly_rate * (hours_week1 + hours_week2) = 250 := by
  sorry

end NUMINAMATH_CALUDE_pet_sitting_earnings_l3272_327242


namespace NUMINAMATH_CALUDE_equation_satisfied_l3272_327257

theorem equation_satisfied (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_satisfied_l3272_327257


namespace NUMINAMATH_CALUDE_cone_base_radius_l3272_327289

/-- A right circular cone with given volume and height has a specific base radius -/
theorem cone_base_radius (V : ℝ) (h : ℝ) (r : ℝ) : 
  V = 24 * Real.pi ∧ h = 6 → V = (1/3) * Real.pi * r^2 * h → r = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3272_327289


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3272_327240

/-- Represents the number of animals in the farm -/
structure FarmAnimals where
  pigs : ℕ
  hens : ℕ

/-- The total number of legs in the farm -/
def totalLegs (animals : FarmAnimals) : ℕ :=
  4 * animals.pigs + 2 * animals.hens

/-- The total number of heads in the farm -/
def totalHeads (animals : FarmAnimals) : ℕ :=
  animals.pigs + animals.hens

/-- The condition given in the problem -/
def satisfiesCondition (animals : FarmAnimals) : Prop :=
  totalLegs animals = 3 * totalHeads animals + 36

theorem infinitely_many_solutions : 
  ∀ n : ℕ, ∃ animals : FarmAnimals, satisfiesCondition animals ∧ animals.pigs = n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3272_327240


namespace NUMINAMATH_CALUDE_sum_of_coordinates_B_l3272_327246

/-- Given points A(0, 0) and B(x, 3) where the slope of AB is 3/4, 
    prove that the sum of B's coordinates is 7. -/
theorem sum_of_coordinates_B (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  (3 - 0) / (x - 0) = 3 / 4 →
  x + 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_B_l3272_327246


namespace NUMINAMATH_CALUDE_line_outside_circle_l3272_327228

/-- A circle with a given diameter -/
structure Circle where
  diameter : ℝ

/-- A line with a given distance from a point -/
structure Line where
  distanceFromPoint : ℝ

/-- Relationship between a line and a circle -/
inductive Relationship
  | inside
  | tangent
  | outside

/-- Function to determine the relationship between a line and a circle -/
def relationshipBetweenLineAndCircle (c : Circle) (l : Line) : Relationship :=
  sorry

/-- Theorem stating that a line is outside a circle under given conditions -/
theorem line_outside_circle (c : Circle) (l : Line) 
  (h1 : c.diameter = 4)
  (h2 : l.distanceFromPoint = 3) :
  relationshipBetweenLineAndCircle c l = Relationship.outside :=
sorry

end NUMINAMATH_CALUDE_line_outside_circle_l3272_327228


namespace NUMINAMATH_CALUDE_f_properties_l3272_327223

noncomputable def f (x a : ℝ) := 2 * (Real.cos x)^2 + Real.sin (2 * x) + a

theorem f_properties (a : ℝ) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f x a = f (x + T) a ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f x a = f (x + T') a) → T ≤ T') ∧
  (∀ (k : ℤ), ∀ (x : ℝ), x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) →
    ∀ (y : ℝ), y ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) →
      x ≤ y → f x a ≤ f y a) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 6) → f x a ≤ 2) →
  a = 1 - Real.sqrt 2 ∧
  ∀ (k : ℤ), ∀ (x : ℝ), f x a = f (k * Real.pi + Real.pi / 4 - x) a :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3272_327223


namespace NUMINAMATH_CALUDE_feed_animals_theorem_l3272_327208

/-- The number of ways to feed animals in a conservatory -/
def feed_animals (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * n * feed_animals (n - 1)

/-- Theorem: Given 5 pairs of different animals, alternating between male and female,
    and starting with a female hippopotamus, there are 2880 ways to complete feeding all animals -/
theorem feed_animals_theorem : feed_animals 5 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_feed_animals_theorem_l3272_327208


namespace NUMINAMATH_CALUDE_non_binary_listeners_l3272_327267

/-- Represents the survey data from StreamNow -/
structure StreamNowSurvey where
  total_listeners : ℕ
  male_listeners : ℕ
  female_non_listeners : ℕ
  non_binary_non_listeners : ℕ
  total_non_listeners : ℕ

/-- Theorem stating the number of non-binary listeners based on the survey data -/
theorem non_binary_listeners (survey : StreamNowSurvey) 
  (h1 : survey.total_listeners = 250)
  (h2 : survey.male_listeners = 85)
  (h3 : survey.female_non_listeners = 95)
  (h4 : survey.non_binary_non_listeners = 45)
  (h5 : survey.total_non_listeners = 230) :
  survey.total_listeners - survey.male_listeners - survey.female_non_listeners = 70 :=
by sorry

end NUMINAMATH_CALUDE_non_binary_listeners_l3272_327267


namespace NUMINAMATH_CALUDE_club_officer_selection_l3272_327202

/-- The number of ways to select three distinct positions from a group of n people --/
def selectThreePositions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The number of club members --/
def clubMembers : ℕ := 12

theorem club_officer_selection :
  selectThreePositions clubMembers = 1320 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l3272_327202


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l3272_327220

theorem maximum_marks_calculation (percentage : ℝ) (scored_marks : ℝ) (maximum_marks : ℝ) :
  percentage = 90 →
  scored_marks = 405 →
  percentage / 100 * maximum_marks = scored_marks →
  maximum_marks = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l3272_327220


namespace NUMINAMATH_CALUDE_bill_sunday_miles_l3272_327254

/-- Proves that Bill ran 9 miles on Sunday given the problem conditions --/
theorem bill_sunday_miles : ℕ → ℕ → Prop :=
  fun (bill_saturday : ℕ) (bill_sunday : ℕ) =>
    let julia_sunday := 2 * bill_sunday
    bill_saturday + bill_sunday + julia_sunday = 32 ∧
    bill_sunday = bill_saturday + 4 →
    bill_sunday = 9

/-- Proof of the theorem --/
lemma bill_sunday_miles_proof : ∃ (bill_saturday : ℕ), bill_sunday_miles bill_saturday (bill_saturday + 4) :=
  sorry

end NUMINAMATH_CALUDE_bill_sunday_miles_l3272_327254


namespace NUMINAMATH_CALUDE_vector_addition_l3272_327247

theorem vector_addition : 
  let v1 : Fin 3 → ℝ := ![(-5 : ℝ), 1, -4]
  let v2 : Fin 3 → ℝ := ![0, 8, -4]
  v1 + v2 = ![(-5 : ℝ), 9, -8] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l3272_327247


namespace NUMINAMATH_CALUDE_als_original_portion_l3272_327210

theorem als_original_portion
  (total_initial : ℝ)
  (total_final : ℝ)
  (h_total_initial : total_initial = 1200)
  (h_total_final : total_final = 1800)
  (a b c : ℝ)
  (h_initial_sum : a + b + c = total_initial)
  (h_final_sum : (a - 150) + (2 * b) + (3 * c) = total_final) :
  a = 550 := by
sorry

end NUMINAMATH_CALUDE_als_original_portion_l3272_327210


namespace NUMINAMATH_CALUDE_fish_caught_difference_l3272_327250

/-- Represents the number of fish caught by each fisherman -/
def fish_caught (season_length first_rate second_rate_1 second_rate_2 second_rate_3 : ℕ) 
  (second_period_1 second_period_2 : ℕ) : ℕ := 
  let first_total := first_rate * season_length
  let second_total := second_rate_1 * second_period_1 + 
                      second_rate_2 * second_period_2 + 
                      second_rate_3 * (season_length - second_period_1 - second_period_2)
  (max first_total second_total) - (min first_total second_total)

/-- The difference in fish caught between the two fishermen is 3 -/
theorem fish_caught_difference : 
  fish_caught 213 3 1 2 4 30 60 = 3 := by sorry

end NUMINAMATH_CALUDE_fish_caught_difference_l3272_327250


namespace NUMINAMATH_CALUDE_henrikhs_distance_l3272_327299

/-- The number of blocks Henrikh lives from his office. -/
def blocks : ℕ :=
  sorry

/-- The time in minutes it takes Henrikh to walk to work. -/
def walkTime : ℚ :=
  blocks

/-- The time in minutes it takes Henrikh to cycle to work. -/
def cycleTime : ℚ :=
  blocks * (20 / 60)

theorem henrikhs_distance :
  blocks = 12 ∧ walkTime = cycleTime + 8 :=
by sorry

end NUMINAMATH_CALUDE_henrikhs_distance_l3272_327299


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_difference_l3272_327243

def first_integer : ℤ := 11
def second_integer : ℤ := 13
def third_integer : ℤ := 15

theorem consecutive_odd_integers_difference :
  (second_integer + third_integer) - first_integer = 17 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_difference_l3272_327243


namespace NUMINAMATH_CALUDE_inequality_solution_l3272_327207

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the inequality function
def f (x m : ℝ) : Prop := lg x ^ 2 - (2 + m) * lg x + m - 1 > 0

-- State the theorem
theorem inequality_solution :
  ∀ m : ℝ, |m| ≤ 1 →
    {x : ℝ | f x m} = {x : ℝ | 0 < x ∧ x < (1/10) ∨ x > 1000} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3272_327207


namespace NUMINAMATH_CALUDE_project_cost_increase_l3272_327288

def initial_lumber_cost : ℝ := 450
def initial_nails_cost : ℝ := 30
def initial_fabric_cost : ℝ := 80
def lumber_inflation_rate : ℝ := 0.20
def nails_inflation_rate : ℝ := 0.10
def fabric_inflation_rate : ℝ := 0.05

theorem project_cost_increase :
  let initial_total_cost := initial_lumber_cost + initial_nails_cost + initial_fabric_cost
  let new_lumber_cost := initial_lumber_cost * (1 + lumber_inflation_rate)
  let new_nails_cost := initial_nails_cost * (1 + nails_inflation_rate)
  let new_fabric_cost := initial_fabric_cost * (1 + fabric_inflation_rate)
  let new_total_cost := new_lumber_cost + new_nails_cost + new_fabric_cost
  new_total_cost - initial_total_cost = 97 := by
sorry

end NUMINAMATH_CALUDE_project_cost_increase_l3272_327288


namespace NUMINAMATH_CALUDE_grasshopper_jumps_l3272_327211

/-- Represents the state of three objects in a line -/
inductive Position
| ABC
| ACB
| BAC
| BCA
| CAB
| CBA

/-- Represents a single jump of one object over another -/
def jump (p : Position) : Position :=
  match p with
  | Position.ABC => Position.BAC
  | Position.ACB => Position.CAB
  | Position.BAC => Position.BCA
  | Position.BCA => Position.CBA
  | Position.CAB => Position.ACB
  | Position.CBA => Position.ABC

/-- Applies n jumps to a given position -/
def jumpN (p : Position) (n : Nat) : Position :=
  match n with
  | 0 => p
  | n + 1 => jump (jumpN p n)

theorem grasshopper_jumps (n : Nat) (h : Odd n) :
  ∀ p : Position, jumpN p n ≠ p :=
sorry

end NUMINAMATH_CALUDE_grasshopper_jumps_l3272_327211


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3272_327279

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of possible 6-digit numbers -/
def total_numbers : ℕ := 9 * 10^(num_digits - 1)

/-- The number of 6-digit numbers with no zeros -/
def numbers_without_zero : ℕ := 9^num_digits

/-- Theorem: The number of 6-digit numbers with at least one zero is 368,559 -/
theorem six_digit_numbers_with_zero : 
  total_numbers - numbers_without_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3272_327279


namespace NUMINAMATH_CALUDE_fraction_problem_l3272_327249

theorem fraction_problem (N : ℝ) (x : ℝ) 
  (h1 : N = 24.000000000000004) 
  (h2 : (1/4) * N = x * (N + 1) + 1) : 
  x = 0.20000000000000004 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3272_327249


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3272_327221

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (x + 12) = 10) ∧ (x = 88) := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3272_327221


namespace NUMINAMATH_CALUDE_logarithm_inequality_l3272_327232

theorem logarithm_inequality (m n p : ℝ) 
  (hm : 0 < m ∧ m < 1) 
  (hn : 0 < n ∧ n < 1) 
  (hp : 0 < p ∧ p < 1) 
  (h_log : Real.log m / Real.log 3 = Real.log n / Real.log 5 ∧ 
           Real.log n / Real.log 5 = Real.log p / Real.log 10) : 
  m^(1/3) < n^(1/5) ∧ n^(1/5) < p^(1/10) := by
sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l3272_327232


namespace NUMINAMATH_CALUDE_pyramid_max_volume_l3272_327282

/-- The maximum volume of a pyramid with given base side lengths and angle constraints. -/
theorem pyramid_max_volume (AB AC : ℝ) (sin_BAC : ℝ) (max_lateral_angle : ℝ) :
  AB = 3 →
  AC = 5 →
  sin_BAC = 3/5 →
  max_lateral_angle = 60 * π / 180 →
  ∃ (V : ℝ), V = (5 * Real.sqrt 174) / 4 ∧ 
    ∀ (V' : ℝ), V' ≤ V := by
  sorry

end NUMINAMATH_CALUDE_pyramid_max_volume_l3272_327282


namespace NUMINAMATH_CALUDE_middle_number_problem_l3272_327234

theorem middle_number_problem (x y z : ℤ) : 
  x < y ∧ y < z →
  x + y = 18 →
  x + z = 25 →
  y + z = 27 →
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_middle_number_problem_l3272_327234


namespace NUMINAMATH_CALUDE_money_left_l3272_327294

def initial_amount : ℕ := 20
def num_items : ℕ := 4
def cost_per_item : ℕ := 2

theorem money_left : initial_amount - (num_items * cost_per_item) = 12 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l3272_327294


namespace NUMINAMATH_CALUDE_log_equation_solution_l3272_327275

theorem log_equation_solution (x : ℝ) (h1 : x > 0) (h2 : 2 * x ≠ 1) (h3 : 4 * x ≠ 1) :
  (Real.log (4 * x) / Real.log (2 * x)) + (Real.log (16 * x) / Real.log (4 * x)) = 4 ↔ 
  x = 1 ∨ x = 1 / (2 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3272_327275


namespace NUMINAMATH_CALUDE_february_monthly_fee_calculation_l3272_327259

/-- Represents the monthly membership fee and per-class fee structure -/
structure FeeStructure where
  monthly_fee : ℝ
  per_class_fee : ℝ

/-- Calculates the total bill given a fee structure and number of classes -/
def total_bill (fs : FeeStructure) (classes : ℕ) : ℝ :=
  fs.monthly_fee + fs.per_class_fee * classes

/-- Represents the fee structure with a 10% increase in monthly fee -/
def increased_fee_structure (fs : FeeStructure) : FeeStructure :=
  { monthly_fee := 1.1 * fs.monthly_fee
    per_class_fee := fs.per_class_fee }

theorem february_monthly_fee_calculation 
  (feb_fs : FeeStructure)
  (h1 : total_bill feb_fs 4 = 30.72)
  (h2 : total_bill (increased_fee_structure feb_fs) 8 = 54.72) :
  feb_fs.monthly_fee = 7.47 := by
  sorry

#eval (7.47 : Float).toString

end NUMINAMATH_CALUDE_february_monthly_fee_calculation_l3272_327259


namespace NUMINAMATH_CALUDE_equation_solution_l3272_327204

theorem equation_solution :
  ∃! x : ℚ, 7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + x :=
by
  use -22/3
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3272_327204


namespace NUMINAMATH_CALUDE_frank_unfilled_boxes_l3272_327222

/-- Given a total number of boxes and a number of filled boxes,
    calculate the number of unfilled boxes. -/
def unfilled_boxes (total : ℕ) (filled : ℕ) : ℕ :=
  total - filled

/-- Theorem: Frank has 5 unfilled boxes -/
theorem frank_unfilled_boxes :
  unfilled_boxes 13 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_frank_unfilled_boxes_l3272_327222


namespace NUMINAMATH_CALUDE_unstable_products_selection_l3272_327266

theorem unstable_products_selection (n : ℕ) (d : ℕ) (k : ℕ) (h1 : n = 10) (h2 : d = 2) (h3 : k = 3) :
  (Nat.choose (n - d) 1 * d * Nat.choose (d - 1) 1) = 32 :=
sorry

end NUMINAMATH_CALUDE_unstable_products_selection_l3272_327266


namespace NUMINAMATH_CALUDE_product_equals_sum_solution_l3272_327262

theorem product_equals_sum_solution :
  ∀ (a b c d e f : ℕ),
    a * b * c * d * e * f = a + b + c + d + e + f →
    ((a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 6) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 6 ∧ f = 2) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 6) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 6 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 6 ∧ e = 1 ∧ f = 2) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 6 ∧ e = 2 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 6) ∨
     (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 6 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 6 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 6 ∧ d = 1 ∧ e = 1 ∧ f = 2) ∨
     (a = 1 ∧ b = 1 ∧ c = 6 ∧ d = 1 ∧ e = 2 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 6 ∧ d = 2 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 6) ∨
     (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 6 ∧ f = 1) ∨
     (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 6 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 2 ∧ c = 6 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 6 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 2) ∨
     (a = 1 ∧ b = 6 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 1) ∨
     (a = 1 ∧ b = 6 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 6 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 6) ∨
     (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 6 ∧ f = 1) ∨
     (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 6 ∧ e = 1 ∧ f = 1) ∨
     (a = 2 ∧ b = 1 ∧ c = 6 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 2 ∧ b = 6 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 6 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 2) ∨
     (a = 6 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 1) ∨
     (a = 6 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 1) ∨
     (a = 6 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 6 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_product_equals_sum_solution_l3272_327262


namespace NUMINAMATH_CALUDE_original_sandbox_capacity_l3272_327218

/-- Given a rectangular sandbox, this theorem proves that if a new sandbox with twice the dimensions
    has a capacity of 80 cubic feet, then the original sandbox has a capacity of 10 cubic feet. -/
theorem original_sandbox_capacity
  (length width height : ℝ)
  (new_sandbox_capacity : ℝ → ℝ → ℝ → ℝ)
  (h_new_sandbox : new_sandbox_capacity (2 * length) (2 * width) (2 * height) = 80) :
  length * width * height = 10 := by
  sorry

end NUMINAMATH_CALUDE_original_sandbox_capacity_l3272_327218


namespace NUMINAMATH_CALUDE_min_ab_min_a_plus_2b_min_ab_attained_min_a_plus_2b_attained_l3272_327244

-- Define the condition for positive real numbers a and b
def condition (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a * b = 2 * a + b

-- Theorem for the minimum value of ab
theorem min_ab (a b : ℝ) (h : condition a b) : a * b ≥ 8 := by
  sorry

-- Theorem for the minimum value of a + 2b
theorem min_a_plus_2b (a b : ℝ) (h : condition a b) : a + 2 * b ≥ 9 := by
  sorry

-- The minimum values are attained
theorem min_ab_attained : ∃ a b : ℝ, condition a b ∧ a * b = 8 := by
  sorry

theorem min_a_plus_2b_attained : ∃ a b : ℝ, condition a b ∧ a + 2 * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_ab_min_a_plus_2b_min_ab_attained_min_a_plus_2b_attained_l3272_327244


namespace NUMINAMATH_CALUDE_pie_eating_contest_l3272_327251

theorem pie_eating_contest (erik_pie frank_pie : ℝ) 
  (h_erik : erik_pie = 0.67)
  (h_frank : frank_pie = 0.33) :
  erik_pie - frank_pie = 0.34 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l3272_327251


namespace NUMINAMATH_CALUDE_snow_probability_first_week_l3272_327235

theorem snow_probability_first_week (p1 p2 : ℝ) : 
  p1 = 1/3 → p2 = 1/4 → 
  (1 - (1 - p1)^4 * (1 - p2)^3) = 11/12 :=
by sorry

end NUMINAMATH_CALUDE_snow_probability_first_week_l3272_327235


namespace NUMINAMATH_CALUDE_myrtle_dropped_eggs_l3272_327213

/-- Represents the problem of calculating how many eggs Myrtle dropped --/
theorem myrtle_dropped_eggs (hens : ℕ) (eggs_per_hen : ℕ) (days : ℕ) (neighbor_took : ℕ) (myrtle_has : ℕ)
  (h1 : hens = 3)
  (h2 : eggs_per_hen = 3)
  (h3 : days = 7)
  (h4 : neighbor_took = 12)
  (h5 : myrtle_has = 46) :
  hens * eggs_per_hen * days - neighbor_took - myrtle_has = 5 := by
  sorry

#check myrtle_dropped_eggs

end NUMINAMATH_CALUDE_myrtle_dropped_eggs_l3272_327213


namespace NUMINAMATH_CALUDE_dodgeball_team_theorem_l3272_327215

/-- The number of players in the dodgeball league -/
def total_players : ℕ := 12

/-- The number of players on each team -/
def team_size : ℕ := 6

/-- The number of times two specific players are on the same team -/
def same_team_count : ℕ := 210

/-- The total number of possible team combinations -/
def total_combinations : ℕ := Nat.choose total_players team_size

theorem dodgeball_team_theorem :
  ∀ (player1 player2 : Fin total_players),
    player1 ≠ player2 →
    (Nat.choose (total_players - 2) (team_size - 2) : ℕ) = same_team_count :=
by sorry

end NUMINAMATH_CALUDE_dodgeball_team_theorem_l3272_327215


namespace NUMINAMATH_CALUDE_area_of_triangle_def_l3272_327298

/-- Triangle DEF with vertices D, E, and F -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The line on which point F lies -/
def line_equation (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 = 9

/-- Calculate the area of a triangle given its vertices -/
def triangle_area (t : Triangle) : ℝ :=
  sorry

/-- The main theorem stating that the area of triangle DEF is 10 -/
theorem area_of_triangle_def :
  ∀ (t : Triangle),
    t.D = (4, 0) →
    t.E = (0, 4) →
    line_equation t.F →
    triangle_area t = 10 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_def_l3272_327298


namespace NUMINAMATH_CALUDE_equation_solution_l3272_327224

theorem equation_solution (x : ℝ) (h : x > 0) :
  (x - 3) / 8 = 5 / (x - 8) ↔ x = 16 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3272_327224


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3272_327236

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3, 4, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3272_327236


namespace NUMINAMATH_CALUDE_pencil_cost_l3272_327291

theorem pencil_cost (pen_price pencil_price : ℚ) 
  (eq1 : 5 * pen_price + 4 * pencil_price = 310)
  (eq2 : 3 * pen_price + 6 * pencil_price = 238) :
  pencil_price = 130 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l3272_327291


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l3272_327209

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The circle E passing through (0,0), (1,1), and (2,0) -/
def circle_E : Circle :=
  { center := (1, 0), radius := 1 }

/-- Point P -/
def point_P : ℝ × ℝ := (2, 3)

/-- Theorem stating the properties of circle E and line l -/
theorem circle_and_tangent_line :
  (∀ (x y : ℝ), (x - circle_E.center.1)^2 + (y - circle_E.center.2)^2 = circle_E.radius^2 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 0)) ∧
  (∃ (l : Line), 
    (l.a * point_P.1 + l.b * point_P.2 + l.c = 0) ∧
    (∀ (x y : ℝ), (x - circle_E.center.1)^2 + (y - circle_E.center.2)^2 = circle_E.radius^2 →
      (l.a * x + l.b * y + l.c)^2 ≥ (l.a^2 + l.b^2) * circle_E.radius^2) ∧
    ((l.a = 1 ∧ l.b = 0 ∧ l.c = -2) ∨ (l.a = 4 ∧ l.b = -3 ∧ l.c = 1))) :=
sorry


end NUMINAMATH_CALUDE_circle_and_tangent_line_l3272_327209


namespace NUMINAMATH_CALUDE_average_math_score_l3272_327241

/-- Represents the total number of students -/
def total_students : ℕ := 500

/-- Represents the number of male students -/
def male_students : ℕ := 300

/-- Represents the number of female students -/
def female_students : ℕ := 200

/-- Represents the sample size -/
def sample_size : ℕ := 60

/-- Represents the average score of male students in the sample -/
def male_avg_score : ℝ := 110

/-- Represents the average score of female students in the sample -/
def female_avg_score : ℝ := 100

/-- Theorem stating that the average math score of first-year students is 106 points -/
theorem average_math_score : 
  (male_students : ℝ) / total_students * male_avg_score + 
  (female_students : ℝ) / total_students * female_avg_score = 106 := by
  sorry

end NUMINAMATH_CALUDE_average_math_score_l3272_327241


namespace NUMINAMATH_CALUDE_paths_count_is_36_l3272_327248

/-- Represents the circular arrangement of numbers -/
structure CircularArrangement where
  center : Nat
  surrounding : Nat
  zeroAdjacent : Nat
  fiveAdjacent : Nat

/-- Calculates the number of distinct paths to form 2005 -/
def countPaths (arrangement : CircularArrangement) : Nat :=
  arrangement.surrounding * arrangement.zeroAdjacent * arrangement.fiveAdjacent

/-- The specific arrangement for the problem -/
def problemArrangement : CircularArrangement :=
  { center := 2
  , surrounding := 6
  , zeroAdjacent := 2
  , fiveAdjacent := 3 }

theorem paths_count_is_36 :
  countPaths problemArrangement = 36 := by
  sorry

end NUMINAMATH_CALUDE_paths_count_is_36_l3272_327248


namespace NUMINAMATH_CALUDE_profit_increase_condition_l3272_327261

/-- The selling price function -/
def price (t : ℤ) : ℚ := (1/4) * t + 30

/-- The daily sales volume function -/
def sales_volume (t : ℤ) : ℚ := 120 - 2 * t

/-- The daily profit function after donation -/
def profit (t : ℤ) (n : ℚ) : ℚ :=
  (price t - 20 - n) * sales_volume t

/-- The derivative of the profit function with respect to t -/
def profit_derivative (t : ℤ) (n : ℚ) : ℚ :=
  -t + 2*n + 10

theorem profit_increase_condition (n : ℚ) :
  (∀ t : ℤ, 1 ≤ t ∧ t ≤ 28 → profit_derivative t n > 0) ↔
  (8.75 < n ∧ n ≤ 9.25) :=
sorry

end NUMINAMATH_CALUDE_profit_increase_condition_l3272_327261


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l3272_327276

/-- A natural number is composite if it has more than two divisors -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ k * m = n

/-- A natural number can be expressed as the sum of two composite numbers -/
def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ a + b = n

/-- 11 is the largest natural number that cannot be expressed as the sum of two composite numbers -/
theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l3272_327276
