import Mathlib

namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l53_5386

theorem factorial_fraction_equality : (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l53_5386


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_140_l53_5352

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_140 :
  rectangle_area 1225 10 = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_140_l53_5352


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l53_5374

theorem solve_quadratic_equation (x : ℝ) :
  2 * (x - 3)^2 - 98 = 0 → x = 10 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l53_5374


namespace NUMINAMATH_CALUDE_women_in_room_l53_5305

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 7 / 9 →
  initial_men + 8 = 28 →
  2 * (initial_women - 2) = 48 :=
by sorry

end NUMINAMATH_CALUDE_women_in_room_l53_5305


namespace NUMINAMATH_CALUDE_remainder_four_eleven_mod_five_l53_5308

theorem remainder_four_eleven_mod_five : 4^11 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_four_eleven_mod_five_l53_5308


namespace NUMINAMATH_CALUDE_journey_distance_l53_5332

theorem journey_distance (train_fraction : ℚ) (bus_fraction : ℚ) (walk_distance : ℝ) :
  train_fraction = 3/5 →
  bus_fraction = 7/20 →
  walk_distance = 6.5 →
  1 - (train_fraction + bus_fraction) = walk_distance / 130 →
  130 = (walk_distance * 20 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l53_5332


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l53_5328

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse a b) : ℝ := sorry

/-- The right focus of an ellipse -/
def right_focus (e : Ellipse a b) : Point := sorry

/-- The intersection points of a line and an ellipse -/
def intersection_points (e : Ellipse a b) (l : Line) : (Point × Point) := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if two line segments are perpendicular -/
def perpendicular (p1 p2 p3 : Point) : Prop := sorry

theorem ellipse_eccentricity_theorem 
  (a b : ℝ) (e : Ellipse a b) (l : Line) :
  let F := right_focus e
  let (A, B) := intersection_points e l
  perpendicular A F B ∧ distance A F = 3 * distance B F →
  eccentricity e = Real.sqrt 10 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l53_5328


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l53_5306

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (a b c : ℕ) : ℕ := a * 7^2 + b * 7 + c

/-- Theorem: If 451 in base 7 equals xy in base 10 (where x and y are single digits),
    then (x * y) / 10 = 0.6 -/
theorem base_conversion_theorem (x y : ℕ) (h1 : x < 10) (h2 : y < 10) 
    (h3 : base7ToBase10 4 5 1 = 10 * x + y) : 
    (x * y : ℚ) / 10 = 6 / 10 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l53_5306


namespace NUMINAMATH_CALUDE_evaluate_expression_l53_5318

theorem evaluate_expression : (0.5^2 + 0.05^3) / (0.005^3) = 2000100 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l53_5318


namespace NUMINAMATH_CALUDE_solution_set_and_inequality_l53_5389

-- Define the set T
def T : Set ℝ := {t | t > 1}

-- Define the function f(x) = |x-2| + |x-3|
def f (x : ℝ) : ℝ := |x - 2| + |x - 3|

theorem solution_set_and_inequality :
  -- Part 1: T is the set of all t for which |x-2|+|x-3| < t has a non-empty solution set
  (∀ t : ℝ, (∃ x : ℝ, f x < t) ↔ t ∈ T) ∧
  -- Part 2: For all a, b ∈ T, ab + 1 > a + b
  (∀ a b : ℝ, a ∈ T → b ∈ T → a * b + 1 > a + b) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_and_inequality_l53_5389


namespace NUMINAMATH_CALUDE_min_distance_circle_point_l53_5399

noncomputable section

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 + 1)^2 = 4}

-- Define point Q
def point_Q : ℝ × ℝ := (Real.sqrt 2, -Real.sqrt 2)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_circle_point :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 ∧
  ∀ (P : ℝ × ℝ), P ∈ circle_C →
  distance P point_Q ≥ min_dist :=
sorry

end

end NUMINAMATH_CALUDE_min_distance_circle_point_l53_5399


namespace NUMINAMATH_CALUDE_line_equation_l53_5362

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line -/
theorem line_equation (x y : ℝ) : 
  (2 * x - 3 * y - 3 = 0) →   -- First given line
  (x + y + 2 = 0) →           -- Second given line
  (∃ k : ℝ, 3 * x + y - k = 0) →  -- Parallel line condition
  (15 * x + 5 * y + 16 = 0) := by  -- Equation to prove
sorry

end NUMINAMATH_CALUDE_line_equation_l53_5362


namespace NUMINAMATH_CALUDE_calculation_1_calculation_2_calculation_3_calculation_4_calculation_5_calculation_6_l53_5388

theorem calculation_1 : 320 + 16 * 27 = 752 := by sorry

theorem calculation_2 : 1500 - 125 * 8 = 500 := by sorry

theorem calculation_3 : 22 * 22 - 84 = 400 := by sorry

theorem calculation_4 : 25 * 8 * 9 = 1800 := by sorry

theorem calculation_5 : (25 + 38) * 15 = 945 := by sorry

theorem calculation_6 : (62 + 12) * 38 = 2812 := by sorry

end NUMINAMATH_CALUDE_calculation_1_calculation_2_calculation_3_calculation_4_calculation_5_calculation_6_l53_5388


namespace NUMINAMATH_CALUDE_train_arrival_theorem_l53_5393

/-- Represents a time with day, hour, and minute -/
structure Time where
  day : String
  hour : Nat
  minute : Nat

/-- Represents the journey of the train -/
structure TrainJourney where
  startTime : Time
  firstLegDuration : Nat
  secondLegDuration : Nat
  layoverDuration : Nat
  timeZonesCrossed : Nat
  timeZoneDifference : Nat

def calculateArrivalTime (journey : TrainJourney) : Time :=
  sorry

theorem train_arrival_theorem (journey : TrainJourney) 
  (h1 : journey.startTime = ⟨"Tuesday", 5, 0⟩)
  (h2 : journey.firstLegDuration = 12)
  (h3 : journey.secondLegDuration = 21)
  (h4 : journey.layoverDuration = 3)
  (h5 : journey.timeZonesCrossed = 2)
  (h6 : journey.timeZoneDifference = 1) :
  calculateArrivalTime journey = ⟨"Wednesday", 9, 0⟩ :=
by
  sorry

#check train_arrival_theorem

end NUMINAMATH_CALUDE_train_arrival_theorem_l53_5393


namespace NUMINAMATH_CALUDE_increasing_function_properties_l53_5394

/-- A function f is increasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem increasing_function_properties
  (f : ℝ → ℝ) (hf : IncreasingFunction f) :
  (∀ a b : ℝ, a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b)) ∧
  (∀ a b : ℝ, f a + f b ≥ f (-a) + f (-b) → a + b ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_properties_l53_5394


namespace NUMINAMATH_CALUDE_stack_toppled_plates_l53_5385

/-- The number of plates in Alice's stack when it toppled over --/
def total_plates (initial_plates added_plates : ℕ) : ℕ :=
  initial_plates + added_plates

/-- Theorem: The total number of plates when the stack toppled is 64 --/
theorem stack_toppled_plates :
  total_plates 27 37 = 64 := by
  sorry

end NUMINAMATH_CALUDE_stack_toppled_plates_l53_5385


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_300_l53_5322

theorem modular_inverse_13_mod_300 : ∃ x : ℕ, x < 300 ∧ (13 * x) % 300 = 1 :=
  by
    use 277
    sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_300_l53_5322


namespace NUMINAMATH_CALUDE_tangent_product_equals_three_l53_5395

theorem tangent_product_equals_three :
  Real.tan (20 * π / 180) * Real.tan (40 * π / 180) * Real.tan (60 * π / 180) * Real.tan (80 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equals_three_l53_5395


namespace NUMINAMATH_CALUDE_triangle_angle_B_l53_5319

theorem triangle_angle_B (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  3 * a * Real.cos C = 2 * c * Real.cos A →
  Real.tan A = 1 / 3 →
  B = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l53_5319


namespace NUMINAMATH_CALUDE_tshirt_shop_weekly_earnings_l53_5359

/-- Represents the T-shirt shop's operations and calculates weekly earnings -/
def TShirtShopEarnings : ℕ :=
  let women_shirt_price : ℕ := 18
  let men_shirt_price : ℕ := 15
  let women_shirt_interval : ℕ := 30  -- in minutes
  let men_shirt_interval : ℕ := 40    -- in minutes
  let daily_open_hours : ℕ := 12
  let days_per_week : ℕ := 7
  let minutes_per_hour : ℕ := 60

  let women_shirts_per_day : ℕ := (minutes_per_hour / women_shirt_interval) * daily_open_hours
  let men_shirts_per_day : ℕ := (minutes_per_hour / men_shirt_interval) * daily_open_hours
  
  let daily_earnings : ℕ := women_shirts_per_day * women_shirt_price + men_shirts_per_day * men_shirt_price
  
  daily_earnings * days_per_week

/-- Theorem stating that the weekly earnings of the T-shirt shop is $4914 -/
theorem tshirt_shop_weekly_earnings : TShirtShopEarnings = 4914 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_shop_weekly_earnings_l53_5359


namespace NUMINAMATH_CALUDE_student_grade_proof_l53_5321

def courses_last_year : ℕ := 6
def avg_grade_last_year : ℝ := 100
def courses_year_before : ℕ := 5
def avg_grade_two_years : ℝ := 77
def total_courses : ℕ := courses_last_year + courses_year_before

theorem student_grade_proof :
  ∃ (avg_grade_year_before : ℝ),
    avg_grade_year_before * courses_year_before + avg_grade_last_year * courses_last_year =
    avg_grade_two_years * total_courses ∧
    avg_grade_year_before = 49.4 := by
  sorry

end NUMINAMATH_CALUDE_student_grade_proof_l53_5321


namespace NUMINAMATH_CALUDE_min_cubes_is_four_l53_5324

/-- Represents a cube with two protruding snaps on opposite sides and four receptacle holes --/
structure Cube where
  snaps : Fin 2
  holes : Fin 4

/-- Represents an assembly of cubes --/
def Assembly := List Cube

/-- Checks if an assembly has only receptacle holes visible --/
def Assembly.onlyHolesVisible (a : Assembly) : Prop :=
  sorry

/-- The minimum number of cubes required for a valid assembly --/
def minCubesForValidAssembly : ℕ :=
  sorry

/-- Theorem stating that the minimum number of cubes for a valid assembly is 4 --/
theorem min_cubes_is_four :
  minCubesForValidAssembly = 4 :=
sorry

end NUMINAMATH_CALUDE_min_cubes_is_four_l53_5324


namespace NUMINAMATH_CALUDE_max_expensive_price_theorem_l53_5339

/-- Represents a set of products with their prices -/
structure ProductSet where
  prices : Finset ℝ
  count : Nat
  avg_price : ℝ
  min_price : ℝ
  low_price_count : Nat
  low_price_threshold : ℝ

/-- The maximum possible price for the most expensive product -/
def max_expensive_price (ps : ProductSet) : ℝ :=
  ps.count * ps.avg_price
    - (ps.low_price_count * ps.min_price
      + (ps.count - ps.low_price_count - 1) * ps.low_price_threshold)

/-- Theorem stating the maximum price of the most expensive product -/
theorem max_expensive_price_theorem (ps : ProductSet)
  (h_count : ps.count = 25)
  (h_avg_price : ps.avg_price = 1200)
  (h_min_price : ps.min_price = 400)
  (h_low_price_count : ps.low_price_count = 12)
  (h_low_price_threshold : ps.low_price_threshold = 1000)
  (h_prices_above_min : ∀ p ∈ ps.prices, p ≥ ps.min_price)
  (h_low_price_count_correct : (ps.prices.filter (· < ps.low_price_threshold)).card = ps.low_price_count) :
  max_expensive_price ps = 13200 := by
  sorry

end NUMINAMATH_CALUDE_max_expensive_price_theorem_l53_5339


namespace NUMINAMATH_CALUDE_permutations_with_four_transpositions_l53_5361

/-- The number of elements in the permutation -/
def n : ℕ := 6

/-- The total number of permutations of n elements -/
def total_permutations : ℕ := n.factorial

/-- The number of even permutations -/
def even_permutations : ℕ := total_permutations / 2

/-- The number of permutations that require i transpositions to become the identity permutation -/
def num_permutations (i : ℕ) : ℕ := sorry

/-- The theorem stating that the number of permutations requiring 4 transpositions is 304 -/
theorem permutations_with_four_transpositions :
  num_permutations 4 = 304 :=
sorry

end NUMINAMATH_CALUDE_permutations_with_four_transpositions_l53_5361


namespace NUMINAMATH_CALUDE_absolute_value_non_positive_l53_5348

theorem absolute_value_non_positive (y : ℚ) : 
  |4 * y - 6| ≤ 0 ↔ y = 3/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_non_positive_l53_5348


namespace NUMINAMATH_CALUDE_difference_equals_1011_l53_5368

/-- The sum of consecutive odd numbers from 1 to 2021 -/
def sum_odd : ℕ := (2021 + 1) / 2 ^ 2

/-- The sum of consecutive even numbers from 2 to 2020 -/
def sum_even : ℕ := (2020 / 2) * (2020 / 2 + 1)

/-- The difference between the sum of odd numbers and the sum of even numbers -/
def difference : ℕ := sum_odd - sum_even

theorem difference_equals_1011 : difference = 1011 := by
  sorry

end NUMINAMATH_CALUDE_difference_equals_1011_l53_5368


namespace NUMINAMATH_CALUDE_original_number_proof_l53_5350

theorem original_number_proof (x : ℝ) : ((3 * x^2 + 8) * 2) / 4 = 56 → x = 2 * Real.sqrt 78 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l53_5350


namespace NUMINAMATH_CALUDE_customers_left_l53_5302

theorem customers_left (initial : ℕ) (additional : ℕ) (final : ℕ) : 
  initial = 47 → additional = 20 → final = 26 → initial - (initial - additional + final) = 41 := by
sorry

end NUMINAMATH_CALUDE_customers_left_l53_5302


namespace NUMINAMATH_CALUDE_line_parabola_single_intersection_l53_5354

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line passing through (-3, 1) with slope k
def line (k x y : ℝ) : Prop := y - 1 = k * (x + 3)

-- Define the condition for the line to intersect the parabola at exactly one point
def single_intersection (k : ℝ) : Prop :=
  (k = 0 ∨ k = -1 ∨ k = 2/3) ∧
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2)

-- Theorem statement
theorem line_parabola_single_intersection (k : ℝ) :
  single_intersection k ↔ k = 0 ∨ k = -1 ∨ k = 2/3 :=
sorry

end NUMINAMATH_CALUDE_line_parabola_single_intersection_l53_5354


namespace NUMINAMATH_CALUDE_new_parabola_equation_l53_5377

/-- The original quadratic function -/
def original_function (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- The vertex of the original parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- The axis of symmetry of the original parabola -/
def axis_of_symmetry : ℝ := 1

/-- The line that intersects the new parabola -/
def intersecting_line (m : ℝ) (x : ℝ) : ℝ := m * x - 2

/-- The point of intersection between the new parabola and the line -/
def intersection_point : ℝ × ℝ := (2, 4)

/-- The equation of the new parabola -/
def new_parabola (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 4

theorem new_parabola_equation :
  ∃ (t : ℝ), ∀ (x : ℝ),
    new_parabola x = -3 * (x - axis_of_symmetry)^2 + vertex.2 + t ∧
    new_parabola intersection_point.1 = intersection_point.2 :=
by sorry

end NUMINAMATH_CALUDE_new_parabola_equation_l53_5377


namespace NUMINAMATH_CALUDE_grape_lollipops_count_l53_5372

/-- Given a total number of lollipops and the number of flavors for non-cherry lollipops,
    calculate the number of lollipops of a specific non-cherry flavor. -/
def grape_lollipops (total : ℕ) (non_cherry_flavors : ℕ) : ℕ :=
  (total / 2) / non_cherry_flavors

/-- Theorem stating that with 42 total lollipops and 3 non-cherry flavors,
    the number of grape lollipops is 7. -/
theorem grape_lollipops_count :
  grape_lollipops 42 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_grape_lollipops_count_l53_5372


namespace NUMINAMATH_CALUDE_statue_increase_factor_l53_5312

/-- The factor by which the number of statues increased in the second year --/
def increase_factor : ℝ := 4

/-- The initial number of statues --/
def initial_statues : ℕ := 4

/-- The number of statues added in the third year --/
def added_third_year : ℕ := 12

/-- The number of statues broken in the third year --/
def broken_third_year : ℕ := 3

/-- The final number of statues after four years --/
def final_statues : ℕ := 31

theorem statue_increase_factor : 
  (initial_statues : ℝ) * increase_factor + 
  (added_third_year : ℝ) - (broken_third_year : ℝ) + 
  2 * (broken_third_year : ℝ) = (final_statues : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_statue_increase_factor_l53_5312


namespace NUMINAMATH_CALUDE_max_area_is_100_l53_5392

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Checks if the rectangle satisfies the given conditions -/
def isValidRectangle (r : Rectangle) : Prop :=
  r.length + r.width = 20 ∧ Even r.width

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  r.length * r.width

/-- Theorem: The maximum area of a valid rectangle is 100 -/
theorem max_area_is_100 :
  ∃ (r : Rectangle), isValidRectangle r ∧
    area r = 100 ∧
    ∀ (s : Rectangle), isValidRectangle s → area s ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_max_area_is_100_l53_5392


namespace NUMINAMATH_CALUDE_barry_sotter_magic_barry_sotter_days_l53_5363

theorem barry_sotter_magic (n : ℕ) : (3/2 : ℝ)^n ≥ 50 ↔ n ≥ 10 := by sorry

theorem barry_sotter_days : ∃ (n : ℕ), (∀ (m : ℕ), (3/2 : ℝ)^m ≥ 50 → n ≤ m) ∧ (3/2 : ℝ)^n ≥ 50 :=
by
  use 10
  sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_barry_sotter_days_l53_5363


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l53_5355

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2004 - 1) (2^1995 - 1) = 2^9 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l53_5355


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l53_5313

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y z : ℝ) :
  S.card = 60 →
  x = 48 ∧ y = 58 ∧ z = 52 →
  (S.sum id) / S.card = 42 →
  ((S.sum id) - (x + y + z)) / (S.card - 3) = 41.4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l53_5313


namespace NUMINAMATH_CALUDE_octal_addition_47_56_l53_5379

/-- Represents a digit in the octal system -/
def OctalDigit := Fin 8

/-- Represents a number in the octal system as a list of digits -/
def OctalNumber := List OctalDigit

/-- Addition operation for octal numbers -/
def octal_add : OctalNumber → OctalNumber → OctalNumber := sorry

/-- Conversion from a natural number to an octal number -/
def nat_to_octal : ℕ → OctalNumber := sorry

/-- Conversion from an octal number to a natural number -/
def octal_to_nat : OctalNumber → ℕ := sorry

theorem octal_addition_47_56 :
  octal_add (nat_to_octal 47) (nat_to_octal 56) = nat_to_octal 125 := by sorry

end NUMINAMATH_CALUDE_octal_addition_47_56_l53_5379


namespace NUMINAMATH_CALUDE_flight_duration_sum_l53_5365

-- Define the flight departure and arrival times in minutes since midnight
def departure_time : ℕ := 10 * 60 + 34
def arrival_time : ℕ := 13 * 60 + 18

-- Define the flight duration in hours and minutes
def flight_duration (h m : ℕ) : Prop :=
  h * 60 + m = arrival_time - departure_time ∧ 0 < m ∧ m < 60

-- Theorem statement
theorem flight_duration_sum :
  ∃ (h m : ℕ), flight_duration h m ∧ h + m = 46 :=
sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l53_5365


namespace NUMINAMATH_CALUDE_bacterium_probability_l53_5351

/-- The probability of selecting a single bacterium in a smaller volume from a larger volume --/
theorem bacterium_probability (total_volume small_volume : ℝ) (h1 : total_volume > 0) 
  (h2 : small_volume > 0) (h3 : small_volume ≤ total_volume) :
  small_volume / total_volume = 0.05 → 
  (total_volume = 2 ∧ small_volume = 0.1) := by
  sorry


end NUMINAMATH_CALUDE_bacterium_probability_l53_5351


namespace NUMINAMATH_CALUDE_cubic_polynomials_constant_term_l53_5343

/-- Given two cubic polynomials with specific root relationships, 
    this theorem states the possible values for the constant term of the first polynomial. -/
theorem cubic_polynomials_constant_term (c d : ℝ) (u v : ℝ) : 
  (∃ w : ℝ, u^3 + c*u + d = 0 ∧ v^3 + c*v + d = 0 ∧ w^3 + c*w + d = 0) →
  (∃ w : ℝ, (u+2)^3 + c*(u+2) + (d-120) = 0 ∧ 
            (v-5)^3 + c*(v-5) + (d-120) = 0 ∧ 
             w^3 + c*w + (d-120) = 0) →
  d = 396 ∨ d = 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_constant_term_l53_5343


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l53_5371

theorem right_triangle_hypotenuse (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : q < p) (hpq2 : p < q * Real.sqrt 1.8) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b = p ∧
    (1/3 : ℝ) * Real.sqrt (a^2 + 4*b^2) + (1/3 : ℝ) * Real.sqrt (4*a^2 + b^2) = q ∧
    c^2 = a^2 + b^2 ∧
    c^2 = (p^4 - 9*q^4) / (2*(p^2 - 5*q^2)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l53_5371


namespace NUMINAMATH_CALUDE_johny_travel_distance_l53_5303

theorem johny_travel_distance (S : ℝ) : 
  S ≥ 0 →
  S + (S + 20) + 2*(S + 20) = 220 →
  S = 40 := by
sorry

end NUMINAMATH_CALUDE_johny_travel_distance_l53_5303


namespace NUMINAMATH_CALUDE_max_cupcakes_eaten_l53_5364

/-- Given 30 cupcakes shared among three people, where one person eats twice as much as the first
    and the same as the second, the maximum number of cupcakes the first person could have eaten is 6. -/
theorem max_cupcakes_eaten (total : ℕ) (ben charles diana : ℕ) : 
  total = 30 →
  diana = 2 * ben →
  diana = charles →
  total = ben + charles + diana →
  ben ≤ 6 ∧ ∃ ben', ben' = 6 ∧ 
    ∃ charles' diana', 
      diana' = 2 * ben' ∧ 
      diana' = charles' ∧ 
      total = ben' + charles' + diana' :=
by sorry

end NUMINAMATH_CALUDE_max_cupcakes_eaten_l53_5364


namespace NUMINAMATH_CALUDE_complex_equation_solution_l53_5398

theorem complex_equation_solution (a : ℝ) : (a + Complex.I)^2 = 2 * Complex.I → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l53_5398


namespace NUMINAMATH_CALUDE_gcf_of_2550_and_7140_l53_5315

theorem gcf_of_2550_and_7140 : Nat.gcd 2550 7140 = 510 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_2550_and_7140_l53_5315


namespace NUMINAMATH_CALUDE_unique_weights_count_l53_5373

/-- Represents the available weights in grams -/
def weights : List Nat := [1, 3, 7]

/-- Computes all possible combinations of weights -/
def allCombinations (weights : List Nat) : List Nat :=
  sorry

/-- Counts the number of unique weights that can be measured -/
def countUniqueWeights (weights : List Nat) : Nat :=
  sorry

/-- Theorem stating that the number of unique weights that can be measured is 7 -/
theorem unique_weights_count :
  countUniqueWeights weights = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_weights_count_l53_5373


namespace NUMINAMATH_CALUDE_min_value_theorem_l53_5369

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b + a * c = 4) :
  (2 / a) + (2 / (b + c)) + (8 / (a + b + c)) ≥ 4 ∧
  ((2 / a) + (2 / (b + c)) + (8 / (a + b + c)) = 4 ↔ a = 2 ∧ b + c = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l53_5369


namespace NUMINAMATH_CALUDE_unique_integer_root_difference_l53_5375

/-- Given an integer n, A = √(n² + 24), and B = √(n² - 9),
    prove that n = 5 is the only value for which A - B is an integer. -/
theorem unique_integer_root_difference (n : ℤ) : 
  (∃ m : ℤ, Real.sqrt (n^2 + 24) - Real.sqrt (n^2 - 9) = m) ↔ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_root_difference_l53_5375


namespace NUMINAMATH_CALUDE_fourth_quadrant_condition_l53_5329

theorem fourth_quadrant_condition (m : ℝ) :
  let z := (m + Complex.I) / (1 + Complex.I)
  (z.re > 0 ∧ z.im < 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_quadrant_condition_l53_5329


namespace NUMINAMATH_CALUDE_blue_yellow_probability_l53_5309

/-- The probability of drawing a blue chip first and then a yellow chip without replacement -/
def draw_blue_then_yellow (blue : ℕ) (yellow : ℕ) : ℚ :=
  (blue : ℚ) / (blue + yellow) * yellow / (blue + yellow - 1)

/-- Theorem stating the probability of drawing a blue chip first and then a yellow chip
    without replacement from a bag containing 10 blue chips and 5 yellow chips -/
theorem blue_yellow_probability :
  draw_blue_then_yellow 10 5 = 5 / 21 := by
  sorry

#eval draw_blue_then_yellow 10 5

end NUMINAMATH_CALUDE_blue_yellow_probability_l53_5309


namespace NUMINAMATH_CALUDE_puppies_adoption_time_l53_5356

theorem puppies_adoption_time (initial_puppies : ℕ) (additional_puppies : ℕ) (adoption_rate : ℕ) :
  initial_puppies = 10 →
  additional_puppies = 15 →
  adoption_rate = 7 →
  (∃ (days : ℕ), days = 4 ∧ days * adoption_rate ≥ initial_puppies + additional_puppies ∧
   (days - 1) * adoption_rate < initial_puppies + additional_puppies) :=
by sorry

end NUMINAMATH_CALUDE_puppies_adoption_time_l53_5356


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_294_490_l53_5383

theorem lcm_gcf_ratio_294_490 : 
  (Nat.lcm 294 490) / (Nat.gcd 294 490) = 15 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_294_490_l53_5383


namespace NUMINAMATH_CALUDE_exp_13pi_over_2_equals_i_l53_5311

-- Define Euler's formula
axiom euler_formula (θ : ℝ) : Complex.exp (θ * Complex.I) = Complex.cos θ + Complex.I * Complex.sin θ

-- State the theorem
theorem exp_13pi_over_2_equals_i : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by sorry

end NUMINAMATH_CALUDE_exp_13pi_over_2_equals_i_l53_5311


namespace NUMINAMATH_CALUDE_crates_delivered_is_twelve_l53_5376

/-- The number of crates of apples delivered to a factory --/
def crates_delivered (apples_per_crate : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) (boxes_filled : ℕ) : ℕ :=
  (boxes_filled * apples_per_box + rotten_apples) / apples_per_crate

/-- Theorem stating that the number of crates delivered is 12 --/
theorem crates_delivered_is_twelve :
  crates_delivered 42 4 10 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_crates_delivered_is_twelve_l53_5376


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_6_over_sqrt_2_between_1_and_2_l53_5316

theorem sqrt_18_minus_sqrt_6_over_sqrt_2_between_1_and_2 :
  1 < (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 ∧
  (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_6_over_sqrt_2_between_1_and_2_l53_5316


namespace NUMINAMATH_CALUDE_wrapping_paper_usage_l53_5338

theorem wrapping_paper_usage (total_fraction : ℚ) (num_presents : ℕ) :
  total_fraction = 3 / 10 →
  num_presents = 3 →
  (total_fraction / num_presents : ℚ) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_usage_l53_5338


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l53_5337

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem geometric_sequence_ratio 
  (a₁ : ℝ) (q : ℝ) (h1 : q > 0) 
  (h2 : (geometric_sequence a₁ q 3) * (geometric_sequence a₁ q 9) = 
        2 * (geometric_sequence a₁ q 5)^2) : 
  q = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l53_5337


namespace NUMINAMATH_CALUDE_inequality_proof_l53_5384

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha₁ : 0 < a₁) (ha₂ : 0 < a₂) (ha₃ : 0 < a₃) 
  (hb₁ : 0 < b₁) (hb₂ : 0 < b₂) (hb₃ : 0 < b₃) : 
  (a₁*b₂ + a₂*b₁ + a₂*b₃ + a₃*b₂ + a₃*b₁ + a₁*b₃)^2 ≥ 
  4*(a₁*a₂ + a₂*a₃ + a₃*a₁)*(b₁*b₂ + b₂*b₃ + b₃*b₁) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l53_5384


namespace NUMINAMATH_CALUDE_correct_verb_forms_l53_5340

-- Define the structure of a sentence
structure Sentence where
  subject : String
  verb1 : String
  verb2 : String

-- Define a predicate for plural subjects
def is_plural (s : String) : Prop := s.endsWith "s"

-- Define a predicate for partial references
def is_partial_reference (s : String) : Prop := s = "some"

-- Define a predicate for plural verb forms
def is_plural_verb (v : String) : Prop := v = "are" ∨ v = "seem"

-- Theorem statement
theorem correct_verb_forms (s : Sentence) 
  (h1 : is_plural s.subject) 
  (h2 : is_partial_reference "some") : 
  is_plural_verb s.verb1 ∧ is_plural_verb s.verb2 := by
  sorry

-- Example usage
def example_sentence : Sentence := {
  subject := "Such phenomena"
  verb1 := "are"
  verb2 := "seem"
}

#check correct_verb_forms example_sentence

end NUMINAMATH_CALUDE_correct_verb_forms_l53_5340


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_l53_5358

/-- Given a point M which is the midpoint of segment AB, and the coordinates of points M and A,
    prove that the sum of the coordinates of point B is 5. -/
theorem sum_coordinates_of_B (M A B : ℝ × ℝ) : 
  M = (2, 5) →  -- M has coordinates (2, 5)
  A = (6, 3) →  -- A has coordinates (6, 3)
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  B.1 + B.2 = 5 := by  -- The sum of B's coordinates is 5
sorry


end NUMINAMATH_CALUDE_sum_coordinates_of_B_l53_5358


namespace NUMINAMATH_CALUDE_total_chips_count_l53_5346

def plain_chips : ℕ := 4
def bbq_chips : ℕ := 5
def probability_3_bbq : ℚ := 5/42

theorem total_chips_count : 
  let total_chips := plain_chips + bbq_chips
  (Nat.choose bbq_chips 3 : ℚ) / (Nat.choose total_chips 3 : ℚ) = probability_3_bbq →
  total_chips = 9 := by sorry

end NUMINAMATH_CALUDE_total_chips_count_l53_5346


namespace NUMINAMATH_CALUDE_smallest_w_l53_5396

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_w (w : ℕ) : w ≥ 676 ↔ 
  (is_divisible (1452 * w) (2^4)) ∧ 
  (is_divisible (1452 * w) (3^3)) ∧ 
  (is_divisible (1452 * w) (13^3)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_w_l53_5396


namespace NUMINAMATH_CALUDE_honey_distribution_l53_5330

/-- Represents the volume of honey in a barrel -/
structure HoneyVolume where
  volume : ℚ
  positive : volume > 0

/-- The volume of honey in a large barrel -/
def large_barrel : HoneyVolume :=
  { volume := 1, positive := by norm_num }

/-- The volume of honey in a small barrel -/
def small_barrel : HoneyVolume :=
  { volume := 5/9, positive := by norm_num }

/-- The total volume of honey in Winnie-the-Pooh's possession -/
def total_honey : ℚ := 25 * large_barrel.volume

theorem honey_distribution (h : 25 * large_barrel.volume = 45 * small_barrel.volume) :
  total_honey = 20 * large_barrel.volume + 9 * small_barrel.volume :=
by sorry

end NUMINAMATH_CALUDE_honey_distribution_l53_5330


namespace NUMINAMATH_CALUDE_power_sum_integer_l53_5320

theorem power_sum_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m := by
  sorry

end NUMINAMATH_CALUDE_power_sum_integer_l53_5320


namespace NUMINAMATH_CALUDE_triangle_area_l53_5366

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 3 * Real.sqrt 2) (h2 : b = 2 * Real.sqrt 3) (h3 : Real.cos C = 1/3) :
  (1/2) * a * b * Real.sin C = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l53_5366


namespace NUMINAMATH_CALUDE_kylie_apple_picking_l53_5300

/-- Represents the number of apples Kylie picked in the first hour -/
def first_hour_apples : ℕ := 66

/-- Theorem stating that given the conditions of Kylie's apple picking,
    she picked 66 apples in the first hour -/
theorem kylie_apple_picking :
  ∃ (x : ℕ), 
    x + 2*x + x/3 = 220 ∧ 
    x = first_hour_apples :=
by sorry

end NUMINAMATH_CALUDE_kylie_apple_picking_l53_5300


namespace NUMINAMATH_CALUDE_segment_length_l53_5301

/-- Given three points A, B, and C on a line, with AB = 4 and BC = 3,
    the length of AC is either 7 or 1. -/
theorem segment_length (A B C : ℝ) : 
  (B - A = 4) → (C - B = 3 ∨ B - C = 3) → (C - A = 7 ∨ C - A = 1) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l53_5301


namespace NUMINAMATH_CALUDE_max_m_value_l53_5336

/-- Given m > 0 and the inequality holds for all x > 0, the maximum value of m is e^2 -/
theorem max_m_value (m : ℝ) (hm : m > 0) 
  (h : ∀ x > 0, m * x * Real.log x - (x + m) * Real.exp ((x - m) / m) ≤ 0) : 
  m ≤ Real.exp 2 ∧ ∃ m₀ > 0, ∀ ε > 0, ∃ x > 0, 
    (Real.exp 2 - ε) * x * Real.log x - (x + (Real.exp 2 - ε)) * Real.exp ((x - (Real.exp 2 - ε)) / (Real.exp 2 - ε)) > 0 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l53_5336


namespace NUMINAMATH_CALUDE_ellipse_extrema_l53_5347

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4*x

-- Define the function z
def z (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem ellipse_extrema :
  ∀ (x y : ℝ), ellipse x y →
    (z x y ≥ -1/5 ∧ z x y ≤ 16) ∧
    (z (2/5) (3/5) = -1/5 ∧ z (2/5) (-3/5) = -1/5 ∧ z 4 0 = 16) :=
sorry

end NUMINAMATH_CALUDE_ellipse_extrema_l53_5347


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_existence_l53_5387

theorem arithmetic_geometric_progression_existence :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ (d : ℝ), b = a + d ∧ c = b + d) ∧
  (∃ (r : ℝ), (a = b ∧ b = c * r) ∨ (a = b * r ∧ b = c) ∨ (a = c ∧ c = b * r)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_existence_l53_5387


namespace NUMINAMATH_CALUDE_xy_value_l53_5323

theorem xy_value (x y : ℚ) 
  (eq1 : 5 * x + 3 * y + 5 = 0) 
  (eq2 : 3 * x + 5 * y - 5 = 0) : 
  x * y = -25 / 4 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l53_5323


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l53_5342

theorem fraction_equation_solution (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 2) (hy1 : y ≠ 0) (hy2 : y ≠ 3) :
  3 / x + 2 / y = 5 / 6 → x = 18 * y / (5 * y - 12) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l53_5342


namespace NUMINAMATH_CALUDE_number_reduced_by_six_times_l53_5381

/-- 
Given a natural number N that does not end in zero, and a digit a (1 ≤ a ≤ 9) in N,
if replacing a with 0 reduces N by 6 times, then N = 12a.
-/
theorem number_reduced_by_six_times (N : ℕ) (a : ℕ) : 
  (1 ≤ a ∧ a ≤ 9) →  -- a is a single digit
  (∃ k : ℕ, N = 12 * 10^k + 2 * a * 10^k) →  -- N has the form 12a in base 10
  (N % 10 ≠ 0) →  -- N does not end in zero
  (∃ N' : ℕ, N' = N / 10^k ∧ N = 6 * (N' - a * 10^k + 0)) →  -- replacing a with 0 reduces N by 6 times
  N = 12 * a := by
sorry

end NUMINAMATH_CALUDE_number_reduced_by_six_times_l53_5381


namespace NUMINAMATH_CALUDE_doughnuts_eaten_l53_5341

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of doughnuts in the box initially -/
def initial_doughnuts : ℕ := 2 * dozen

/-- The number of doughnuts remaining -/
def remaining_doughnuts : ℕ := 16

/-- The number of doughnuts eaten by the family -/
def eaten_doughnuts : ℕ := initial_doughnuts - remaining_doughnuts

theorem doughnuts_eaten : eaten_doughnuts = 8 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_eaten_l53_5341


namespace NUMINAMATH_CALUDE_f_30_value_l53_5335

def is_valid_f (f : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, f (n + 1) > f n) ∧ 
  (∀ m n : ℕ+, f (m * n) = f m * f n) ∧
  (∀ m n : ℕ+, m ≠ n → m ^ (n : ℕ) = n ^ (m : ℕ) → (f m = n ∨ f n = m))

theorem f_30_value (f : ℕ+ → ℕ+) (h : is_valid_f f) : f 30 = 900 := by
  sorry

end NUMINAMATH_CALUDE_f_30_value_l53_5335


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l53_5344

/-- Proves that the cost of each movie ticket is $10.62 --/
theorem movie_ticket_cost (ticket_count : ℕ) (rental_cost movie_cost total_cost : ℚ) :
  ticket_count = 2 →
  rental_cost = 1.59 →
  movie_cost = 13.95 →
  total_cost = 36.78 →
  ∃ (ticket_price : ℚ), 
    ticket_price * ticket_count + rental_cost + movie_cost = total_cost ∧
    ticket_price = 10.62 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l53_5344


namespace NUMINAMATH_CALUDE_total_pizzas_eaten_l53_5327

/-- The number of pizzas eaten by class A -/
def pizzas_class_a : ℕ := 8

/-- The number of pizzas eaten by class B -/
def pizzas_class_b : ℕ := 7

/-- The total number of pizzas eaten by both classes -/
def total_pizzas : ℕ := pizzas_class_a + pizzas_class_b

theorem total_pizzas_eaten : total_pizzas = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_pizzas_eaten_l53_5327


namespace NUMINAMATH_CALUDE_composite_divisor_inequality_l53_5397

def d (k : ℕ) : ℕ := (Nat.divisors k).card

theorem composite_divisor_inequality (n : ℕ) (h_composite : ¬ Nat.Prime n) :
  ∃ m : ℕ, m > 0 ∧ m ∣ n ∧ m * m ≤ n ∧ d n ≤ d m * d m * d m := by
  sorry

end NUMINAMATH_CALUDE_composite_divisor_inequality_l53_5397


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l53_5367

theorem sum_of_roots_equation (x : ℝ) : 
  (3 = (x^3 - 3*x^2 - 12*x) / (x + 3)) → 
  (∃ y : ℝ, (3 = (y^3 - 3*y^2 - 12*y) / (y + 3)) ∧ (x + y = 6)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l53_5367


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l53_5390

theorem opposite_of_negative_two :
  ∃ x : ℝ, (x + (-2) = 0 ∧ x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l53_5390


namespace NUMINAMATH_CALUDE_comic_book_collections_l53_5325

/-- Kymbrea's initial comic book collection size -/
def kymbrea_initial : ℕ := 50

/-- Kymbrea's monthly comic book addition rate -/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection size -/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book addition rate -/
def lashawn_rate : ℕ := 8

/-- The number of months after which LaShawn's collection will be three times Kymbrea's -/
def months : ℕ := 130

theorem comic_book_collections :
  lashawn_initial + lashawn_rate * months = 3 * (kymbrea_initial + kymbrea_rate * months) :=
by sorry

end NUMINAMATH_CALUDE_comic_book_collections_l53_5325


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l53_5349

theorem arithmetic_mean_difference (p q r : ℝ) (G : ℝ) : 
  G = (p * q * r) ^ (1/3) →
  (p + q) / 2 = 10 →
  (q + r) / 2 = 25 →
  r - p = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l53_5349


namespace NUMINAMATH_CALUDE_root_domain_implies_a_bound_l53_5353

/-- The equation has real roots for m in this set -/
def A : Set ℝ := {m | ∃ x, (m + 1) * x^2 - m * x + m - 1 = 0}

/-- The domain of the function f(x) -/
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a > 0}

/-- The main theorem -/
theorem root_domain_implies_a_bound (a : ℝ) : A ⊆ B a → a > 2/3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_root_domain_implies_a_bound_l53_5353


namespace NUMINAMATH_CALUDE_factor_theorem_l53_5326

theorem factor_theorem (p q : ℝ) : 
  (∀ x : ℝ, (x - 3) * (x + 5) = x^2 + p*x + q) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_l53_5326


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l53_5334

theorem profit_percentage_previous_year 
  (revenue_previous : ℝ) 
  (profit_previous : ℝ) 
  (revenue_decline : ℝ) 
  (profit_percentage_current : ℝ) 
  (profit_ratio : ℝ) 
  (h1 : revenue_decline = 0.3)
  (h2 : profit_percentage_current = 0.1)
  (h3 : profit_ratio = 0.6999999999999999)
  (h4 : profit_previous * profit_ratio = 
        (1 - revenue_decline) * revenue_previous * profit_percentage_current) :
  profit_previous / revenue_previous = 0.1 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l53_5334


namespace NUMINAMATH_CALUDE_alyssa_kittens_l53_5304

/-- The number of kittens Alyssa initially had -/
def initial_kittens : ℕ := 8

/-- The number of kittens Alyssa gave to her friends -/
def given_away_kittens : ℕ := 4

/-- The number of kittens Alyssa now has -/
def remaining_kittens : ℕ := initial_kittens - given_away_kittens

theorem alyssa_kittens : remaining_kittens = 4 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_kittens_l53_5304


namespace NUMINAMATH_CALUDE_area_difference_square_inscribed_triangle_l53_5307

/-- Given an isosceles right triangle inscribed in a square, where the hypotenuse
    of the triangle is the diagonal of the square and has length 8√2 cm,
    prove that the area difference between the square and the triangle is 32 cm². -/
theorem area_difference_square_inscribed_triangle :
  ∀ (square_side triangle_side : ℝ),
  square_side * square_side * 2 = (8 * Real.sqrt 2) ^ 2 →
  triangle_side * triangle_side * 2 = (8 * Real.sqrt 2) ^ 2 →
  square_side ^ 2 - (triangle_side ^ 2 / 2) = 32 :=
by sorry

end NUMINAMATH_CALUDE_area_difference_square_inscribed_triangle_l53_5307


namespace NUMINAMATH_CALUDE_flag_arrangement_remainder_l53_5360

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def M : ℕ :=
  13 * Nat.choose 14 10 - 2 * Nat.choose 13 10

/-- The theorem stating the remainder when M is divided by 1000 -/
theorem flag_arrangement_remainder :
  M % 1000 = 441 := by
  sorry

end NUMINAMATH_CALUDE_flag_arrangement_remainder_l53_5360


namespace NUMINAMATH_CALUDE_blackbirds_per_tree_l53_5380

theorem blackbirds_per_tree (num_trees : ℕ) (num_magpies : ℕ) (total_birds : ℕ) 
  (h1 : num_trees = 7)
  (h2 : num_magpies = 13)
  (h3 : total_birds = 34)
  (h4 : ∃ (blackbirds_per_tree : ℕ), num_trees * blackbirds_per_tree + num_magpies = total_birds) :
  ∃ (blackbirds_per_tree : ℕ), blackbirds_per_tree = 3 ∧ 
    num_trees * blackbirds_per_tree + num_magpies = total_birds :=
by
  sorry

end NUMINAMATH_CALUDE_blackbirds_per_tree_l53_5380


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l53_5331

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  is_geometric_sequence a → a 2 = 4 → a 6 = 16 → a 4 = 8 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l53_5331


namespace NUMINAMATH_CALUDE_exactly_three_correct_implies_B_false_l53_5378

-- Define the function f over ℝ
variable (f : ℝ → ℝ)

-- Define the properties stated by each student
def property_A (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≥ f y

def property_B (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≤ f y

def property_C (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + (1 - x)) = f x

def property_D (f : ℝ → ℝ) : Prop :=
  ∃ x, f x < f 0

-- Theorem stating that if exactly three properties are true, then B must be false
theorem exactly_three_correct_implies_B_false (f : ℝ → ℝ) :
  ((property_A f ∧ property_C f ∧ property_D f) ∨
   (property_A f ∧ property_B f ∧ property_C f) ∨
   (property_A f ∧ property_B f ∧ property_D f) ∨
   (property_B f ∧ property_C f ∧ property_D f)) →
  ¬ property_B f :=
sorry

end NUMINAMATH_CALUDE_exactly_three_correct_implies_B_false_l53_5378


namespace NUMINAMATH_CALUDE_jorge_total_goals_l53_5391

theorem jorge_total_goals (last_season_goals this_season_goals : ℕ) 
  (h1 : last_season_goals = 156)
  (h2 : this_season_goals = 187) :
  last_season_goals + this_season_goals = 343 := by
  sorry

end NUMINAMATH_CALUDE_jorge_total_goals_l53_5391


namespace NUMINAMATH_CALUDE_backup_settings_count_l53_5370

/-- Represents the weight of a single piece of silverware in ounces -/
def silverware_weight : ℕ := 4

/-- Represents the number of silverware pieces per setting -/
def silverware_per_setting : ℕ := 3

/-- Represents the weight of a single plate in ounces -/
def plate_weight : ℕ := 12

/-- Represents the number of plates per setting -/
def plates_per_setting : ℕ := 2

/-- Represents the number of tables -/
def num_tables : ℕ := 15

/-- Represents the number of settings per table -/
def settings_per_table : ℕ := 8

/-- Represents the total weight of all settings including backups in ounces -/
def total_weight : ℕ := 5040

/-- Calculates the number of backup settings needed -/
def backup_settings : ℕ := 
  let setting_weight := silverware_weight * silverware_per_setting + plate_weight * plates_per_setting
  let total_settings := num_tables * settings_per_table
  let regular_settings_weight := total_settings * setting_weight
  (total_weight - regular_settings_weight) / setting_weight

theorem backup_settings_count : backup_settings = 20 := by
  sorry

end NUMINAMATH_CALUDE_backup_settings_count_l53_5370


namespace NUMINAMATH_CALUDE_slope_of_AB_l53_5382

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  P.1 = 1 ∧ P.2 = Real.sqrt 2 ∧ parabola P.1 P.2

-- Define complementary inclination angles
def complementary_angles (k : ℝ) (PA PB : ℝ → ℝ) : Prop :=
  (∀ x, PA x = k*(x - 1) + Real.sqrt 2) ∧
  (∀ x, PB x = -k*(x - 1) + Real.sqrt 2)

-- Define intersection points
def intersection_points (A B : ℝ × ℝ) (PA PB : ℝ → ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  A.2 = PA A.1 ∧ B.2 = PB B.1

-- Theorem statement
theorem slope_of_AB (P A B : ℝ × ℝ) (k : ℝ) (PA PB : ℝ → ℝ) :
  point_on_parabola P →
  complementary_angles k PA PB →
  intersection_points A B PA PB →
  (B.2 - A.2) / (B.1 - A.1) = -2 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_slope_of_AB_l53_5382


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l53_5357

theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c ≥ 0 ↔ -1 ≤ x ∧ x ≤ 2) :
  a + b = 0 ∧ a + b + c > 0 ∧ c > 0 ∧ b > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l53_5357


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l53_5333

/-- An isosceles trapezoid with specific dimensions -/
structure IsoscelesTrapezoid where
  smallBase : ℝ
  largeBase : ℝ
  acuteAngle : ℝ
  heightPerp : ℝ

/-- A solid of revolution formed by rotating an isosceles trapezoid around its smaller base -/
structure SolidOfRevolution where
  trapezoid : IsoscelesTrapezoid
  surfaceArea : ℝ
  volume : ℝ

/-- The theorem stating the surface area and volume of the solid of revolution -/
theorem isosceles_trapezoid_rotation (t : IsoscelesTrapezoid) 
  (h1 : t.smallBase = 2)
  (h2 : t.largeBase = 3)
  (h3 : t.acuteAngle = π / 3)
  (h4 : t.heightPerp = 3) :
  ∃ (s : SolidOfRevolution), 
    s.trapezoid = t ∧ 
    s.surfaceArea = 4 * π * Real.sqrt 3 ∧ 
    s.volume = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l53_5333


namespace NUMINAMATH_CALUDE_shaded_area_in_circle_configuration_l53_5310

/-- The area of the shaded region in a circle configuration --/
theorem shaded_area_in_circle_configuration (R : ℝ) (h : R = 9) :
  let r : ℝ := R / 2
  let larger_circle_area : ℝ := π * R^2
  let smaller_circle_area : ℝ := π * r^2
  let total_smaller_circles_area : ℝ := 3 * smaller_circle_area
  let shaded_area : ℝ := larger_circle_area - total_smaller_circles_area
  shaded_area = 20.25 * π :=
by sorry


end NUMINAMATH_CALUDE_shaded_area_in_circle_configuration_l53_5310


namespace NUMINAMATH_CALUDE_three_fourths_cubed_l53_5314

theorem three_fourths_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end NUMINAMATH_CALUDE_three_fourths_cubed_l53_5314


namespace NUMINAMATH_CALUDE_no_right_triangle_with_integer_side_l53_5345

theorem no_right_triangle_with_integer_side : 
  ¬ ∃ (x : ℤ), 
    (12 < x ∧ x < 30) ∧ 
    (x^2 = 12^2 + 30^2 ∨ 30^2 = 12^2 + x^2 ∨ 12^2 = 30^2 + x^2) :=
by sorry

#check no_right_triangle_with_integer_side

end NUMINAMATH_CALUDE_no_right_triangle_with_integer_side_l53_5345


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l53_5317

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10 % 10) + (n % 10)

theorem sum_of_digits_of_power : sum_of_digits ((3 + 4) ^ 11) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l53_5317
