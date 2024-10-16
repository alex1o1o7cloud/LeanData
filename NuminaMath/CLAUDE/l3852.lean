import Mathlib

namespace NUMINAMATH_CALUDE_sweater_discount_percentage_l3852_385214

theorem sweater_discount_percentage (final_price saved : ℝ) : 
  final_price = 27 → saved = 3 → (saved / (final_price + saved)) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_sweater_discount_percentage_l3852_385214


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3852_385281

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a + b + c = 40 →
  (1/2) * a * b = 24 →
  a^2 + b^2 = c^2 →
  c = 18.8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3852_385281


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_area_l3852_385236

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  K : Point3D
  L : Point3D
  M : Point3D
  N : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Calculates the Euclidean distance between two points -/
def distance (p q : Point3D) : ℝ := sorry

/-- Calculates the angle between three points -/
def angle (p q r : Point3D) : ℝ := sorry

/-- Calculates the spherical distance between two points on a sphere -/
def sphericalDistance (s : Sphere) (p q : Point3D) : ℝ := sorry

/-- Checks if a point is on a sphere -/
def isOnSphere (s : Sphere) (p : Point3D) : Prop := sorry

/-- The set of points on the sphere satisfying the distance condition -/
def distanceSet (s : Sphere) (t : Tetrahedron) : Set Point3D :=
  {p : Point3D | isOnSphere s p ∧ 
    sphericalDistance s p t.K + sphericalDistance s p t.L + 
    sphericalDistance s p t.M + sphericalDistance s p t.N ≤ 6 * Real.pi}

/-- Calculates the area of a set on a sphere -/
def sphericalArea (s : Sphere) (set : Set Point3D) : ℝ := sorry

theorem tetrahedron_sphere_area 
  (t : Tetrahedron) 
  (s : Sphere) 
  (h1 : distance t.K t.L = 5)
  (h2 : distance t.N t.M = 6)
  (h3 : angle t.L t.M t.N = 35 * Real.pi / 180)
  (h4 : angle t.K t.N t.M = 35 * Real.pi / 180)
  (h5 : angle t.L t.N t.M = 55 * Real.pi / 180)
  (h6 : angle t.K t.M t.N = 55 * Real.pi / 180)
  (h7 : isOnSphere s t.K ∧ isOnSphere s t.L ∧ isOnSphere s t.M ∧ isOnSphere s t.N) :
  sphericalArea s (distanceSet s t) = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_area_l3852_385236


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_sum_sqrt_l3852_385284

theorem sqrt_sum_equals_sqrt_of_sum_sqrt (a b : ℚ) :
  (Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3)) ↔
  ((a = 1/2 ∧ b = 3/2) ∨ (a = 3/2 ∧ b = 1/2)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_sum_sqrt_l3852_385284


namespace NUMINAMATH_CALUDE_square_difference_of_roots_l3852_385280

theorem square_difference_of_roots (α β : ℝ) : 
  α ≠ β ∧ α^2 - 3*α + 1 = 0 ∧ β^2 - 3*β + 1 = 0 → (α - β)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_roots_l3852_385280


namespace NUMINAMATH_CALUDE_exists_special_function_l3852_385216

/-- The closed interval [0, 1] -/
def ClosedUnitInterval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

/-- A continuous function from [0, 1] to [0, 1] -/
def ContinuousUnitFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ ∀ x ∈ ClosedUnitInterval, f x ∈ ClosedUnitInterval

/-- A line in ℝ² -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The number of intersections between a function and a line -/
def NumberOfIntersections (f : ℝ → ℝ) (l : Line) : ℕ := sorry

/-- The existence of a function with the required properties -/
theorem exists_special_function :
  ∃ g : ℝ → ℝ,
    ContinuousUnitFunction g ∧
    (∀ l : Line, NumberOfIntersections g l < ω) ∧
    (∀ n : ℕ, ∃ l : Line, NumberOfIntersections g l > n) :=
sorry

end NUMINAMATH_CALUDE_exists_special_function_l3852_385216


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l3852_385275

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the third side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that line segments 4, 5, and 9 cannot form a triangle -/
theorem cannot_form_triangle : ¬(can_form_triangle 4 5 9) := by
  sorry


end NUMINAMATH_CALUDE_cannot_form_triangle_l3852_385275


namespace NUMINAMATH_CALUDE_perfume_price_with_tax_l3852_385282

/-- Calculates the total price including tax given the original price and tax rate. -/
def totalPriceWithTax (originalPrice taxRate : ℝ) : ℝ :=
  originalPrice * (1 + taxRate)

/-- Theorem stating that for a product with an original price of $92 and a tax rate of 7.5%,
    the total price including tax is $98.90. -/
theorem perfume_price_with_tax :
  totalPriceWithTax 92 0.075 = 98.90 := by
  sorry

end NUMINAMATH_CALUDE_perfume_price_with_tax_l3852_385282


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3852_385201

-- Part 1
theorem simplify_expression_1 (x : ℝ) (hx : x ≠ 0) :
  5 * x^2 * x^4 + x^8 / (-x)^2 = 6 * x^6 :=
sorry

-- Part 2
theorem simplify_expression_2 (x y : ℝ) (hx : x ≠ 0) :
  ((x + y)^2 - (x - y)^2) / (2 * x) = 2 * y :=
sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3852_385201


namespace NUMINAMATH_CALUDE_points_lost_in_last_round_l3852_385256

-- Define the variables
def first_round_points : ℕ := 17
def second_round_points : ℕ := 6
def final_points : ℕ := 7

-- Define the theorem
theorem points_lost_in_last_round :
  (first_round_points + second_round_points) - final_points = 16 := by
  sorry

end NUMINAMATH_CALUDE_points_lost_in_last_round_l3852_385256


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3852_385269

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (a + Complex.I) / (1 + 2 * Complex.I)) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3852_385269


namespace NUMINAMATH_CALUDE_pet_store_birds_l3852_385299

/-- Calculates the total number of birds in a pet store with the given conditions -/
def totalBirds (totalCages : Nat) (emptyCages : Nat) (initialParrots : Nat) (initialParakeets : Nat) : Nat :=
  let nonEmptyCages := totalCages - emptyCages
  let parrotSum := nonEmptyCages * (2 * initialParrots + (nonEmptyCages - 1)) / 2
  let parakeetSum := nonEmptyCages * (2 * initialParakeets + 2 * (nonEmptyCages - 1)) / 2
  parrotSum + parakeetSum

/-- Theorem stating that the total number of birds in the pet store is 399 -/
theorem pet_store_birds : totalBirds 17 3 2 7 = 399 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l3852_385299


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3852_385266

/-- Proves that a train of given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 215) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3852_385266


namespace NUMINAMATH_CALUDE_whitewashing_cost_calculation_l3852_385278

/-- Calculates the cost of white washing a room with given specifications. -/
def whitewashingCost (roomLength roomWidth roomHeight : ℝ)
                     (doorCount doorLength doorWidth : ℝ)
                     (windowCount windowLength windowWidth : ℝ)
                     (costPerSqFt additionalPaintPercentage : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength + roomWidth) * roomHeight
  let doorArea := doorCount * doorLength * doorWidth
  let windowArea := windowCount * windowLength * windowWidth
  let paintableArea := wallArea - doorArea - windowArea
  let totalPaintArea := paintableArea * (1 + additionalPaintPercentage)
  totalPaintArea * costPerSqFt

/-- Theorem stating the cost of white washing the room with given specifications. -/
theorem whitewashing_cost_calculation :
  whitewashingCost 25 15 12 2 6 3 5 4 3 7 0.1 = 6652.8 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_calculation_l3852_385278


namespace NUMINAMATH_CALUDE_total_carvings_eq_56_l3852_385242

/-- The number of wood carvings that can be contained in each shelf -/
def carvings_per_shelf : ℕ := 8

/-- The number of shelves filled with carvings -/
def filled_shelves : ℕ := 7

/-- The total number of wood carvings displayed -/
def total_carvings : ℕ := carvings_per_shelf * filled_shelves

theorem total_carvings_eq_56 : total_carvings = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_carvings_eq_56_l3852_385242


namespace NUMINAMATH_CALUDE_remainder_of_sum_first_six_primes_div_seventh_prime_l3852_385277

-- Define the first seven prime numbers
def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

-- Define the sum of the first six primes
def sum_first_six_primes : Nat := (first_seven_primes.take 6).sum

-- Define the seventh prime
def seventh_prime : Nat := first_seven_primes[6]

-- Theorem statement
theorem remainder_of_sum_first_six_primes_div_seventh_prime :
  sum_first_six_primes % seventh_prime = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_first_six_primes_div_seventh_prime_l3852_385277


namespace NUMINAMATH_CALUDE_right_triangle_area_l3852_385202

theorem right_triangle_area (a b : ℝ) (h1 : a = 6) (h2 : b = 8) :
  (1/2 : ℝ) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3852_385202


namespace NUMINAMATH_CALUDE_vertex_y_coordinate_l3852_385274

/-- The y-coordinate of the vertex of the parabola y = 3x^2 - 6x + 2 is -1 -/
theorem vertex_y_coordinate (x y : ℝ) : 
  y = 3 * x^2 - 6 * x + 2 → 
  ∃ x₀, ∀ x', 3 * x'^2 - 6 * x' + 2 ≥ 3 * x₀^2 - 6 * x₀ + 2 ∧ 
            3 * x₀^2 - 6 * x₀ + 2 = -1 :=
by sorry

end NUMINAMATH_CALUDE_vertex_y_coordinate_l3852_385274


namespace NUMINAMATH_CALUDE_line_equation_sum_l3852_385222

/-- Proves that for a line with slope 8 passing through the point (-2, 4),
    if its equation is of the form y = mx + b, then m + b = 28. -/
theorem line_equation_sum (m b : ℝ) : 
  m = 8 ∧ 4 = m * (-2) + b → m + b = 28 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_sum_l3852_385222


namespace NUMINAMATH_CALUDE_soccer_league_total_games_l3852_385262

theorem soccer_league_total_games (n : ℕ) (m : ℕ) (p : ℕ) :
  n = 20 →  -- number of teams
  m = 3 →   -- number of mid-season tournament matches per team
  p = 8 →   -- number of teams in playoffs
  (n * (n - 1) / 2) + (n * m / 2) + (p - 1) * 2 = 454 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_total_games_l3852_385262


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_perpendicular_line_proof_l3852_385263

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point 
  (given_line : Line) 
  (point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 3 ∧ 
  given_line.b = -4 ∧ 
  given_line.c = 6 ∧
  point.x = 4 ∧ 
  point.y = -1 ∧
  result_line.a = 4 ∧ 
  result_line.b = 3 ∧ 
  result_line.c = -13 ∧
  point.liesOn result_line ∧
  Line.perpendicular given_line result_line

-- The proof of the theorem
theorem perpendicular_line_proof : 
  ∃ (given_line result_line : Line) (point : Point),
  perpendicular_line_through_point given_line point result_line := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_perpendicular_line_proof_l3852_385263


namespace NUMINAMATH_CALUDE_equation_condition_l3852_385234

theorem equation_condition (a b c : ℕ+) (hb : b < 12) (hc : c < 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c ↔ b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_condition_l3852_385234


namespace NUMINAMATH_CALUDE_product_of_terms_l3852_385204

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 = 2 →
  a 2 * a 3 * a 5 * a 6 = 16 := by
sorry

end NUMINAMATH_CALUDE_product_of_terms_l3852_385204


namespace NUMINAMATH_CALUDE_calendar_reuse_calendar_2032_reuse_l3852_385261

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

def calendar_repeat_cycle : ℕ := 28

theorem calendar_reuse (start_year : ℕ) (reuse_year : ℕ) : Prop :=
  start_year = 2032 →
  is_leap_year start_year →
  reuse_year > start_year →
  is_leap_year reuse_year →
  reuse_year - start_year = calendar_repeat_cycle * ((reuse_year - start_year) / calendar_repeat_cycle) →
  reuse_year = 2060

theorem calendar_2032_reuse :
  ∃ (reuse_year : ℕ), calendar_reuse 2032 reuse_year :=
sorry

end NUMINAMATH_CALUDE_calendar_reuse_calendar_2032_reuse_l3852_385261


namespace NUMINAMATH_CALUDE_transformed_graph_point_l3852_385254

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem transformed_graph_point (h : f 4 = 7) : 
  ∃ (x y : ℝ), 2 * y = 3 * f (4 * x) + 5 ∧ x + y = 14 := by
sorry

end NUMINAMATH_CALUDE_transformed_graph_point_l3852_385254


namespace NUMINAMATH_CALUDE_min_value_expression_l3852_385233

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) ≥ 8 ∧
  ((x^2 / (y - 1)) + (y^2 / (x - 1)) = 8 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3852_385233


namespace NUMINAMATH_CALUDE_min_value_expression_l3852_385231

/-- The minimum value of a specific expression given certain constraints -/
theorem min_value_expression (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) 
  (h_prod : x * y * z * w = 3) :
  x^2 + 4*x*y + 9*y^2 + 6*y*z + 8*z^2 + 3*x*w + 4*w^2 ≥ 81.25 ∧ 
  ∃ (x₀ y₀ z₀ w₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ w₀ > 0 ∧ 
    x₀ * y₀ * z₀ * w₀ = 3 ∧
    x₀^2 + 4*x₀*y₀ + 9*y₀^2 + 6*y₀*z₀ + 8*z₀^2 + 3*x₀*w₀ + 4*w₀^2 = 81.25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3852_385231


namespace NUMINAMATH_CALUDE_binary_1010001011_equals_base7_1620_l3852_385272

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem binary_1010001011_equals_base7_1620 :
  decimal_to_base7 (binary_to_decimal [true, true, false, true, false, false, false, false, true, false]) = [1, 6, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_binary_1010001011_equals_base7_1620_l3852_385272


namespace NUMINAMATH_CALUDE_required_speed_increase_l3852_385225

/-- Represents the travel scenario for Ivan's commute --/
structure TravelScenario where
  usual_travel_time : ℝ
  usual_speed : ℝ
  late_start : ℝ
  speed_increase : ℝ
  time_saved : ℝ

/-- The theorem stating the required speed increase to arrive on time --/
theorem required_speed_increase (scenario : TravelScenario) 
  (h1 : scenario.late_start = 40)
  (h2 : scenario.speed_increase = 0.6)
  (h3 : scenario.time_saved = 65)
  (h4 : scenario.usual_travel_time / (1 + scenario.speed_increase) = 
        scenario.usual_travel_time - scenario.time_saved) :
  (scenario.usual_travel_time / (scenario.usual_travel_time - scenario.late_start)) - 1 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_required_speed_increase_l3852_385225


namespace NUMINAMATH_CALUDE_dana_saturday_hours_l3852_385293

theorem dana_saturday_hours
  (hourly_rate : ℕ)
  (friday_hours : ℕ)
  (sunday_hours : ℕ)
  (total_earnings : ℕ)
  (h1 : hourly_rate = 13)
  (h2 : friday_hours = 9)
  (h3 : sunday_hours = 3)
  (h4 : total_earnings = 286) :
  (total_earnings - (friday_hours + sunday_hours) * hourly_rate) / hourly_rate = 10 :=
by sorry

end NUMINAMATH_CALUDE_dana_saturday_hours_l3852_385293


namespace NUMINAMATH_CALUDE_negation_forall_geq_zero_equivalent_exists_lt_zero_l3852_385221

theorem negation_forall_geq_zero_equivalent_exists_lt_zero :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_forall_geq_zero_equivalent_exists_lt_zero_l3852_385221


namespace NUMINAMATH_CALUDE_garage_wheels_l3852_385205

/-- The number of bikes that can be assembled -/
def bikes_assembled : ℕ := 7

/-- The number of wheels required for each bike -/
def wheels_per_bike : ℕ := 2

/-- Theorem: The total number of bike wheels in the garage is 14 -/
theorem garage_wheels : bikes_assembled * wheels_per_bike = 14 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_l3852_385205


namespace NUMINAMATH_CALUDE_cosh_inequality_l3852_385267

theorem cosh_inequality (c : ℝ) :
  (∀ x : ℝ, (Real.exp x + Real.exp (-x)) / 2 ≤ Real.exp (c * x^2)) ↔ c ≥ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_cosh_inequality_l3852_385267


namespace NUMINAMATH_CALUDE_two_digit_sum_theorem_l3852_385258

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : tens ≤ 9 ∧ ones ≤ 9

/-- The sum of five identical two-digit numbers equals another two-digit number -/
def sum_property (ab mb : TwoDigitNumber) : Prop :=
  5 * (10 * ab.tens + ab.ones) = 10 * mb.tens + mb.ones

/-- Different letters represent different digits -/
def different_digits (ab mb : TwoDigitNumber) : Prop :=
  ab.tens ≠ ab.ones ∧ 
  (ab.tens ≠ mb.tens ∨ ab.ones ≠ mb.ones)

theorem two_digit_sum_theorem (ab mb : TwoDigitNumber) 
  (h_sum : sum_property ab mb) 
  (h_diff : different_digits ab mb) : 
  (ab.tens = 1 ∧ ab.ones = 0) ∨ (ab.tens = 1 ∧ ab.ones = 5) :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_theorem_l3852_385258


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l3852_385218

/-- Represents the capacity of each pitcher in milliliters -/
def pitcher_capacity : ℕ := 800

/-- Represents the fraction of orange juice in the first pitcher -/
def first_pitcher_fraction : ℚ := 1/2

/-- Represents the fraction of orange juice in the second pitcher -/
def second_pitcher_fraction : ℚ := 1/4

/-- Calculates the total volume of orange juice in both pitchers -/
def total_orange_juice : ℚ := 
  pitcher_capacity * first_pitcher_fraction + pitcher_capacity * second_pitcher_fraction

/-- Calculates the total volume of the mixture after filling both pitchers completely -/
def total_mixture : ℕ := 2 * pitcher_capacity

/-- Theorem stating that the fraction of orange juice in the final mixture is 3/8 -/
theorem orange_juice_fraction : 
  (total_orange_juice : ℚ) / (total_mixture : ℚ) = 3/8 := by sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l3852_385218


namespace NUMINAMATH_CALUDE_exam_average_l3852_385290

theorem exam_average (n1 n2 : ℕ) (avg1 avg2 : ℚ) (h1 : n1 = 15) (h2 : n2 = 10) 
  (h3 : avg1 = 80/100) (h4 : avg2 = 90/100) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 84/100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l3852_385290


namespace NUMINAMATH_CALUDE_election_defeat_margin_l3852_385291

theorem election_defeat_margin 
  (total_votes : ℕ) 
  (invalid_votes : ℕ) 
  (defeated_percentage : ℚ) :
  total_votes = 90830 →
  invalid_votes = 83 →
  defeated_percentage = 45/100 →
  ∃ (valid_votes winning_votes losing_votes : ℕ),
    valid_votes = total_votes - invalid_votes ∧
    losing_votes = ⌊(defeated_percentage : ℝ) * valid_votes⌋ ∧
    winning_votes = valid_votes - losing_votes ∧
    winning_votes - losing_votes = 9074 :=
by sorry

end NUMINAMATH_CALUDE_election_defeat_margin_l3852_385291


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3852_385270

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 3x^2 - 6 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3852_385270


namespace NUMINAMATH_CALUDE_shopping_visit_problem_l3852_385240

theorem shopping_visit_problem (
  num_stores : ℕ
  ) (total_visits : ℕ)
  (two_store_visitors : ℕ)
  (h1 : num_stores = 8)
  (h2 : total_visits = 21)
  (h3 : two_store_visitors = 8)
  (h4 : two_store_visitors * 2 ≤ total_visits) :
  ∃ (max_stores_visited : ℕ) (total_shoppers : ℕ),
    max_stores_visited = 5 ∧
    total_shoppers = 9 ∧
    max_stores_visited ≤ num_stores ∧
    total_shoppers * 1 ≤ total_visits ∧
    total_shoppers ≥ two_store_visitors + 1 :=
by sorry

end NUMINAMATH_CALUDE_shopping_visit_problem_l3852_385240


namespace NUMINAMATH_CALUDE_tan_function_property_l3852_385294

theorem tan_function_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * (x - c)) = a * Real.tan (b * (x - c) + π)) →  -- period is π/4
  (a * Real.tan (b * (π/3 - c)) = -4) →  -- passes through (π/3, -4)
  (b * (π/4 - c) = π/2) →  -- vertical asymptote at x = π/4
  4 * a * b = 64 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_function_property_l3852_385294


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_199_l3852_385243

theorem inverse_of_3_mod_199 : ∃ x : ℕ, x < 199 ∧ (3 * x) % 199 = 1 :=
by
  use 133
  sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_199_l3852_385243


namespace NUMINAMATH_CALUDE_range_of_m_l3852_385298

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ∈ Set.range f) →  -- domain of f is [-2, 2]
  (∀ x y, x ∈ Set.Icc (-2 : ℝ) 2 → y ∈ Set.Icc (-2 : ℝ) 2 → x < y → f x < f y) →  -- f is increasing on [-2, 2]
  f (1 - m) < f m →  -- given condition
  m ∈ Set.Ioo (1/2 : ℝ) 2 :=  -- conclusion: m is in the open interval (1/2, 2]
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3852_385298


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l3852_385287

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 6 = 0 → n ≤ 990 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l3852_385287


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_zero_l3852_385238

theorem function_equality_implies_sum_zero
  (A B : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_f : ∀ x, f x = A * x^2 + B)
  (h_g : ∀ x, g x = B * x^2 + A)
  (h_neq : A ≠ B)
  (h_eq : ∀ x, f (g x) - g (f x) = 3 * (B - A)) :
  A + B = 0 := by
sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_zero_l3852_385238


namespace NUMINAMATH_CALUDE_multiple_problem_l3852_385239

theorem multiple_problem (smaller larger : ℝ) (multiple : ℕ) : 
  smaller + larger = 24 →
  smaller = 10 →
  7 * smaller = multiple * larger →
  multiple = 5 := by
sorry

end NUMINAMATH_CALUDE_multiple_problem_l3852_385239


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l3852_385265

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l3852_385265


namespace NUMINAMATH_CALUDE_polynomial_roots_comparison_l3852_385255

theorem polynomial_roots_comparison (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_order_a : a₁ ≤ a₂ ∧ a₂ ≤ a₃)
  (h_order_b : b₁ ≤ b₂ ∧ b₂ ≤ b₃)
  (h_sum : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
  (h_prod : a₁*a₂ + a₂*a₃ + a₁*a₃ = b₁*b₂ + b₂*b₃ + b₁*b₃)
  (h_first : a₁ ≤ b₁) :
  a₃ ≤ b₃ := by sorry

end NUMINAMATH_CALUDE_polynomial_roots_comparison_l3852_385255


namespace NUMINAMATH_CALUDE_cube_volume_l3852_385212

/-- Represents a rectangular box with given dimensions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ :=
  b.length * b.width * b.height

/-- Represents the problem setup -/
def problemSetup : Box × ℕ :=
  (⟨7, 18, 3⟩, 42)

/-- Theorem stating the volume of each cube -/
theorem cube_volume (box : Box) (num_cubes : ℕ) 
  (h1 : box = problemSetup.1) 
  (h2 : num_cubes = problemSetup.2) : 
  (boxVolume box) / num_cubes = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l3852_385212


namespace NUMINAMATH_CALUDE_simplified_fraction_equals_ten_l3852_385215

theorem simplified_fraction_equals_ten (x y z : ℝ) 
  (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (15 * x^2 * y^4 * z^2) / (9 * x * y^3 * z) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_equals_ten_l3852_385215


namespace NUMINAMATH_CALUDE_parallelepiped_edge_length_l3852_385251

/-- A rectangular parallelepiped constructed from unit cubes -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ
  volume : ℕ
  edge_min : ℕ
  total_cubes : ℕ

/-- The total length of all edges of a rectangular parallelepiped -/
def total_edge_length (p : Parallelepiped) : ℕ :=
  4 * (p.length + p.width + p.height)

/-- Theorem: The total edge length of the specific parallelepiped is 96 cm -/
theorem parallelepiped_edge_length :
  ∀ (p : Parallelepiped),
    p.volume = p.length * p.width * p.height →
    p.total_cubes = 440 →
    p.edge_min = 5 →
    p.length ≥ p.edge_min →
    p.width ≥ p.edge_min →
    p.height ≥ p.edge_min →
    total_edge_length p = 96 := by
  sorry

#check parallelepiped_edge_length

end NUMINAMATH_CALUDE_parallelepiped_edge_length_l3852_385251


namespace NUMINAMATH_CALUDE_equal_angles_in_intersecting_circles_l3852_385241

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point type
def Point := ℝ × ℝ

-- Define the angle type
def Angle := ℝ

-- Define the function to check if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Define the function to check if a point lies on a circle
def on_circle (p : Point) (c : Circle) : Prop := sorry

-- Define the function to calculate the angle between three points
def angle (p1 p2 p3 : Point) : Angle := sorry

-- Define the theorem
theorem equal_angles_in_intersecting_circles 
  (c1 c2 : Circle) 
  (K M A B C D : Point) : 
  (∃ (K M : Point), on_circle K c1 ∧ on_circle K c2 ∧ on_circle M c1 ∧ on_circle M c2) →
  (on_circle A c1 ∧ on_circle B c2 ∧ collinear K A B) →
  (on_circle C c1 ∧ on_circle D c2 ∧ collinear K C D) →
  angle M A B = angle M C D := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_in_intersecting_circles_l3852_385241


namespace NUMINAMATH_CALUDE_absolute_value_sum_inequality_l3852_385210

theorem absolute_value_sum_inequality (b : ℝ) :
  (∃ x : ℝ, |x - 5| + |x - 10| > b) ↔ (0 < b ∧ b < 5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_inequality_l3852_385210


namespace NUMINAMATH_CALUDE_greatest_base_seven_digit_sum_proof_l3852_385245

/-- The greatest possible sum of the digits in the base-seven representation of a positive integer less than 2019 -/
def greatest_base_seven_digit_sum : ℕ := 22

/-- A function that converts a natural number to its base-seven representation -/
def to_base_seven (n : ℕ) : List ℕ := sorry

/-- A function that calculates the sum of digits in a list -/
def digit_sum (digits : List ℕ) : ℕ := sorry

theorem greatest_base_seven_digit_sum_proof :
  ∀ n : ℕ, 0 < n → n < 2019 →
  digit_sum (to_base_seven n) ≤ greatest_base_seven_digit_sum ∧
  ∃ m : ℕ, 0 < m ∧ m < 2019 ∧ digit_sum (to_base_seven m) = greatest_base_seven_digit_sum :=
sorry

end NUMINAMATH_CALUDE_greatest_base_seven_digit_sum_proof_l3852_385245


namespace NUMINAMATH_CALUDE_investment_ratio_l3852_385295

def investment_x : ℤ := 5000
def investment_y : ℤ := 15000

theorem investment_ratio :
  (investment_x : ℚ) / investment_y = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_investment_ratio_l3852_385295


namespace NUMINAMATH_CALUDE_picnic_group_size_l3852_385253

theorem picnic_group_size (initial_group : ℕ) (new_avg_age : ℝ) (final_avg_age : ℝ) :
  initial_group = 15 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  ∃ (new_members : ℕ), 
    new_members = 1 ∧
    (initial_group : ℝ) * final_avg_age = 
      (initial_group : ℝ) * new_avg_age + (new_members : ℝ) * (final_avg_age - new_avg_age) :=
by sorry

end NUMINAMATH_CALUDE_picnic_group_size_l3852_385253


namespace NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l3852_385252

theorem degenerate_ellipse_max_y_coordinate (x y : ℝ) :
  (x - 3)^2 / 49 + (y - 2)^2 / 25 = 0 → y ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l3852_385252


namespace NUMINAMATH_CALUDE_omega_range_l3852_385207

theorem omega_range (ω : ℝ) (h1 : ω > 1/4) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x) - Real.cos (ω * x)
  (∀ k : ℤ, (3*π/4 + k*π) / ω ∉ Set.Ioo (2*π) (3*π)) →
  ω ∈ Set.union (Set.Icc (3/8) (7/12)) (Set.Icc (7/8) (11/12)) :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l3852_385207


namespace NUMINAMATH_CALUDE_subset_of_square_eq_self_l3852_385235

theorem subset_of_square_eq_self : {1} ⊆ {x : ℝ | x^2 = x} := by sorry

end NUMINAMATH_CALUDE_subset_of_square_eq_self_l3852_385235


namespace NUMINAMATH_CALUDE_acute_angle_through_point_l3852_385230

theorem acute_angle_through_point (α : Real) : 
  α > 0 ∧ α < Real.pi/2 →
  (∃ (r : Real), r > 0 ∧ r * (Real.cos α) = Real.cos (40 * Real.pi/180) + 1 ∧ 
                            r * (Real.sin α) = Real.sin (40 * Real.pi/180)) →
  α = 20 * Real.pi/180 := by
sorry

end NUMINAMATH_CALUDE_acute_angle_through_point_l3852_385230


namespace NUMINAMATH_CALUDE_circle_equation_l3852_385208

/-- Given a circle C with center (a, 0) tangent to the line y = (√3/3)x at point N(3, √3),
    prove that the equation of circle C is (x-4)² + y² = 4 -/
theorem circle_equation (a : ℝ) :
  let C : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = ((3 - a)^2 + 3)}
  let l : Set (ℝ × ℝ) := {p | p.2 = (Real.sqrt 3 / 3) * p.1}
  let N : ℝ × ℝ := (3, Real.sqrt 3)
  (N ∈ C) ∧ (N ∈ l) ∧ (∀ p ∈ C, p ≠ N → p ∉ l) →
  C = {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 4} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3852_385208


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3852_385229

theorem diophantine_equation_solutions :
  ∀ a b : ℕ, a^2 = b * (b + 7) → (a = 12 ∧ b = 9) ∨ (a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3852_385229


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3852_385297

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by
  sorry

theorem negation_of_specific_proposition : 
  (¬ ∀ x : ℝ, x^2 + x > 2) ↔ (∃ x : ℝ, x^2 + x ≤ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3852_385297


namespace NUMINAMATH_CALUDE_arthur_leftover_is_four_l3852_385289

/-- The amount of money Arthur has leftover after selling his basketball cards and buying comic books -/
def arthursLeftover (cardValue : ℚ) (numCards : ℕ) (comicBookPrice : ℚ) : ℚ :=
  let totalCardValue := cardValue * numCards
  let numComicBooks := (totalCardValue / comicBookPrice).floor
  totalCardValue - numComicBooks * comicBookPrice

/-- Theorem stating that Arthur will have $4 leftover -/
theorem arthur_leftover_is_four :
  arthursLeftover (5/100) 2000 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arthur_leftover_is_four_l3852_385289


namespace NUMINAMATH_CALUDE_hexadecagon_diagonals_l3852_385200

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexadecagon is a 16-sided polygon -/
def hexadecagon_sides : ℕ := 16

theorem hexadecagon_diagonals :
  num_diagonals hexadecagon_sides = 104 := by
  sorry

end NUMINAMATH_CALUDE_hexadecagon_diagonals_l3852_385200


namespace NUMINAMATH_CALUDE_square_point_B_coordinates_l3852_385276

/-- A square in a 2D plane -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Predicate to check if a line is parallel to the x-axis -/
def parallelToXAxis (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : Prop :=
  p1.2 = p2.2

theorem square_point_B_coordinates :
  ∀ (s : Square),
    s.A = (1, -2) →
    s.C = (4, 1) →
    parallelToXAxis s.A s.B →
    s.B = (4, -2) := by
  sorry

end NUMINAMATH_CALUDE_square_point_B_coordinates_l3852_385276


namespace NUMINAMATH_CALUDE_max_marks_calculation_l3852_385285

/-- Given a passing threshold, a student's score, and the shortfall to pass,
    calculate the maximum possible marks. -/
theorem max_marks_calculation (passing_threshold : ℚ) (score : ℕ) (shortfall : ℕ) :
  passing_threshold = 30 / 100 →
  score = 212 →
  shortfall = 28 →
  (score + shortfall) / passing_threshold = 800 :=
by sorry

end NUMINAMATH_CALUDE_max_marks_calculation_l3852_385285


namespace NUMINAMATH_CALUDE_litter_patrol_cans_l3852_385296

theorem litter_patrol_cans (total_litter : ℕ) (glass_bottles : ℕ) (aluminum_cans : ℕ) : 
  total_litter = 18 → glass_bottles = 10 → aluminum_cans = total_litter - glass_bottles → 
  aluminum_cans = 8 := by sorry

end NUMINAMATH_CALUDE_litter_patrol_cans_l3852_385296


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l3852_385292

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 100) :
  x^2 + y^2 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l3852_385292


namespace NUMINAMATH_CALUDE_least_four_digit_special_number_l3852_385246

/-- A function that checks if a number has all different digits -/
def has_different_digits (n : ℕ) : Prop := sorry

/-- A function that checks if a number is divisible by all of its digits -/
def divisible_by_digits (n : ℕ) : Prop := sorry

theorem least_four_digit_special_number :
  ∀ n : ℕ,
  1000 ≤ n →
  n < 10000 →
  has_different_digits n →
  divisible_by_digits n →
  n % 5 = 0 →
  1425 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_four_digit_special_number_l3852_385246


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3852_385220

theorem log_equality_implies_ratio_one (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 18 ∧ 
       Real.log q / Real.log 18 = Real.log (p + q) / Real.log 32) : 
  p / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3852_385220


namespace NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l3852_385209

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeros in 300! is 74 -/
theorem factorial_300_trailing_zeros :
  trailingZeros 300 = 74 := by sorry

end NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l3852_385209


namespace NUMINAMATH_CALUDE_watch_time_calculation_l3852_385232

-- Define constants based on the problem conditions
def regular_season_episodes : ℕ := 22
def third_season_episodes : ℕ := 24
def last_season_extra_episodes : ℕ := 4
def previous_seasons : ℕ := 9
def early_episode_length : ℚ := 1/2
def later_episode_length : ℚ := 3/4
def bonus_episodes : ℕ := 5
def bonus_episode_length : ℚ := 1
def crossover_episode_length : ℚ := 3/2
def marathon_length : ℚ := 5
def daily_watch_time : ℚ := 2

-- Theorem to prove
theorem watch_time_calculation :
  let total_episodes := 
    3 * regular_season_episodes + 2 + -- First three seasons
    6 * regular_season_episodes + -- Seasons 4-9
    (regular_season_episodes + last_season_extra_episodes) -- Last season
  let total_hours := 
    (3 * regular_season_episodes + 2) * early_episode_length + -- First three seasons
    (6 * regular_season_episodes) * later_episode_length + -- Seasons 4-9
    (regular_season_episodes + last_season_extra_episodes) * later_episode_length + -- Last season
    bonus_episodes * bonus_episode_length + -- Bonus episodes
    crossover_episode_length -- Crossover episode
  let remaining_hours := total_hours - marathon_length
  let days_to_finish := remaining_hours / daily_watch_time
  days_to_finish = 77 := by sorry

end NUMINAMATH_CALUDE_watch_time_calculation_l3852_385232


namespace NUMINAMATH_CALUDE_max_ratio_of_two_digit_integers_l3852_385203

theorem max_ratio_of_two_digit_integers (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →  -- x is a two-digit positive integer
  (10 ≤ y ∧ y ≤ 99) →  -- y is a two-digit positive integer
  x > y →              -- x is greater than y
  (x + y) / 2 = 70 →   -- their mean is 70
  (∀ a b : ℕ, (10 ≤ a ∧ a ≤ 99) → (10 ≤ b ∧ b ≤ 99) → a > b → (a + b) / 2 = 70 → x / y ≥ a / b) →
  x / y = 99 / 41 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_of_two_digit_integers_l3852_385203


namespace NUMINAMATH_CALUDE_right_trapezoid_with_inscribed_circle_sides_l3852_385273

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  R : ℝ
  shorter_base : ℝ
  longer_base : ℝ
  longer_leg : ℝ
  shorter_base_eq : shorter_base = 4/3 * R

/-- Theorem: In a right trapezoid with an inscribed circle of radius R and shorter base 4/3 R, 
    the longer base is 4R and the longer leg is 10/3 R -/
theorem right_trapezoid_with_inscribed_circle_sides 
  (t : RightTrapezoidWithInscribedCircle) : 
  t.longer_base = 4 * t.R ∧ t.longer_leg = 10/3 * t.R := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_with_inscribed_circle_sides_l3852_385273


namespace NUMINAMATH_CALUDE_triangle_property_l3852_385283

theorem triangle_property (a b c A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b * Real.cos A + Real.sqrt 3 * b * Real.sin A - c - a = 0 →
  b = Real.sqrt 3 →
  B = π / 3 ∧ ∀ (a' c' : ℝ), a' + c' ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3852_385283


namespace NUMINAMATH_CALUDE_modified_rectangle_remaining_length_l3852_385226

/-- The total length of remaining segments after modifying a rectangle --/
def remaining_length (height width top_right_removed middle_left_removed bottom_removed top_left_removed : ℕ) : ℕ :=
  (height - middle_left_removed) + 
  (width - bottom_removed) + 
  (width - top_right_removed) + 
  (height - top_left_removed)

/-- Theorem stating the total length of remaining segments in the modified rectangle --/
theorem modified_rectangle_remaining_length :
  remaining_length 10 7 2 2 3 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_modified_rectangle_remaining_length_l3852_385226


namespace NUMINAMATH_CALUDE_slope_of_line_l_l3852_385288

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

-- Define the center of the circle
def center : ℝ × ℝ := (3, 5)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 5 = k * (x - 3)

-- Define points A, B, and P
variables (A B P : ℝ × ℝ)

-- State that A and B are on the circle C
axiom A_on_circle : circle_C A.1 A.2
axiom B_on_circle : circle_C B.1 B.2

-- State that A, B, and P are on line l
axiom A_on_line : ∃ k, line_l k A.1 A.2
axiom B_on_line : ∃ k, line_l k B.1 B.2
axiom P_on_line : ∃ k, line_l k P.1 P.2

-- State that P is on the y-axis
axiom P_on_y_axis : P.1 = 0

-- State the vector relationship
axiom vector_relation : 2 * (A.1 - P.1, A.2 - P.2) = (B.1 - P.1, B.2 - P.2)

-- Theorem to prove
theorem slope_of_line_l : ∃ k, (k = 2 ∨ k = -2) ∧ line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧ line_l k P.1 P.2 :=
sorry

end NUMINAMATH_CALUDE_slope_of_line_l_l3852_385288


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_plus_sqrt_ten_bounds_l3852_385211

theorem sqrt_two_times_sqrt_eight_plus_sqrt_ten_bounds :
  8 < Real.sqrt 2 * (Real.sqrt 8 + Real.sqrt 10) ∧
  Real.sqrt 2 * (Real.sqrt 8 + Real.sqrt 10) < 9 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_plus_sqrt_ten_bounds_l3852_385211


namespace NUMINAMATH_CALUDE_granger_son_age_l3852_385224

/-- Mr. Granger's age in relation to his son's age -/
def granger_age (son_age : ℕ) : ℕ := 2 * son_age + 10

/-- Mr. Granger's age last year in relation to his son's age last year -/
def granger_age_last_year (son_age : ℕ) : ℕ := 3 * (son_age - 1) - 4

/-- Theorem stating that Mr. Granger's son is 16 years old -/
theorem granger_son_age : 
  ∃ (son_age : ℕ), son_age > 0 ∧ 
  granger_age son_age - 1 = granger_age_last_year son_age ∧ 
  son_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_granger_son_age_l3852_385224


namespace NUMINAMATH_CALUDE_little_krish_sweet_expense_l3852_385264

theorem little_krish_sweet_expense (initial_amount : ℚ) (friend_gift : ℚ) (amount_left : ℚ) :
  initial_amount = 200.50 →
  friend_gift = 25.20 →
  amount_left = 114.85 →
  initial_amount - 2 * friend_gift - amount_left = 35.25 := by
  sorry

end NUMINAMATH_CALUDE_little_krish_sweet_expense_l3852_385264


namespace NUMINAMATH_CALUDE_coTerminalAngle_equiv_neg525_l3852_385268

/-- The angle that shares the same terminal side as -525° -/
def coTerminalAngle (k : ℤ) : ℝ := 195 + k * 360

/-- Proves that coTerminalAngle shares the same terminal side as -525° -/
theorem coTerminalAngle_equiv_neg525 :
  ∀ k : ℤ, ∃ n : ℤ, coTerminalAngle k = -525 + n * 360 := by sorry

end NUMINAMATH_CALUDE_coTerminalAngle_equiv_neg525_l3852_385268


namespace NUMINAMATH_CALUDE_simplify_expression_l3852_385249

theorem simplify_expression (a b : ℝ) : (2 * a^2)^3 * (-a * b) = -8 * a^7 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3852_385249


namespace NUMINAMATH_CALUDE_piggy_bank_dime_difference_l3852_385248

theorem piggy_bank_dime_difference :
  ∀ (nickels dimes half_dollars : ℕ),
    nickels + dimes + half_dollars = 100 →
    5 * nickels + 10 * dimes + 50 * half_dollars = 1350 →
    ∃ (max_dimes min_dimes : ℕ),
      (∀ d : ℕ, (∃ n h : ℕ, n + d + h = 100 ∧ 5 * n + 10 * d + 50 * h = 1350) → d ≤ max_dimes) ∧
      (∀ d : ℕ, (∃ n h : ℕ, n + d + h = 100 ∧ 5 * n + 10 * d + 50 * h = 1350) → d ≥ min_dimes) ∧
      max_dimes - min_dimes = 162 :=
by sorry

end NUMINAMATH_CALUDE_piggy_bank_dime_difference_l3852_385248


namespace NUMINAMATH_CALUDE_sachins_age_l3852_385244

/-- Proves that Sachin's age is 38.5 years given the conditions -/
theorem sachins_age (sachin rahul : ℝ) 
  (h1 : sachin = rahul + 7)
  (h2 : sachin / rahul = 11 / 9) : 
  sachin = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_sachins_age_l3852_385244


namespace NUMINAMATH_CALUDE_job_completion_time_l3852_385257

theorem job_completion_time (x : ℝ) (h1 : x > 0) (h2 : 9/x + 4/10 = 1) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3852_385257


namespace NUMINAMATH_CALUDE_women_in_luxury_class_l3852_385247

theorem women_in_luxury_class 
  (total_passengers : ℕ) 
  (women_percentage : ℚ) 
  (luxury_class_percentage : ℚ) 
  (h1 : total_passengers = 300) 
  (h2 : women_percentage = 80 / 100) 
  (h3 : luxury_class_percentage = 15 / 100) : 
  ℕ := by
  sorry

#check women_in_luxury_class

end NUMINAMATH_CALUDE_women_in_luxury_class_l3852_385247


namespace NUMINAMATH_CALUDE_second_year_decrease_is_twenty_percent_l3852_385206

/-- Represents the population change over two years --/
structure PopulationChange where
  initial_population : ℕ
  first_year_increase : ℚ
  final_population : ℕ

/-- Calculates the percentage decrease in the second year --/
def second_year_decrease (pc : PopulationChange) : ℚ :=
  let first_year_population := pc.initial_population * (1 + pc.first_year_increase)
  1 - (pc.final_population : ℚ) / first_year_population

/-- Theorem: Given the specified population change, the second year decrease is 20% --/
theorem second_year_decrease_is_twenty_percent 
  (pc : PopulationChange) 
  (h1 : pc.initial_population = 10000)
  (h2 : pc.first_year_increase = 1/5)
  (h3 : pc.final_population = 9600) : 
  second_year_decrease pc = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_second_year_decrease_is_twenty_percent_l3852_385206


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3852_385271

theorem trigonometric_identity : 
  Real.sin (30 * π / 180) + Real.cos (120 * π / 180) + 2 * Real.cos (45 * π / 180) - Real.sqrt 3 * Real.tan (30 * π / 180) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3852_385271


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3852_385286

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3852_385286


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3852_385279

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 3 = 61 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3852_385279


namespace NUMINAMATH_CALUDE_score_for_91_correct_out_of_100_l3852_385227

/-- Calculates the score for a test based on the number of correct responses and total questions. -/
def calculateScore (totalQuestions : ℕ) (correctResponses : ℕ) : ℤ :=
  correctResponses - 2 * (totalQuestions - correctResponses)

/-- Proves that for a 100-question test with 91 correct responses, the calculated score is 73. -/
theorem score_for_91_correct_out_of_100 :
  calculateScore 100 91 = 73 := by
  sorry

end NUMINAMATH_CALUDE_score_for_91_correct_out_of_100_l3852_385227


namespace NUMINAMATH_CALUDE_chord_length_l3852_385217

/-- Theorem: The length of a chord in a circle is √(2ar), where r is the radius of the circle
    and a is the distance from one end of the chord to the tangent drawn through its other end. -/
theorem chord_length (r a : ℝ) (hr : r > 0) (ha : a > 0) :
  ∃ (chord_length : ℝ), chord_length = Real.sqrt (2 * a * r) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l3852_385217


namespace NUMINAMATH_CALUDE_proposition_3_proposition_4_l3852_385250

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)
variable (linePerpendicularToPlane : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel
local infix:50 " ⊥ " => perpendicular
local infix:50 " ∥ᵖ " => planeParallel
local infix:50 " ⊥ᵖ " => planePerpendicular
local infix:50 " ∥ᵖˡ " => lineParallelToPlane
local infix:50 " ⊥ᵖˡ " => linePerpendicularToPlane

-- Theorem statements
theorem proposition_3 (m n : Line) (α β : Plane) :
  m ⊥ᵖˡ α → n ∥ᵖˡ β → α ∥ᵖ β → m ⊥ n := by sorry

theorem proposition_4 (m n : Line) (α β : Plane) :
  m ⊥ᵖˡ α → n ⊥ᵖˡ β → α ⊥ᵖ β → m ⊥ n := by sorry

end NUMINAMATH_CALUDE_proposition_3_proposition_4_l3852_385250


namespace NUMINAMATH_CALUDE_allen_book_pages_l3852_385237

/-- Calculates the total number of pages in a book based on daily reading rate and days to finish -/
def total_pages (pages_per_day : ℕ) (days_to_finish : ℕ) : ℕ :=
  pages_per_day * days_to_finish

/-- Proves that Allen's book has 120 pages given his reading rate and time to finish -/
theorem allen_book_pages :
  let pages_per_day : ℕ := 10
  let days_to_finish : ℕ := 12
  total_pages pages_per_day days_to_finish = 120 := by
  sorry

end NUMINAMATH_CALUDE_allen_book_pages_l3852_385237


namespace NUMINAMATH_CALUDE_fundraiser_total_amount_l3852_385219

/-- The total amount promised in a fundraiser -/
theorem fundraiser_total_amount (received : ℕ) (sally_owed : ℕ) (carl_owed : ℕ) (amy_owed : ℕ) :
  received = 285 →
  sally_owed = 35 →
  carl_owed = 35 →
  amy_owed = 30 →
  received + sally_owed + carl_owed + amy_owed + amy_owed / 2 = 400 := by
  sorry

#check fundraiser_total_amount

end NUMINAMATH_CALUDE_fundraiser_total_amount_l3852_385219


namespace NUMINAMATH_CALUDE_segment_length_product_l3852_385260

theorem segment_length_product (a : ℝ) : 
  (∃ a₁ a₂ : ℝ, 
    (∀ a : ℝ, (3 * a - 5)^2 + (2 * a - 5)^2 = 125 ↔ (a = a₁ ∨ a = a₂)) ∧
    a₁ * a₂ = -749/676) :=
by sorry

end NUMINAMATH_CALUDE_segment_length_product_l3852_385260


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a15_l3852_385228

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_a15 (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a 7 = 8 → a 8 = 7 → a 15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a15_l3852_385228


namespace NUMINAMATH_CALUDE_bianca_cupcake_sale_l3852_385223

/-- Bianca's cupcake sale problem --/
theorem bianca_cupcake_sale (initial : ℕ) (made_later : ℕ) (left_at_end : ℕ) :
  initial + made_later - left_at_end = (initial + made_later) - left_at_end :=
by
  sorry

/-- Solving Bianca's cupcake sale problem --/
def solve_bianca_cupcake_sale (initial : ℕ) (made_later : ℕ) (left_at_end : ℕ) : ℕ :=
  initial + made_later - left_at_end

#eval solve_bianca_cupcake_sale 14 17 25

end NUMINAMATH_CALUDE_bianca_cupcake_sale_l3852_385223


namespace NUMINAMATH_CALUDE_theme_park_triplets_l3852_385259

theorem theme_park_triplets (total_cost mother_charge child_charge_per_year : ℚ)
  (h_total_cost : total_cost = 15.25)
  (h_mother_charge : mother_charge = 6.95)
  (h_child_charge : child_charge_per_year = 0.55)
  : ∃ (triplet_age youngest_age : ℕ),
    triplet_age > youngest_age ∧
    youngest_age = 3 ∧
    total_cost = mother_charge + child_charge_per_year * (3 * triplet_age + youngest_age) :=
by sorry

end NUMINAMATH_CALUDE_theme_park_triplets_l3852_385259


namespace NUMINAMATH_CALUDE_parents_present_l3852_385213

theorem parents_present (total_people pupils teachers : ℕ) 
  (h1 : total_people = 1541)
  (h2 : pupils = 724)
  (h3 : teachers = 744) :
  total_people - (pupils + teachers) = 73 := by
  sorry

end NUMINAMATH_CALUDE_parents_present_l3852_385213
