import Mathlib

namespace NUMINAMATH_CALUDE_unique_intersection_l1993_199317

/-- The value of b for which the graphs of y = bx^2 + 5x + 3 and y = -2x - 3 intersect at exactly one point -/
def b : ℚ := 49 / 24

/-- The first function: f(x) = bx^2 + 5x + 3 -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 3

/-- The second function: g(x) = -2x - 3 -/
def g (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem stating that the graphs of f and g intersect at exactly one point -/
theorem unique_intersection : ∃! x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l1993_199317


namespace NUMINAMATH_CALUDE_solve_equation_l1993_199332

theorem solve_equation (x : ℝ) : (1 / 2) * (1 / 7) * x = 14 → x = 196 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1993_199332


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l1993_199331

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l1993_199331


namespace NUMINAMATH_CALUDE_golden_ratio_roots_l1993_199308

theorem golden_ratio_roots (r : ℝ) : r^2 = r + 1 → r^6 = 8*r + 5 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_roots_l1993_199308


namespace NUMINAMATH_CALUDE_equidistant_points_count_l1993_199307

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its normal vector and distance from origin --/
structure Line where
  normal : ℝ × ℝ
  distance : ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- Distance between a point and a line --/
def distancePointToLine (p : Point) (l : Line) : ℝ := sorry

/-- Distance between a point and a circle --/
def distancePointToCircle (p : Point) (c : Circle) : ℝ := sorry

/-- Check if a line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- The main theorem --/
theorem equidistant_points_count
  (c : Circle)
  (t1 t2 : Line)
  (h1 : c.radius = 4)
  (h2 : isTangent t1 c)
  (h3 : isTangent t2 c)
  (h4 : t1.distance = 4)
  (h5 : t2.distance = 6)
  (h6 : t1.normal = t2.normal) :
  ∃! (s : Finset Point), 
    (∀ p ∈ s, distancePointToCircle p c = distancePointToLine p t1 ∧ 
                distancePointToCircle p c = distancePointToLine p t2) ∧
    s.card = 2 := by sorry

end NUMINAMATH_CALUDE_equidistant_points_count_l1993_199307


namespace NUMINAMATH_CALUDE_cara_age_l1993_199392

/-- Given the ages of three generations in a family, prove Cara's age. -/
theorem cara_age (cara_age mom_age grandma_age : ℕ) 
  (h1 : cara_age + 20 = mom_age)
  (h2 : mom_age + 15 = grandma_age)
  (h3 : grandma_age = 75) :
  cara_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_cara_age_l1993_199392


namespace NUMINAMATH_CALUDE_average_increase_is_three_l1993_199327

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  averageRuns : Rat

/-- Calculates the increase in average runs after a new inning -/
def averageIncrease (prev : BatsmanPerformance) (newRuns : Nat) (newAverage : Rat) : Rat :=
  newAverage - prev.averageRuns

/-- Theorem: The increase in the batsman's average is 3 runs -/
theorem average_increase_is_three 
  (prev : BatsmanPerformance) 
  (h1 : prev.innings = 16) 
  (h2 : (prev.totalRuns + 87 : Rat) / 17 = 39) : 
  averageIncrease prev 87 39 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_increase_is_three_l1993_199327


namespace NUMINAMATH_CALUDE_kiran_money_l1993_199356

/-- Given the ratios of money between Ravi, Giri, and Kiran, and Ravi's amount of money,
    prove that Kiran has $105. -/
theorem kiran_money (ravi giri kiran : ℚ) : 
  (ravi / giri = 6 / 7) →
  (giri / kiran = 6 / 15) →
  (ravi = 36) →
  (kiran = 105) := by
sorry

end NUMINAMATH_CALUDE_kiran_money_l1993_199356


namespace NUMINAMATH_CALUDE_converse_is_false_l1993_199314

theorem converse_is_false : ¬∀ x : ℝ, x > 0 → x - 3 > 0 := by sorry

end NUMINAMATH_CALUDE_converse_is_false_l1993_199314


namespace NUMINAMATH_CALUDE_arithmetic_polynomial_root_count_l1993_199369

/-- Represents a polynomial of degree 5 with integer coefficients forming an arithmetic sequence. -/
structure ArithmeticPolynomial where
  a : ℤ
  d : ℤ  -- Common difference of the arithmetic sequence

/-- The number of integer roots (counting multiplicity) of an ArithmeticPolynomial. -/
def integerRootCount (p : ArithmeticPolynomial) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of integer roots. -/
theorem arithmetic_polynomial_root_count (p : ArithmeticPolynomial) :
  integerRootCount p ∈ ({0, 1, 2, 3, 5} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_polynomial_root_count_l1993_199369


namespace NUMINAMATH_CALUDE_distance_calculation_l1993_199346

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 74

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time difference between Maxwell's start and Brad's start in hours -/
def time_difference : ℝ := 1

/-- Total time until Maxwell and Brad meet in hours -/
def total_time : ℝ := 8

theorem distance_calculation :
  distance_between_homes = 
    maxwell_speed * total_time + 
    brad_speed * (total_time - time_difference) :=
by sorry

end NUMINAMATH_CALUDE_distance_calculation_l1993_199346


namespace NUMINAMATH_CALUDE_ping_pong_table_distribution_l1993_199324

theorem ping_pong_table_distribution (total_tables : Nat) (total_players : Nat)
  (h_tables : total_tables = 15)
  (h_players : total_players = 38) :
  ∃ (singles_tables doubles_tables : Nat),
    singles_tables + doubles_tables = total_tables ∧
    2 * singles_tables + 4 * doubles_tables = total_players ∧
    singles_tables = 11 ∧
    doubles_tables = 4 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_table_distribution_l1993_199324


namespace NUMINAMATH_CALUDE_average_speed_first_part_is_35_l1993_199395

-- Define the total trip duration in hours
def total_trip_duration : ℝ := 24

-- Define the average speed for the entire trip in miles per hour
def average_speed_entire_trip : ℝ := 50

-- Define the duration of the first part of the trip in hours
def first_part_duration : ℝ := 4

-- Define the speed for the remaining part of the trip in miles per hour
def remaining_part_speed : ℝ := 53

-- Define the average speed for the first part of the trip
def average_speed_first_part : ℝ := 35

-- Theorem statement
theorem average_speed_first_part_is_35 :
  total_trip_duration * average_speed_entire_trip =
  first_part_duration * average_speed_first_part +
  (total_trip_duration - first_part_duration) * remaining_part_speed :=
by sorry

end NUMINAMATH_CALUDE_average_speed_first_part_is_35_l1993_199395


namespace NUMINAMATH_CALUDE_canoe_production_sum_l1993_199335

theorem canoe_production_sum : 
  let a : ℕ := 8  -- first term
  let r : ℕ := 3  -- common ratio
  let n : ℕ := 8  -- number of terms
  let sum := a * (r^n - 1) / (r - 1)
  sum = 26240 := by sorry

end NUMINAMATH_CALUDE_canoe_production_sum_l1993_199335


namespace NUMINAMATH_CALUDE_min_value_theorem_l1993_199322

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - 3 = 0) :
  2 / x + 1 / y ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ - 3 = 0 ∧ 2 / x₀ + 1 / y₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1993_199322


namespace NUMINAMATH_CALUDE_field_B_most_stable_l1993_199393

-- Define the variances for each field
def variance_A : ℝ := 3.6
def variance_B : ℝ := 2.89
def variance_C : ℝ := 13.4
def variance_D : ℝ := 20.14

-- Define a function to compare two variances
def is_more_stable (v1 v2 : ℝ) : Prop := v1 < v2

-- Theorem stating that Field B has the lowest variance
theorem field_B_most_stable :
  is_more_stable variance_B variance_A ∧
  is_more_stable variance_B variance_C ∧
  is_more_stable variance_B variance_D :=
by sorry

end NUMINAMATH_CALUDE_field_B_most_stable_l1993_199393


namespace NUMINAMATH_CALUDE_scalar_mult_assoc_l1993_199363

variable (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem scalar_mult_assoc (a : V) (h : a ≠ 0) :
  (-4 : ℝ) • (3 • a) = (-12 : ℝ) • a := by sorry

end NUMINAMATH_CALUDE_scalar_mult_assoc_l1993_199363


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1993_199362

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 8*x + c > 0) ↔ (0 < c ∧ c < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1993_199362


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1993_199396

def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B : Set ℝ := {x | x > 1}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Ici (2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1993_199396


namespace NUMINAMATH_CALUDE_exponent_division_l1993_199353

theorem exponent_division (a : ℝ) : a^7 / a^4 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1993_199353


namespace NUMINAMATH_CALUDE_percentage_increase_in_earnings_l1993_199376

theorem percentage_increase_in_earnings (initial_earnings new_earnings : ℝ) 
  (h1 : initial_earnings = 60)
  (h2 : new_earnings = 84) :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_in_earnings_l1993_199376


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1993_199313

/-- Given a point A(2,4) and a circle x^2 + y^2 = 4, 
    the tangent line from A to the circle has equation x = 2 or 3x - 4y + 10 = 0 -/
theorem tangent_line_to_circle (A : ℝ × ℝ) (circle : Set (ℝ × ℝ)) :
  A = (2, 4) →
  circle = {(x, y) | x^2 + y^2 = 4} →
  (∃ (k : ℝ), (∀ (x y : ℝ), (x, y) ∈ circle → 
    (x = 2 ∨ 3*x - 4*y + 10 = 0) ↔ 
    ((x - 2)^2 + (y - 4)^2 = ((x - 0)^2 + (y - 0)^2 - 4) / 4))) := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_to_circle_l1993_199313


namespace NUMINAMATH_CALUDE_f_eight_eq_twelve_f_two_f_odd_l1993_199361

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that f is not identically zero
axiom f_not_zero : ∃ x, f x ≠ 0

-- Define the functional equation
axiom f_eq (x y : ℝ) : f (x * y) = x * f y + y * f x

-- Theorem 1: f(8) = 12f(2)
theorem f_eight_eq_twelve_f_two : f 8 = 12 * f 2 := by sorry

-- Theorem 2: f is an odd function
theorem f_odd : ∀ x, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_f_eight_eq_twelve_f_two_f_odd_l1993_199361


namespace NUMINAMATH_CALUDE_square_difference_theorem_l1993_199336

theorem square_difference_theorem (a b A : ℝ) : 
  (5*a + 3*b)^2 = (5*a - 3*b)^2 + A → A = 60*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l1993_199336


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1993_199357

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) →
  a - b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1993_199357


namespace NUMINAMATH_CALUDE_probability_of_six_red_balls_l1993_199344

def total_balls : ℕ := 100
def red_balls : ℕ := 80
def white_balls : ℕ := 20
def drawn_balls : ℕ := 10
def red_drawn : ℕ := 6

theorem probability_of_six_red_balls :
  (Nat.choose red_balls red_drawn * Nat.choose white_balls (drawn_balls - red_drawn)) / 
  Nat.choose total_balls drawn_balls = 
  (Nat.choose red_balls red_drawn * Nat.choose white_balls (drawn_balls - red_drawn)) / 
  Nat.choose total_balls drawn_balls := by sorry

end NUMINAMATH_CALUDE_probability_of_six_red_balls_l1993_199344


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l1993_199325

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l1993_199325


namespace NUMINAMATH_CALUDE_intersection_point_is_7_neg8_l1993_199364

/-- Two lines in 2D space --/
structure TwoLines where
  line1 : ℝ → ℝ × ℝ
  line2 : ℝ → ℝ × ℝ

/-- The given two lines from the problem --/
def givenLines : TwoLines where
  line1 := λ t => (1 + 2*t, 1 - 3*t)
  line2 := λ u => (5 + 4*u, -9 + 2*u)

/-- Definition of intersection point --/
def isIntersectionPoint (p : ℝ × ℝ) (lines : TwoLines) : Prop :=
  ∃ t u, lines.line1 t = p ∧ lines.line2 u = p

/-- Theorem stating that (7, -8) is the intersection point of the given lines --/
theorem intersection_point_is_7_neg8 :
  isIntersectionPoint (7, -8) givenLines := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_7_neg8_l1993_199364


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l1993_199374

theorem subtraction_preserves_inequality (a b : ℝ) : a < b → a - 1 < b - 1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l1993_199374


namespace NUMINAMATH_CALUDE_lauren_reaches_andrea_in_30_minutes_l1993_199330

/-- Represents the scenario of Andrea and Lauren biking towards each other --/
structure BikingScenario where
  initial_distance : ℝ
  andrea_speed : ℝ
  lauren_speed : ℝ
  decrease_rate : ℝ
  flat_tire_time : ℝ
  lauren_delay : ℝ

/-- Calculates the total time for Lauren to reach Andrea --/
def totalTime (scenario : BikingScenario) : ℝ :=
  sorry

/-- The theorem stating that Lauren reaches Andrea after 30 minutes --/
theorem lauren_reaches_andrea_in_30_minutes (scenario : BikingScenario)
  (h1 : scenario.initial_distance = 30)
  (h2 : scenario.andrea_speed = 2 * scenario.lauren_speed)
  (h3 : scenario.decrease_rate = 2)
  (h4 : scenario.flat_tire_time = 10)
  (h5 : scenario.lauren_delay = 5) :
  totalTime scenario = 30 :=
sorry

end NUMINAMATH_CALUDE_lauren_reaches_andrea_in_30_minutes_l1993_199330


namespace NUMINAMATH_CALUDE_frog_hop_probability_l1993_199350

/-- Represents a position on a 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents the possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines whether a position is on the edge of the grid -/
def isEdgePosition (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Defines the next position after a hop in a given direction -/
def nextPosition (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down => ⟨p.x, (p.y + 3) % 4⟩
  | Direction.Left => ⟨(p.x + 3) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- The probability of reaching an edge position within n hops -/
def probReachEdge (n : Nat) (start : Position) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem frog_hop_probability :
  probReachEdge 5 ⟨2, 2⟩ = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_frog_hop_probability_l1993_199350


namespace NUMINAMATH_CALUDE_switch_connections_l1993_199342

theorem switch_connections (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_switch_connections_l1993_199342


namespace NUMINAMATH_CALUDE_vector_angle_equality_l1993_199338

/-- Given vectors a and b in ℝ², find the value of m such that the angle between
    c = m * a + b and a equals the angle between c and b. -/
theorem vector_angle_equality (a b : ℝ × ℝ) (m : ℝ) : 
  a = (1, 2) →
  b = (4, 2) →
  let c := (m * a.1 + b.1, m * a.2 + b.2)
  (c.1 * a.1 + c.2 * a.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (c.1^2 + c.2^2)) =
  (c.1 * b.1 + c.2 * b.2) / (Real.sqrt (b.1^2 + b.2^2) * Real.sqrt (c.1^2 + c.2^2)) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_equality_l1993_199338


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1993_199329

theorem sum_of_cubes (x y : ℝ) (h1 : x * y = 15) (h2 : x + y = 11) :
  x^3 + y^3 = 836 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1993_199329


namespace NUMINAMATH_CALUDE_pauls_crayons_l1993_199337

theorem pauls_crayons (erasers_birthday : ℕ) (crayons_left : ℕ) (crayons_difference : ℕ) :
  erasers_birthday = 38 →
  crayons_left = 391 →
  crayons_difference = 353 →
  crayons_left = erasers_birthday + crayons_difference →
  crayons_left = 391 :=
by sorry

end NUMINAMATH_CALUDE_pauls_crayons_l1993_199337


namespace NUMINAMATH_CALUDE_system_solution_l1993_199318

theorem system_solution : ∃ (x y : ℝ), 
  (2 * x^2 - 3 * x * y + y^2 = 3 ∧ x^2 + 2 * x * y - 2 * y^2 = 6) ∧
  ((x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1993_199318


namespace NUMINAMATH_CALUDE_sin_pi_minus_alpha_l1993_199348

theorem sin_pi_minus_alpha (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.sin α) : 
  Real.sin (π - α) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_minus_alpha_l1993_199348


namespace NUMINAMATH_CALUDE_exponent_simplification_l1993_199303

theorem exponent_simplification (a : ℝ) : (36 * a^9)^4 * (63 * a^9)^4 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l1993_199303


namespace NUMINAMATH_CALUDE_joanna_estimate_l1993_199320

theorem joanna_estimate (u v ε₁ ε₂ : ℝ) (h1 : u > v) (h2 : v > 0) (h3 : ε₁ > 0) (h4 : ε₂ > 0) :
  (u + ε₁) - (v - ε₂) > u - v := by
  sorry

end NUMINAMATH_CALUDE_joanna_estimate_l1993_199320


namespace NUMINAMATH_CALUDE_triangle_inequality_l1993_199368

theorem triangle_inequality (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  Real.sqrt x + Real.sqrt y > Real.sqrt z →
  Real.sqrt y + Real.sqrt z > Real.sqrt x →
  Real.sqrt z + Real.sqrt x > Real.sqrt y →
  x / y + y / z + z / x = 5 →
  x * (y^2 - 2*z^2) / z + y * (z^2 - 2*x^2) / x + z * (x^2 - 2*y^2) / y ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1993_199368


namespace NUMINAMATH_CALUDE_lucille_remaining_cents_l1993_199341

-- Define the problem parameters
def cents_per_weed : ℕ := 6
def weeds_flower_bed : ℕ := 11
def weeds_vegetable_patch : ℕ := 14
def weeds_grass : ℕ := 32
def soda_cost : ℕ := 99

-- Calculate the total weeds pulled
def total_weeds_pulled : ℕ := weeds_flower_bed + weeds_vegetable_patch + weeds_grass / 2

-- Calculate the earnings
def earnings : ℕ := total_weeds_pulled * cents_per_weed

-- Calculate the remaining cents
def remaining_cents : ℕ := earnings - soda_cost

-- Theorem to prove
theorem lucille_remaining_cents : remaining_cents = 147 := by
  sorry

end NUMINAMATH_CALUDE_lucille_remaining_cents_l1993_199341


namespace NUMINAMATH_CALUDE_square_not_always_positive_l1993_199367

theorem square_not_always_positive : ¬ ∀ x : ℝ, x^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l1993_199367


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1993_199355

/-- A quadratic function with graph opening upwards and vertex at (1, -2) -/
def quadratic_function (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem quadratic_function_properties :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x : ℝ, quadratic_function x = a * (x - 1)^2 - 2) ∧
  (∀ x : ℝ, quadratic_function x ≥ -2) ∧
  quadratic_function 1 = -2 := by
sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1993_199355


namespace NUMINAMATH_CALUDE_integral_sin_over_one_minus_cos_squared_l1993_199399

theorem integral_sin_over_one_minus_cos_squared (f : ℝ → ℝ) :
  (∫ x in Set.Icc (π / 2) π, (2 * Real.sin x) / ((1 - Real.cos x)^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin_over_one_minus_cos_squared_l1993_199399


namespace NUMINAMATH_CALUDE_orthic_triangle_inradius_bound_l1993_199381

/-- Given a triangle ABC with circumradius R = 1 and inradius r, 
    the inradius P of its orthic triangle A'B'C' satisfies P ≤ 1 - (1/3)(1+r)^2 -/
theorem orthic_triangle_inradius_bound (R r P : ℝ) : 
  R = 1 → 0 < r → r ≤ 1/2 → P ≤ 1 - (1/3) * (1 + r)^2 := by
  sorry

end NUMINAMATH_CALUDE_orthic_triangle_inradius_bound_l1993_199381


namespace NUMINAMATH_CALUDE_fourteenSidedFigureArea_l1993_199378

/-- A point in 2D space represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the fourteen-sided figure -/
def vertices : List Point := [
  ⟨1, 2⟩, ⟨1, 3⟩, ⟨2, 4⟩, ⟨3, 5⟩, ⟨4, 6⟩, ⟨5, 5⟩, ⟨6, 5⟩,
  ⟨7, 4⟩, ⟨7, 3⟩, ⟨6, 2⟩, ⟨5, 1⟩, ⟨4, 1⟩, ⟨3, 1⟩, ⟨2, 2⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ :=
  sorry -- Implement the calculation of polygon area

/-- Theorem stating that the area of the fourteen-sided figure is 14 square centimeters -/
theorem fourteenSidedFigureArea : polygonArea vertices = 14 := by
  sorry

end NUMINAMATH_CALUDE_fourteenSidedFigureArea_l1993_199378


namespace NUMINAMATH_CALUDE_expression_equality_l1993_199380

theorem expression_equality (x : ℝ) (h : x ≥ 1) :
  let expr := Real.sqrt (x + 2 * Real.sqrt (x - 1)) + Real.sqrt (x - 2 * Real.sqrt (x - 1))
  (x ≤ 2 → expr = 2) ∧ (x > 2 → expr = 2 * Real.sqrt (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1993_199380


namespace NUMINAMATH_CALUDE_find_other_number_l1993_199343

theorem find_other_number (a b : ℤ) (h1 : 3 * a + 4 * b = 161) (h2 : a = 17 ∨ b = 17) : 
  (a = 31 ∨ b = 31) :=
sorry

end NUMINAMATH_CALUDE_find_other_number_l1993_199343


namespace NUMINAMATH_CALUDE_existence_of_subsets_l1993_199302

/-- The set M containing integers from 1 to 10000 -/
def M : Set ℕ := Finset.range 10000

/-- The property that defines the required subsets -/
def has_unique_intersection (A : Finset (Set ℕ)) : Prop :=
  ∀ z ∈ M, ∃ B : Finset (Set ℕ), B ⊆ A ∧ B.card = 8 ∧ (⋂₀ B.toSet : Set ℕ) = {z}

/-- The main theorem stating the existence of 16 subsets with the required property -/
theorem existence_of_subsets : ∃ A : Finset (Set ℕ), A.card = 16 ∧ has_unique_intersection A := by
  sorry

end NUMINAMATH_CALUDE_existence_of_subsets_l1993_199302


namespace NUMINAMATH_CALUDE_sum_odd_integers_7_to_35_l1993_199371

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

theorem sum_odd_integers_7_to_35 :
  ∃ n : ℕ,
    (∀ k ∈ Finset.range n, is_odd (7 + 2 * k)) ∧
    (7 + 2 * (n - 1) = 35) ∧
    (arithmetic_sum 7 35 n = 315) :=
sorry

end NUMINAMATH_CALUDE_sum_odd_integers_7_to_35_l1993_199371


namespace NUMINAMATH_CALUDE_real_roots_iff_k_leq_five_l1993_199311

theorem real_roots_iff_k_leq_five (k : ℝ) :
  (∃ x : ℝ, (k - 3) * x^2 - 4 * x + 2 = 0) ↔ k ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_leq_five_l1993_199311


namespace NUMINAMATH_CALUDE_alfreds_savings_l1993_199310

/-- Alfred's savings problem -/
theorem alfreds_savings (goal : ℝ) (months : ℕ) (monthly_savings : ℝ) 
  (h1 : goal = 1000)
  (h2 : months = 12)
  (h3 : monthly_savings = 75) :
  goal - (monthly_savings * months) = 100 := by
  sorry

end NUMINAMATH_CALUDE_alfreds_savings_l1993_199310


namespace NUMINAMATH_CALUDE_inverse_mod_89_l1993_199345

theorem inverse_mod_89 (h : (9⁻¹ : ZMod 89) = 79) : (81⁻¹ : ZMod 89) = 11 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_89_l1993_199345


namespace NUMINAMATH_CALUDE_brick_length_proof_l1993_199365

/-- Given a courtyard and brick specifications, prove the length of each brick -/
theorem brick_length_proof (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_width : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 18 →
  brick_width = 0.1 →
  total_bricks = 22500 →
  ∃ (brick_length : ℝ),
    brick_length = 0.2 ∧
    courtyard_length * courtyard_width * 10000 = 
      total_bricks * brick_length * brick_width := by
  sorry

end NUMINAMATH_CALUDE_brick_length_proof_l1993_199365


namespace NUMINAMATH_CALUDE_max_nmmm_value_l1993_199319

/-- Represents a three-digit number where all digits are the same -/
def three_digit_same (d : ℕ) : ℕ := 100 * d + 10 * d + d

/-- Represents a four-digit number NMMM where the last three digits are the same -/
def four_digit_nmmm (n m : ℕ) : ℕ := 1000 * n + 100 * m + 10 * m + m

/-- The maximum value of NMMM given the problem conditions -/
theorem max_nmmm_value :
  ∀ m : ℕ,
  1 ≤ m → m ≤ 9 →
  (∃ n : ℕ, four_digit_nmmm n m = m * three_digit_same m) →
  (∀ k : ℕ, k ≤ 9 → 
    (∃ l : ℕ, four_digit_nmmm l k = k * three_digit_same k) →
    four_digit_nmmm l k ≤ 3996) :=
by sorry

end NUMINAMATH_CALUDE_max_nmmm_value_l1993_199319


namespace NUMINAMATH_CALUDE_intersection_polar_coords_l1993_199316

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 2
def C₂ (t x y : ℝ) : Prop := x = 2 - t ∧ y = t

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop :=
  C₁ x y ∧ ∃ t, C₂ t x y

-- Define polar coordinates
def polar_coords (x y ρ θ : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem statement
theorem intersection_polar_coords :
  ∃ x y : ℝ, intersection_point x y ∧ 
  polar_coords x y (Real.sqrt 2) (Real.pi / 4) :=
sorry

end NUMINAMATH_CALUDE_intersection_polar_coords_l1993_199316


namespace NUMINAMATH_CALUDE_sum_of_digits_n_plus_5_l1993_199385

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_n_plus_5 (n : ℕ) (h1 : S n = 365) (h2 : n % 8 = S n % 8) :
  S (n + 5) = 370 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_n_plus_5_l1993_199385


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l1993_199379

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def count_perfect_square_factors (a b c : ℕ) : ℕ :=
  (a/2 + 1) * ((b/2 + 1) * (c/2 + 1))

theorem perfect_square_factors_count :
  count_perfect_square_factors 10 12 15 = 336 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l1993_199379


namespace NUMINAMATH_CALUDE_amp_eight_five_l1993_199305

-- Define the & operation
def amp (a b : ℤ) : ℤ := (a + b)^2 * (a - b)

-- State the theorem
theorem amp_eight_five : amp 8 5 = 507 := by
  sorry

end NUMINAMATH_CALUDE_amp_eight_five_l1993_199305


namespace NUMINAMATH_CALUDE_line_equation_proof_l1993_199328

/-- A line that passes through a point and intersects both axes -/
structure IntersectingLine where
  -- The point through which the line passes
  P : ℝ × ℝ
  -- The point where the line intersects the x-axis
  A : ℝ × ℝ
  -- The point where the line intersects the y-axis
  B : ℝ × ℝ
  -- Ensure A is on the x-axis
  hA : A.2 = 0
  -- Ensure B is on the y-axis
  hB : B.1 = 0
  -- Ensure P is the midpoint of AB
  hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- The equation of the line is 3x - 2y + 24 = 0 -/
def lineEquation (l : IntersectingLine) : Prop :=
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | 3 * p.1 - 2 * p.2 + 24 = 0} ↔ 
    ∃ t : ℝ, (x, y) = (1 - t) • l.A + t • l.B

/-- The main theorem -/
theorem line_equation_proof (l : IntersectingLine) (h : l.P = (-4, 6)) : 
  lineEquation l := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1993_199328


namespace NUMINAMATH_CALUDE_abc_product_l1993_199366

theorem abc_product (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 30) (h5 : 1 / a + 1 / b + 1 / c + 450 / (a * b * c) = 1) :
  a * b * c = 1912 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1993_199366


namespace NUMINAMATH_CALUDE_sum_of_first_and_third_l1993_199388

theorem sum_of_first_and_third (A B C : ℝ) : 
  A + B + C = 330 → 
  A = 2 * B → 
  C = A / 3 → 
  B = 90 → 
  A + C = 240 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_and_third_l1993_199388


namespace NUMINAMATH_CALUDE_second_divisor_problem_l1993_199347

theorem second_divisor_problem (x : ℚ) : 
  (((377 / 13) / x) * (1 / 4)) / 2 = 0.125 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l1993_199347


namespace NUMINAMATH_CALUDE_second_digit_max_l1993_199333

def original_number : ℚ := 0.123456789

-- Function to change a digit to 9 and swap with the next digit
def change_and_swap (n : ℚ) (pos : ℕ) : ℚ := sorry

-- Function to get the maximum value after change and swap operations
def max_after_change_and_swap (n : ℚ) : ℚ := sorry

-- Theorem stating that changing the second digit gives the maximum value
theorem second_digit_max :
  change_and_swap original_number 2 = max_after_change_and_swap original_number :=
sorry

end NUMINAMATH_CALUDE_second_digit_max_l1993_199333


namespace NUMINAMATH_CALUDE_tensor_plus_relation_l1993_199351

-- Define a structure for pairs of real numbers
structure Pair :=
  (x : ℝ)
  (y : ℝ)

-- Define equality for pairs
def pair_eq (a b : Pair) : Prop :=
  a.x = b.x ∧ a.y = b.y

-- Define the ⊗ operation
def tensor (a b : Pair) : Pair :=
  ⟨a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x⟩

-- Define the ⊕ operation
def plus (a b : Pair) : Pair :=
  ⟨a.x + b.x, a.y + b.y⟩

-- State the theorem
theorem tensor_plus_relation (p q : ℝ) :
  pair_eq (tensor ⟨1, 2⟩ ⟨p, q⟩) ⟨5, 0⟩ →
  pair_eq (plus ⟨1, 2⟩ ⟨p, q⟩) ⟨2, 0⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_tensor_plus_relation_l1993_199351


namespace NUMINAMATH_CALUDE_playground_fence_length_l1993_199315

/-- The side length of the square fence around the playground -/
def playground_side_length : ℝ := 27

/-- The length of the garden -/
def garden_length : ℝ := 12

/-- The width of the garden -/
def garden_width : ℝ := 9

/-- The total fencing for both the playground and the garden -/
def total_fencing : ℝ := 150

/-- Theorem stating that the side length of the square fence around the playground is 27 yards -/
theorem playground_fence_length :
  4 * playground_side_length + 2 * (garden_length + garden_width) = total_fencing :=
sorry

end NUMINAMATH_CALUDE_playground_fence_length_l1993_199315


namespace NUMINAMATH_CALUDE_camel_cost_l1993_199391

/-- The cost relationship between animals and the cost of a camel --/
theorem camel_cost (camel horse ox elephant : ℝ) 
  (h1 : 10 * camel = 24 * horse)
  (h2 : 16 * horse = 4 * ox)
  (h3 : 6 * ox = 4 * elephant)
  (h4 : 10 * elephant = 120000) :
  camel = 4800 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l1993_199391


namespace NUMINAMATH_CALUDE_abs_difference_given_product_and_sum_l1993_199334

theorem abs_difference_given_product_and_sum (a b : ℝ) 
  (h1 : a * b = 6) 
  (h2 : a + b = 7) : 
  |a - b| = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_difference_given_product_and_sum_l1993_199334


namespace NUMINAMATH_CALUDE_median_length_right_triangle_l1993_199354

theorem median_length_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let median := (1 / 2 : ℝ) * c
  median = 5 := by
sorry

end NUMINAMATH_CALUDE_median_length_right_triangle_l1993_199354


namespace NUMINAMATH_CALUDE_complementary_events_l1993_199323

-- Define the sample space for a die throw
def DieThrow : Type := Fin 6

-- Define event A: upward face shows an odd number
def eventA : Set DieThrow := {x | x.val % 2 = 1}

-- Define event B: upward face shows an even number
def eventB : Set DieThrow := {x | x.val % 2 = 0}

-- Theorem stating that A and B are complementary events
theorem complementary_events : 
  eventA ∪ eventB = Set.univ ∧ eventA ∩ eventB = ∅ := by
  sorry

end NUMINAMATH_CALUDE_complementary_events_l1993_199323


namespace NUMINAMATH_CALUDE_paper_tearing_theorem_l1993_199372

/-- Represents the number of pieces after n tearing operations -/
def pieces (n : ℕ) : ℕ := 1 + 4 * n

theorem paper_tearing_theorem :
  (¬ ∃ n : ℕ, pieces n = 1994) ∧ (∃ n : ℕ, pieces n = 1997) := by
  sorry

end NUMINAMATH_CALUDE_paper_tearing_theorem_l1993_199372


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1993_199352

theorem min_value_of_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1993_199352


namespace NUMINAMATH_CALUDE_work_completion_time_l1993_199326

/-- The number of days it takes for the original number of ladies to complete the work -/
def completion_time (original_ladies : ℕ) : ℝ :=
  6

/-- The time it takes for twice the number of ladies to complete half the work -/
def half_work_time (original_ladies : ℕ) : ℝ :=
  3

theorem work_completion_time (original_ladies : ℕ) :
  completion_time original_ladies = 2 * half_work_time original_ladies :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1993_199326


namespace NUMINAMATH_CALUDE_gift_distribution_ways_l1993_199390

theorem gift_distribution_ways (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (Nat.factorial n) / (Nat.factorial (n - k)) = 303600 := by
  sorry

end NUMINAMATH_CALUDE_gift_distribution_ways_l1993_199390


namespace NUMINAMATH_CALUDE_ice_cube_distribution_l1993_199359

theorem ice_cube_distribution (total_ice_cubes : ℕ) (ice_cubes_per_cup : ℕ) (h1 : total_ice_cubes = 30) (h2 : ice_cubes_per_cup = 5) :
  total_ice_cubes / ice_cubes_per_cup = 6 := by
  sorry

end NUMINAMATH_CALUDE_ice_cube_distribution_l1993_199359


namespace NUMINAMATH_CALUDE_increase_by_percentage_l1993_199349

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 250 →
  percentage = 75 →
  final = initial * (1 + percentage / 100) →
  final = 437.5 := by
sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l1993_199349


namespace NUMINAMATH_CALUDE_sequence_length_l1993_199306

theorem sequence_length (n : ℕ) (b : ℕ → ℝ) : 
  (n > 0) →
  (b 0 = 45) →
  (b 1 = 80) →
  (b n = 0) →
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) →
  n = 901 := by
sorry

end NUMINAMATH_CALUDE_sequence_length_l1993_199306


namespace NUMINAMATH_CALUDE_game_converges_to_black_hole_l1993_199300

/-- Represents a three-digit number in the game --/
structure GameNumber :=
  (hundreds : Nat)
  (tens : Nat)
  (ones : Nat)

/-- Counts the number of even digits in a natural number --/
def countEvenDigits (n : Nat) : Nat :=
  sorry

/-- Counts the number of odd digits in a natural number --/
def countOddDigits (n : Nat) : Nat :=
  sorry

/-- Counts the total number of digits in a natural number --/
def countDigits (n : Nat) : Nat :=
  sorry

/-- Converts a natural number to a GameNumber --/
def natToGameNumber (n : Nat) : GameNumber :=
  { hundreds := countEvenDigits n,
    tens := countOddDigits n,
    ones := countDigits n }

/-- Converts a GameNumber to a natural number --/
def gameNumberToNat (g : GameNumber) : Nat :=
  g.hundreds * 100 + g.tens * 10 + g.ones

/-- Applies one step of the game rules --/
def gameStep (n : Nat) : Nat :=
  gameNumberToNat (natToGameNumber n)

/-- The final number reached in the game --/
def blackHoleNumber : Nat := 123

/-- Theorem: The game always ends with the black hole number --/
theorem game_converges_to_black_hole (start : Nat) : 
  ∃ k : Nat, (gameStep^[k] start) = blackHoleNumber :=
sorry

end NUMINAMATH_CALUDE_game_converges_to_black_hole_l1993_199300


namespace NUMINAMATH_CALUDE_tan_no_intersection_l1993_199358

theorem tan_no_intersection :
  ∀ y : ℝ, ¬∃ x : ℝ, x = π/8 ∧ y = Real.tan (2*x + π/4) :=
by sorry

end NUMINAMATH_CALUDE_tan_no_intersection_l1993_199358


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1993_199384

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1993_199384


namespace NUMINAMATH_CALUDE_max_food_per_guest_l1993_199370

/-- Given a total amount of food and a minimum number of guests at a banquet,
    calculate the maximum amount of food an individual guest could have consumed. -/
theorem max_food_per_guest (total_food : ℝ) (min_guests : ℕ) 
    (h1 : total_food = 411) 
    (h2 : min_guests = 165) : 
  (total_food / min_guests : ℝ) = 411 / 165 := by
  sorry

end NUMINAMATH_CALUDE_max_food_per_guest_l1993_199370


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l1993_199375

/-- An arithmetic sequence with 30 terms, first term 3, and last term 89 has its 10th term equal to 30 -/
theorem arithmetic_sequence_10th_term :
  ∀ (a : ℕ → ℚ), 
    (∀ i j : ℕ, i < j → a j - a i = (j - i) * (a 1 - a 0)) →  -- arithmetic sequence
    (a 0 = 3) →                                               -- first term is 3
    (a 29 = 89) →                                             -- last term is 89
    (a 9 = 30) :=                                             -- 10th term (index 9) is 30
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l1993_199375


namespace NUMINAMATH_CALUDE_min_value_3a_plus_2_l1993_199312

theorem min_value_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) :
  ∃ (m : ℝ), (3 * a + 2 ≥ m) ∧ (∀ (x : ℝ), 8 * x^2 + 10 * x + 6 = 2 → 3 * x + 2 ≥ m) ∧ m = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_3a_plus_2_l1993_199312


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l1993_199309

/-- Given the cost of some pens and 5 pencils is Rs. 200, and the cost ratio of one pen to one pencil
    is 5:1, prove that the cost of one dozen pens is Rs. 120. -/
theorem cost_of_dozen_pens (n : ℕ) (x : ℚ) : 
  5 * n * x + 5 * x = 200 →  -- Cost of n pens and 5 pencils is 200
  (5 * x) / x = 5 / 1 →      -- Cost ratio of pen to pencil is 5:1
  12 * (5 * x) = 120 :=      -- Cost of dozen pens is 120
by sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l1993_199309


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l1993_199398

/-- The number of ways to choose 2 items from n items -/
def choose (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem rectangle_formation_count : 
  let horizontal_lines : ℕ := 5
  let vertical_lines : ℕ := 6
  (choose horizontal_lines 2) * (choose vertical_lines 2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l1993_199398


namespace NUMINAMATH_CALUDE_simultaneous_presence_probability_l1993_199377

/-- The probability of two people being at a location simultaneously -/
theorem simultaneous_presence_probability :
  let arrival_window : ℝ := 2  -- 2-hour window
  let stay_duration : ℝ := 1/3  -- 20 minutes in hours
  let total_area : ℝ := arrival_window * arrival_window
  let meeting_area : ℝ := total_area - 2 * (1/2 * stay_duration * (arrival_window - stay_duration))
  meeting_area / total_area = 4/9 := by
sorry

end NUMINAMATH_CALUDE_simultaneous_presence_probability_l1993_199377


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1993_199304

theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + 3 * Complex.I) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1993_199304


namespace NUMINAMATH_CALUDE_max_value_f_l1993_199321

open Real

/-- The maximum value of f(m, n) given the conditions -/
theorem max_value_f (f g : ℝ → ℝ) (m n : ℝ) :
  (∀ x > 0, f x = log x) →
  (∀ x, g x = (2*m + 3)*x + n) →
  (∀ x > 0, f x ≤ g x) →
  let f_mn := (2*m + 3) * n
  ∃ (min_f_mn : ℝ), f_mn ≥ min_f_mn ∧ 
    (∀ m' n', (∀ x > 0, log x ≤ (2*m' + 3)*x + n') → (2*m' + 3) * n' ≥ min_f_mn) →
  (∃ (max_value : ℝ), max_value = 1 / Real.exp 2 ∧
    ∀ m' n', (∀ x > 0, log x ≤ (2*m' + 3)*x + n') →
      let f_m'n' := (2*m' + 3) * n'
      ∃ (min_f_m'n' : ℝ), f_m'n' ≥ min_f_m'n' ∧
        (∀ m'' n'', (∀ x > 0, log x ≤ (2*m'' + 3)*x + n'') → (2*m'' + 3) * n'' ≥ min_f_m'n') →
      min_f_m'n' ≤ max_value) :=
by sorry

end NUMINAMATH_CALUDE_max_value_f_l1993_199321


namespace NUMINAMATH_CALUDE_weight_difference_proof_l1993_199360

/-- Proves that the difference between the average weight of two departing students
    and Joe's weight is 7 kg, given the conditions of the problem. -/
theorem weight_difference_proof
  (initial_avg : ℝ)
  (joe_weight : ℝ)
  (new_avg : ℝ)
  (final_avg : ℝ)
  (h_initial_avg : initial_avg = 30)
  (h_joe_weight : joe_weight = 44)
  (h_new_avg : new_avg = initial_avg + 1)
  (h_final_avg : final_avg = initial_avg)
  : ∃ (n : ℕ) (departing_avg : ℝ),
    (n : ℝ) * initial_avg + joe_weight = (n + 1 : ℝ) * new_avg ∧
    (n + 1 : ℝ) * new_avg - departing_avg * 2 = (n - 1 : ℝ) * final_avg ∧
    joe_weight - departing_avg = 7 :=
by sorry

end NUMINAMATH_CALUDE_weight_difference_proof_l1993_199360


namespace NUMINAMATH_CALUDE_exists_desired_arrangement_l1993_199340

/-- A type representing a 10x10 grid of natural numbers -/
def Grid := Fin 10 → Fin 10 → ℕ

/-- A type representing a domino (1x2 rectangle) in the grid -/
inductive Domino
| horizontal : Fin 10 → Fin 9 → Domino
| vertical : Fin 9 → Fin 10 → Domino

/-- A partition of the grid into dominoes -/
def Partition := List Domino

/-- Function to check if a partition is valid (covers the entire grid without overlaps) -/
def isValidPartition (p : Partition) : Prop := sorry

/-- Function to calculate the sum of numbers in a domino for a given grid -/
def dominoSum (g : Grid) (d : Domino) : ℕ := sorry

/-- Function to count the number of dominoes with even sum in a partition -/
def countEvenSumDominoes (g : Grid) (p : Partition) : ℕ := sorry

/-- The main theorem statement -/
theorem exists_desired_arrangement : 
  ∃ (g : Grid), ∀ (p : Partition), isValidPartition p → countEvenSumDominoes g p = 7 := by sorry

end NUMINAMATH_CALUDE_exists_desired_arrangement_l1993_199340


namespace NUMINAMATH_CALUDE_volleyball_starters_count_l1993_199373

def volleyball_team_size : ℕ := 16
def triplet_size : ℕ := 3
def starter_size : ℕ := 6

def choose_starters (team_size triplet_size starter_size : ℕ) : ℕ :=
  let non_triplet_size := team_size - triplet_size
  let with_one_triplet := triplet_size * Nat.choose non_triplet_size (starter_size - 1)
  let without_triplets := Nat.choose non_triplet_size starter_size
  with_one_triplet + without_triplets

theorem volleyball_starters_count :
  choose_starters volleyball_team_size triplet_size starter_size = 5577 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_starters_count_l1993_199373


namespace NUMINAMATH_CALUDE_baseball_card_theorem_l1993_199389

/-- Represents the amount of money each person has and the cost of the baseball card. -/
structure BaseballCardProblem where
  patricia_money : ℝ
  lisa_money : ℝ
  charlotte_money : ℝ
  card_cost : ℝ

/-- Calculates the additional money required to buy the baseball card. -/
def additional_money_required (problem : BaseballCardProblem) : ℝ :=
  problem.card_cost - (problem.patricia_money + problem.lisa_money + problem.charlotte_money)

/-- Theorem stating the additional money required is $49 given the problem conditions. -/
theorem baseball_card_theorem (problem : BaseballCardProblem) 
  (h1 : problem.patricia_money = 6)
  (h2 : problem.lisa_money = 5 * problem.patricia_money)
  (h3 : problem.lisa_money = 2 * problem.charlotte_money)
  (h4 : problem.card_cost = 100) :
  additional_money_required problem = 49 := by
  sorry

#eval additional_money_required { 
  patricia_money := 6, 
  lisa_money := 30, 
  charlotte_money := 15, 
  card_cost := 100 
}

end NUMINAMATH_CALUDE_baseball_card_theorem_l1993_199389


namespace NUMINAMATH_CALUDE_compound_interest_principal_l1993_199339

/-- Proves that given specific compound interest conditions, the principal amount is 1500 --/
theorem compound_interest_principal :
  ∀ (CI R T P : ℝ),
    CI = 315 →
    R = 10 →
    T = 2 →
    CI = P * ((1 + R / 100) ^ T - 1) →
    P = 1500 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l1993_199339


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l1993_199397

theorem merchant_profit_percentage (C S : ℝ) (h : 24 * C = 16 * S) : 
  (S - C) / C * 100 = 50 :=
sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l1993_199397


namespace NUMINAMATH_CALUDE_range_of_a_l1993_199386

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → a < -x^2 + 2*x) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1993_199386


namespace NUMINAMATH_CALUDE_box_volume_ratio_l1993_199383

/-- The volume of a rectangular box -/
def box_volume (length width height : ℝ) : ℝ := length * width * height

/-- Alex's box dimensions -/
def alex_length : ℝ := 8
def alex_width : ℝ := 6
def alex_height : ℝ := 12

/-- Felicia's box dimensions -/
def felicia_length : ℝ := 12
def felicia_width : ℝ := 6
def felicia_height : ℝ := 8

theorem box_volume_ratio :
  (box_volume alex_length alex_width alex_height) / (box_volume felicia_length felicia_width felicia_height) = 1 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_ratio_l1993_199383


namespace NUMINAMATH_CALUDE_waiter_earnings_l1993_199382

theorem waiter_earnings (total_customers : ℕ) (non_tippers : ℕ) (tip_amount : ℕ) : 
  total_customers = 9 → 
  non_tippers = 5 → 
  tip_amount = 8 → 
  (total_customers - non_tippers) * tip_amount = 32 := by
sorry

end NUMINAMATH_CALUDE_waiter_earnings_l1993_199382


namespace NUMINAMATH_CALUDE_best_optimistic_coefficient_l1993_199301

theorem best_optimistic_coefficient 
  (a b c x : ℝ) 
  (h1 : a < b) 
  (h2 : 0 < x) 
  (h3 : x < 1) 
  (h4 : c = a + x * (b - a)) 
  (h5 : (c - a)^2 = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_best_optimistic_coefficient_l1993_199301


namespace NUMINAMATH_CALUDE_angle_b_value_l1993_199387

-- Define the angles
variable (a b c : ℝ)

-- Define the conditions
axiom straight_line : a + b + c = 180
axiom ratio_b_a : b = 2 * a
axiom ratio_c_b : c = 3 * b

-- Theorem to prove
theorem angle_b_value : b = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_value_l1993_199387


namespace NUMINAMATH_CALUDE_max_value_product_l1993_199394

theorem max_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 8) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 8 → (1 + a) * (1 + b) ≤ (1 + x) * (1 + y)) ∧
  (1 + x) * (1 + y) = 25 :=
sorry

end NUMINAMATH_CALUDE_max_value_product_l1993_199394
