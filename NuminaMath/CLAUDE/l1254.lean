import Mathlib

namespace NUMINAMATH_CALUDE_equality_of_exponents_l1254_125484

-- Define variables
variable (a b c d : ℝ)
variable (x y q z : ℝ)

-- State the theorem
theorem equality_of_exponents 
  (h1 : a^x = c^q) 
  (h2 : a^x = b) 
  (h3 : c^y = a^z) 
  (h4 : c^y = d) 
  : x * y = q * z := by
  sorry

end NUMINAMATH_CALUDE_equality_of_exponents_l1254_125484


namespace NUMINAMATH_CALUDE_binomial_15_12_l1254_125427

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_l1254_125427


namespace NUMINAMATH_CALUDE_soccer_balls_added_l1254_125408

/-- Given the initial number of soccer balls, the number removed, and the final number of balls,
    prove that the number of soccer balls added is 21. -/
theorem soccer_balls_added 
  (initial : ℕ) 
  (removed : ℕ) 
  (final : ℕ) 
  (h1 : initial = 6) 
  (h2 : removed = 3) 
  (h3 : final = 24) : 
  final - (initial - removed) = 21 := by
  sorry

end NUMINAMATH_CALUDE_soccer_balls_added_l1254_125408


namespace NUMINAMATH_CALUDE_number_problem_l1254_125465

theorem number_problem : 
  ∃ x : ℚ, (x / 5 = 3 * (x / 6) - 40) ∧ (x = 400 / 3) := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1254_125465


namespace NUMINAMATH_CALUDE_square_root_of_four_l1254_125423

theorem square_root_of_four : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1254_125423


namespace NUMINAMATH_CALUDE_closest_to_10_percent_increase_l1254_125441

def students : Fin 6 → ℕ
  | 0 => 80  -- 2010
  | 1 => 88  -- 2011
  | 2 => 90  -- 2012
  | 3 => 99  -- 2013
  | 4 => 102 -- 2014
  | 5 => 110 -- 2015

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def isClosestTo10Percent (i : Fin 5) : Prop :=
  percentageIncrease (students i) (students (i.succ)) = 10

theorem closest_to_10_percent_increase :
  (isClosestTo10Percent 0 ∧ isClosestTo10Percent 2) ∧
  (∀ i : Fin 5, isClosestTo10Percent i → (i = 0 ∨ i = 2)) :=
by sorry

end NUMINAMATH_CALUDE_closest_to_10_percent_increase_l1254_125441


namespace NUMINAMATH_CALUDE_circle_diameter_twice_radius_l1254_125405

/-- A circle with a center, radius, and diameter. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  diameter : ℝ

/-- The diameter of a circle is twice its radius. -/
theorem circle_diameter_twice_radius (c : Circle) : c.diameter = 2 * c.radius := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_twice_radius_l1254_125405


namespace NUMINAMATH_CALUDE_max_trailing_zeros_l1254_125424

theorem max_trailing_zeros (a b c : ℕ) (sum_condition : a + b + c = 1003) :
  ∀ n : ℕ, (∃ k : ℕ, a * b * c = k * 10^n) → n ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_l1254_125424


namespace NUMINAMATH_CALUDE_peaches_before_equals_34_l1254_125437

/-- The number of peaches Mike picked from the orchard -/
def peaches_picked : ℕ := 52

/-- The current total number of peaches at the stand -/
def current_total : ℕ := 86

/-- The number of peaches Mike had left at the stand before picking more -/
def peaches_before : ℕ := current_total - peaches_picked

theorem peaches_before_equals_34 : peaches_before = 34 := by sorry

end NUMINAMATH_CALUDE_peaches_before_equals_34_l1254_125437


namespace NUMINAMATH_CALUDE_a2016_equals_2025_l1254_125454

/-- An arithmetic sequence with common difference 2 and a2007 = 2007 -/
def arithmetic_seq (n : ℕ) : ℕ :=
  2007 + 2 * (n - 2007)

/-- Theorem stating that the 2016th term of the sequence is 2025 -/
theorem a2016_equals_2025 : arithmetic_seq 2016 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_a2016_equals_2025_l1254_125454


namespace NUMINAMATH_CALUDE_track_width_l1254_125496

theorem track_width (r₁ r₂ : ℝ) (h : r₁ > r₂) :
  2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi →
  r₁ - r₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_track_width_l1254_125496


namespace NUMINAMATH_CALUDE_max_consecutive_good_proof_l1254_125409

/-- Sum of all positive divisors of n -/
def α (n : ℕ) : ℕ := sorry

/-- A number n is "good" if gcd(n, α(n)) = 1 -/
def is_good (n : ℕ) : Prop := Nat.gcd n (α n) = 1

/-- The maximum number of consecutive good numbers -/
def max_consecutive_good : ℕ := 5

theorem max_consecutive_good_proof :
  ∀ k : ℕ, k > max_consecutive_good →
    ∃ n : ℕ, n ≥ 2 ∧ ∃ i : Fin k, ¬is_good (n + i) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_good_proof_l1254_125409


namespace NUMINAMATH_CALUDE_units_digit_product_minus_cube_l1254_125488

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_product_minus_cube : units_digit (8 * 18 * 1998 - 8^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_minus_cube_l1254_125488


namespace NUMINAMATH_CALUDE_list_price_calculation_l1254_125419

theorem list_price_calculation (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) → 
  list_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_list_price_calculation_l1254_125419


namespace NUMINAMATH_CALUDE_common_measure_proof_l1254_125449

def segment1 : ℚ := 1/5
def segment2 : ℚ := 1/3
def commonMeasure : ℚ := 1/15

theorem common_measure_proof :
  (∃ (n m : ℕ), n * commonMeasure = segment1 ∧ m * commonMeasure = segment2) ∧
  (∀ (x : ℚ), x > 0 → (∃ (n m : ℕ), n * x = segment1 ∧ m * x = segment2) → x ≤ commonMeasure) :=
by sorry

end NUMINAMATH_CALUDE_common_measure_proof_l1254_125449


namespace NUMINAMATH_CALUDE_computation_proof_l1254_125452

theorem computation_proof : 
  20 * (150 / 3 + 36 / 4 + 4 / 25 + 2) = 1223 + 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l1254_125452


namespace NUMINAMATH_CALUDE_inequality_proof_l1254_125494

theorem inequality_proof (x y : ℝ) (n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  (x^n / (x + y^3)) + (y^n / (x^3 + y)) ≥ (2^(4-n)) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1254_125494


namespace NUMINAMATH_CALUDE_bucket_capacity_proof_l1254_125450

theorem bucket_capacity_proof (x : ℝ) : 
  (12 * x = 132 * 5) → x = 55 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_proof_l1254_125450


namespace NUMINAMATH_CALUDE_simplify_2M_minus_N_value_at_specific_points_independence_condition_l1254_125464

-- Define the polynomials M and N
def M (x y : ℝ) : ℝ := x^2 + x*y + 2*y - 2
def N (x y : ℝ) : ℝ := 2*x^2 - 2*x*y + x - 4

-- Theorem 1: Simplification of 2M - N
theorem simplify_2M_minus_N (x y : ℝ) :
  2 * M x y - N x y = 4*x*y + 4*y - x :=
sorry

-- Theorem 2: Value of 2M - N when x = -2 and y = -4
theorem value_at_specific_points :
  2 * M (-2) (-4) - N (-2) (-4) = 18 :=
sorry

-- Theorem 3: Condition for 2M - N to be independent of x
theorem independence_condition (y : ℝ) :
  (∀ x : ℝ, ∃ c : ℝ, 2 * M x y - N x y = c) ↔ y = 1/4 :=
sorry

end NUMINAMATH_CALUDE_simplify_2M_minus_N_value_at_specific_points_independence_condition_l1254_125464


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_positive_l1254_125487

/-- Given an arithmetic sequence {a_n} where S_n denotes the sum of its first n terms,
    if S_(2k+1) > 0, then a_(k+1) > 0. -/
theorem arithmetic_sequence_middle_term_positive
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (k : ℕ)      -- An arbitrary natural number
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence condition
  (h_sum : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2)  -- Sum formula for arithmetic sequence
  (h_positive : S (2 * k + 1) > 0)  -- Given condition
  : a (k + 1) > 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_positive_l1254_125487


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1254_125428

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 - Real.sqrt 5 * Complex.I → z = Real.sqrt 5 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1254_125428


namespace NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_eight_l1254_125448

theorem difference_of_cubes_divisible_by_eight (a b : ℤ) :
  ∃ k : ℤ, (2*a - 1)^3 - (2*b - 1)^3 = 8 * k := by
sorry

end NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_eight_l1254_125448


namespace NUMINAMATH_CALUDE_lionel_walked_four_miles_l1254_125406

-- Define the constants from the problem
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287
def total_feet : ℕ := 25332
def feet_per_yard : ℕ := 3
def feet_per_mile : ℕ := 5280

-- Define Lionel's distance in miles
def lionel_miles : ℚ := 4

-- Theorem statement
theorem lionel_walked_four_miles :
  (total_feet - (esther_yards * feet_per_yard + niklaus_feet)) / feet_per_mile = lionel_miles := by
  sorry

end NUMINAMATH_CALUDE_lionel_walked_four_miles_l1254_125406


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l1254_125460

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = Complex.I * b) →  -- z is purely imaginary
  (∃ c : ℝ, (z - 3)^2 + Complex.I * 5 = Complex.I * c) →  -- (z-3)^2+5i is purely imaginary
  z = Complex.I * 3 ∨ z = Complex.I * (-3) :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l1254_125460


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1254_125466

/-- The number of toppings available for sandwiches -/
def num_toppings : ℕ := 9

/-- The number of patty choices available for sandwiches -/
def num_patties : ℕ := 2

/-- The total number of different sandwich combinations -/
def total_combinations : ℕ := 2^num_toppings * num_patties

theorem sandwich_combinations :
  total_combinations = 1024 :=
sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1254_125466


namespace NUMINAMATH_CALUDE_mans_rowing_speed_in_still_water_l1254_125416

/-- Proves that a man's rowing speed in still water is 15 km/h given the conditions of downstream travel --/
theorem mans_rowing_speed_in_still_water :
  let current_speed : ℝ := 3 -- km/h
  let distance : ℝ := 60 / 1000 -- 60 meters converted to km
  let time : ℝ := 11.999040076793857 / 3600 -- seconds converted to hours
  let downstream_speed : ℝ := distance / time
  downstream_speed = current_speed + 15 := by sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_in_still_water_l1254_125416


namespace NUMINAMATH_CALUDE_intersection_radius_l1254_125443

/-- A sphere intersecting planes -/
structure IntersectingSphere where
  /-- Center of the circle in xz-plane -/
  xz_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in xz-plane -/
  xz_radius : ℝ
  /-- Center of the circle in xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in xy-plane -/
  xy_radius : ℝ

/-- The theorem stating the radius of the xy-plane intersection -/
theorem intersection_radius (sphere : IntersectingSphere) 
  (h1 : sphere.xz_center = (3, 0, 3))
  (h2 : sphere.xz_radius = 2)
  (h3 : sphere.xy_center = (3, 3, 0)) :
  sphere.xy_radius = 3 := by
  sorry


end NUMINAMATH_CALUDE_intersection_radius_l1254_125443


namespace NUMINAMATH_CALUDE_square_roots_of_nine_l1254_125474

theorem square_roots_of_nine :
  {x : ℝ | x ^ 2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_roots_of_nine_l1254_125474


namespace NUMINAMATH_CALUDE_f_inequality_l1254_125470

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Condition 1: Periodicity
axiom periodic (x : ℝ) : f (x + 4) = f x

-- Condition 2: Decreasing on [0, 2]
axiom decreasing (x₁ x₂ : ℝ) (h : 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2) : f x₁ > f x₂

-- Condition 3: Symmetry about y-axis for f(x-2)
axiom symmetry (x : ℝ) : f ((-x) - 2) = f (x - 2)

-- Theorem to prove
theorem f_inequality : f (-1.5) < f 7 ∧ f 7 < f (-4.5) := by sorry

end NUMINAMATH_CALUDE_f_inequality_l1254_125470


namespace NUMINAMATH_CALUDE_maggie_grandfather_subscriptions_l1254_125410

/-- Represents the number of magazine subscriptions Maggie sold to her grandfather. -/
def grandfather_subscriptions : ℕ := sorry

/-- The amount Maggie earns per subscription in dollars. -/
def earnings_per_subscription : ℕ := 5

/-- The number of subscriptions Maggie sold to her parents. -/
def parent_subscriptions : ℕ := 4

/-- The number of subscriptions Maggie sold to the next-door neighbor. -/
def neighbor_subscriptions : ℕ := 2

/-- The number of subscriptions Maggie sold to another neighbor. -/
def other_neighbor_subscriptions : ℕ := 2 * neighbor_subscriptions

/-- The total amount Maggie earned in dollars. -/
def total_earnings : ℕ := 55

/-- Theorem stating that Maggie sold 1 subscription to her grandfather. -/
theorem maggie_grandfather_subscriptions : grandfather_subscriptions = 1 := by
  sorry

end NUMINAMATH_CALUDE_maggie_grandfather_subscriptions_l1254_125410


namespace NUMINAMATH_CALUDE_race_time_proof_l1254_125495

/-- Given a race with the following conditions:
    - The race distance is 240 meters
    - Runner A beats runner B by either 56 meters or 7 seconds
    This theorem proves that runner A's time to complete the race is 23 seconds. -/
theorem race_time_proof (race_distance : ℝ) (distance_diff : ℝ) (time_diff : ℝ) :
  race_distance = 240 ∧ distance_diff = 56 ∧ time_diff = 7 →
  ∃ (time_A : ℝ), time_A = 23 ∧
    (race_distance / time_A = (race_distance - distance_diff) / (time_A + time_diff)) :=
by sorry

end NUMINAMATH_CALUDE_race_time_proof_l1254_125495


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1254_125467

/-- Proves that the cost price of an article is 350, given the selling price and profit percentage. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 455 → profit_percentage = 30 → 
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 350 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1254_125467


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_l1254_125431

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 3/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 3/b = 1 → x/2 + y/3 ≤ a/2 + b/3 :=
by sorry

theorem min_value_is_four (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 3/y = 1) :
  x/2 + y/3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_l1254_125431


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1254_125489

theorem intersection_point_of_lines (x y : ℚ) : 
  (5 * x - 2 * y = 8) ∧ (3 * x + 4 * y = 12) ↔ (x = 28/13 ∧ y = 18/13) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1254_125489


namespace NUMINAMATH_CALUDE_shortest_path_on_parallelepiped_l1254_125463

/-- The shortest path on the surface of a rectangular parallelepiped -/
theorem shortest_path_on_parallelepiped (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let surface_paths := [
    Real.sqrt ((a + c + a)^2 + b^2),
    Real.sqrt ((a + b + a)^2 + c^2),
    Real.sqrt ((b + a + b)^2 + c^2)
  ]
  ∃ (path : ℝ), path ∈ surface_paths ∧ path = Real.sqrt 125 ∧ ∀ x ∈ surface_paths, path ≤ x :=
by sorry

end NUMINAMATH_CALUDE_shortest_path_on_parallelepiped_l1254_125463


namespace NUMINAMATH_CALUDE_inverse_of_three_mod_191_l1254_125491

theorem inverse_of_three_mod_191 : ∃ x : ℕ, x < 191 ∧ (3 * x) % 191 = 1 ∧ x = 64 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_three_mod_191_l1254_125491


namespace NUMINAMATH_CALUDE_elaine_rent_percentage_l1254_125413

/-- The percentage of Elaine's annual earnings spent on rent last year -/
def rent_percentage_last_year : ℝ := 20

/-- Elaine's earnings this year as a percentage of last year's earnings -/
def earnings_ratio : ℝ := 115

/-- The percentage of Elaine's earnings spent on rent this year -/
def rent_percentage_this_year : ℝ := 25

/-- The ratio of this year's rent expenditure to last year's rent expenditure -/
def rent_expenditure_ratio : ℝ := 143.75

theorem elaine_rent_percentage :
  rent_percentage_this_year * earnings_ratio / 100 = 
  rent_expenditure_ratio * rent_percentage_last_year / 100 :=
sorry

end NUMINAMATH_CALUDE_elaine_rent_percentage_l1254_125413


namespace NUMINAMATH_CALUDE_fraction_equality_l1254_125411

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z) / (1/w - 1/z) = 2014) : 
  (w + z) / (w - z) = -2014 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1254_125411


namespace NUMINAMATH_CALUDE_function_has_max_and_min_l1254_125444

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*((a + 2)*x + 1)

-- State the theorem
theorem function_has_max_and_min (a : ℝ) :
  (∃ x₁ x₂ : ℝ, ∀ x : ℝ, f a x₁ ≤ f a x ∧ f a x ≤ f a x₂) ↔ (a > 2 ∨ a < -1) :=
sorry

end NUMINAMATH_CALUDE_function_has_max_and_min_l1254_125444


namespace NUMINAMATH_CALUDE_cubic_complex_equation_l1254_125481

theorem cubic_complex_equation (a b c : ℕ+) :
  c = Complex.I.re * ((a + Complex.I * b) ^ 3 - 107 * Complex.I) →
  c = 198 := by
sorry

end NUMINAMATH_CALUDE_cubic_complex_equation_l1254_125481


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1254_125473

theorem sufficient_not_necessary_condition (x : ℝ) :
  (|x - 1/2| < 1/2 → x < 1) ∧ ¬(x < 1 → |x - 1/2| < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1254_125473


namespace NUMINAMATH_CALUDE_r_bounds_for_area_range_l1254_125403

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 + 2

/-- The line function -/
def line (r : ℝ) (x : ℝ) : ℝ := r - 1

/-- The intersection points of the parabola and the line -/
def intersection_points (r : ℝ) : Set ℝ := {x | parabola x = line r x}

/-- The area of the triangle formed by the vertex of the parabola and the intersection points -/
def triangle_area (r : ℝ) : ℝ := (r - 3)^(3/2)

/-- Theorem stating the relationship between r and the area of the triangle -/
theorem r_bounds_for_area_range :
  ∀ r : ℝ, (16 ≤ triangle_area r ∧ triangle_area r ≤ 128) ↔ (7 ≤ r ∧ r ≤ 19) :=
sorry

end NUMINAMATH_CALUDE_r_bounds_for_area_range_l1254_125403


namespace NUMINAMATH_CALUDE_sin_half_theta_l1254_125475

theorem sin_half_theta (θ : Real) (h1 : |Real.cos θ| = 1/5) (h2 : 5*Real.pi/2 < θ) (h3 : θ < 3*Real.pi) :
  Real.sin (θ/2) = -Real.sqrt 15 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_half_theta_l1254_125475


namespace NUMINAMATH_CALUDE_parabola_hyperbola_coincidence_l1254_125429

/-- The value of p for which the focus of the parabola y^2 = 2px coincides with
    the right vertex of the hyperbola x^2/4 - y^2 = 1 -/
theorem parabola_hyperbola_coincidence (p : ℝ) : 
  (∃ x y : ℝ, y^2 = 2*p*x ∧ x^2/4 - y^2 = 1 ∧ x = p/2 ∧ y = 0) → p = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_coincidence_l1254_125429


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1254_125400

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 2*a*(a-3) - a^2 = a^2 - 6*a := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) : (x-1)*(x+2) - x*(x+1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1254_125400


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1254_125414

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a^3 + b^3 = 152) : 
  a * b = 15 := by sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1254_125414


namespace NUMINAMATH_CALUDE_candle_duration_first_scenario_l1254_125456

/-- The number of nights a candle lasts when burned for a given number of hours per night. -/
def candle_duration (hours_per_night : ℕ) : ℕ :=
  sorry

/-- The number of candles used over a given number of nights when burned for a given number of hours per night. -/
def candles_used (nights : ℕ) (hours_per_night : ℕ) : ℕ :=
  sorry

theorem candle_duration_first_scenario :
  let first_scenario_hours := 1
  let second_scenario_hours := 2
  let second_scenario_nights := 24
  let second_scenario_candles := 6
  candle_duration first_scenario_hours = 8 ∧
  candle_duration second_scenario_hours * second_scenario_candles = second_scenario_nights :=
by sorry

end NUMINAMATH_CALUDE_candle_duration_first_scenario_l1254_125456


namespace NUMINAMATH_CALUDE_five_fourths_of_three_and_one_third_l1254_125446

theorem five_fourths_of_three_and_one_third (x : ℚ) :
  x = 3 + 1 / 3 → (5 / 4 : ℚ) * x = 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_three_and_one_third_l1254_125446


namespace NUMINAMATH_CALUDE_abc_fraction_value_l1254_125440

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 4)
  (h2 : b * c / (b + c) = 5)
  (h3 : c * a / (c + a) = 7) :
  a * b * c / (a * b + b * c + c * a) = 280 / 83 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l1254_125440


namespace NUMINAMATH_CALUDE_num_bottles_is_four_l1254_125498

-- Define the weight of a bag of chips
def bag_weight : ℕ := 400

-- Define the weight difference between a bag of chips and a bottle of juice
def weight_difference : ℕ := 350

-- Define the total weight of 5 bags of chips and some bottles of juice
def total_weight : ℕ := 2200

-- Define the number of bags of chips
def num_bags : ℕ := 5

-- Define the weight of a bottle of juice
def bottle_weight : ℕ := bag_weight - weight_difference

-- Define the function to calculate the number of bottles
def num_bottles : ℕ :=
  (total_weight - num_bags * bag_weight) / bottle_weight

-- Theorem statement
theorem num_bottles_is_four :
  num_bottles = 4 :=
sorry

end NUMINAMATH_CALUDE_num_bottles_is_four_l1254_125498


namespace NUMINAMATH_CALUDE_f_two_equals_two_l1254_125477

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = (x^2 - x)/2 * f x + 2 - x

-- Theorem statement
theorem f_two_equals_two (h : has_property f) : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_two_equals_two_l1254_125477


namespace NUMINAMATH_CALUDE_star_commutative_l1254_125407

variable {M : Type*} [Nonempty M]
variable (star : M → M → M)

axiom left_inverse : ∀ a b : M, star (star a b) b = a
axiom right_inverse : ∀ a b : M, star a (star a b) = b

theorem star_commutative : ∀ a b : M, star a b = star b a := by sorry

end NUMINAMATH_CALUDE_star_commutative_l1254_125407


namespace NUMINAMATH_CALUDE_geometric_progression_a10_l1254_125493

/-- A geometric progression with given conditions -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_progression_a10 (a : ℕ → ℝ) :
  geometric_progression a → a 2 = 2 → a 6 = 162 → a 10 = 13122 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_a10_l1254_125493


namespace NUMINAMATH_CALUDE_fifth_inequality_l1254_125462

theorem fifth_inequality (n : ℕ) (h : n = 6) : 
  1 + (1 / 2^2) + (1 / 3^2) + (1 / 4^2) + (1 / 5^2) + (1 / 6^2) < (2 * n - 1) / n :=
by sorry

end NUMINAMATH_CALUDE_fifth_inequality_l1254_125462


namespace NUMINAMATH_CALUDE_bills_max_papers_l1254_125451

/-- Represents the number of items Bill can buy -/
structure BillsPurchase where
  pens : ℕ
  pencils : ℕ
  papers : ℕ

/-- The cost of Bill's purchase -/
def cost (b : BillsPurchase) : ℕ := 3 * b.pens + 5 * b.pencils + 9 * b.papers

/-- A purchase is valid if it meets the given conditions -/
def isValid (b : BillsPurchase) : Prop :=
  b.pens ≥ 2 ∧ b.pencils ≥ 1 ∧ cost b = 72

/-- The maximum number of papers Bill can buy -/
def maxPapers : ℕ := 6

theorem bills_max_papers :
  ∀ b : BillsPurchase, isValid b → b.papers ≤ maxPapers ∧
  ∃ b' : BillsPurchase, isValid b' ∧ b'.papers = maxPapers :=
sorry

end NUMINAMATH_CALUDE_bills_max_papers_l1254_125451


namespace NUMINAMATH_CALUDE_inequality_solution_l1254_125426

theorem inequality_solution (x : ℕ) : 5 * x + 3 < 3 * (2 + x) ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1254_125426


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l1254_125436

/-- The ratio of cylinder volumes formed by rolling a 6x9 rectangle -/
theorem cylinder_volume_ratio :
  let l₁ : ℝ := 6
  let l₂ : ℝ := 9
  let v₁ : ℝ := l₁ * l₂^2 / (4 * Real.pi)
  let v₂ : ℝ := l₂ * l₁^2 / (4 * Real.pi)
  max v₁ v₂ / min v₁ v₂ = 3/2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l1254_125436


namespace NUMINAMATH_CALUDE_correct_change_l1254_125485

/-- The price of a red candy in won -/
def red_candy_price : ℕ := 350

/-- The price of a blue candy in won -/
def blue_candy_price : ℕ := 180

/-- The number of red candies bought -/
def red_candy_count : ℕ := 3

/-- The number of blue candies bought -/
def blue_candy_count : ℕ := 2

/-- The amount Eunseo pays in won -/
def amount_paid : ℕ := 2000

/-- The change Eunseo should receive -/
def change : ℕ := amount_paid - (red_candy_price * red_candy_count + blue_candy_price * blue_candy_count)

theorem correct_change : change = 590 := by
  sorry

end NUMINAMATH_CALUDE_correct_change_l1254_125485


namespace NUMINAMATH_CALUDE_no_universal_divisibility_l1254_125432

def concatenate_two_digits (a b : Nat) : Nat :=
  10 * a + b

def concatenate_three_digits (a n b : Nat) : Nat :=
  100 * a + 10 * n + b

theorem no_universal_divisibility :
  ∀ n : Nat, ∃ a b : Nat,
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    ¬(concatenate_two_digits a b ∣ concatenate_three_digits a n b) := by
  sorry

end NUMINAMATH_CALUDE_no_universal_divisibility_l1254_125432


namespace NUMINAMATH_CALUDE_packs_per_carton_is_five_l1254_125412

/-- The number of sticks of gum in each pack -/
def sticks_per_pack : ℕ := 3

/-- The number of cartons in each brown box -/
def cartons_per_box : ℕ := 4

/-- The total number of sticks of gum in all brown boxes -/
def total_sticks : ℕ := 480

/-- The number of brown boxes -/
def num_boxes : ℕ := 8

/-- The number of packs of gum in each carton -/
def packs_per_carton : ℕ := total_sticks / (num_boxes * cartons_per_box * sticks_per_pack)

theorem packs_per_carton_is_five : packs_per_carton = 5 := by sorry

end NUMINAMATH_CALUDE_packs_per_carton_is_five_l1254_125412


namespace NUMINAMATH_CALUDE_max_parts_three_planes_exists_eight_parts_l1254_125476

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this problem

/-- The number of parts that a set of planes divides 3D space into -/
def num_parts (planes : List Plane3D) : ℕ :=
  sorry -- Definition not provided as it's not necessary for the statement

/-- Theorem: Three planes can divide 3D space into at most 8 parts -/
theorem max_parts_three_planes :
  ∀ (p1 p2 p3 : Plane3D), num_parts [p1, p2, p3] ≤ 8 :=
sorry

/-- Theorem: There exists a configuration of three planes that divides 3D space into exactly 8 parts -/
theorem exists_eight_parts :
  ∃ (p1 p2 p3 : Plane3D), num_parts [p1, p2, p3] = 8 :=
sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_exists_eight_parts_l1254_125476


namespace NUMINAMATH_CALUDE_distance_between_locations_l1254_125415

theorem distance_between_locations (speed_A speed_B : ℝ) (time : ℝ) (remaining_fraction : ℝ) : 
  speed_A = 60 →
  speed_B = 45 →
  time = 2 →
  remaining_fraction = 2 / 5 →
  (speed_A + speed_B) * time / (1 - remaining_fraction) = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_locations_l1254_125415


namespace NUMINAMATH_CALUDE_fish_catch_problem_l1254_125439

theorem fish_catch_problem (total_fish : ℕ) (tagged_fish : ℕ) (tagged_caught : ℕ) (second_catch : ℕ) :
  total_fish = 250 →
  tagged_fish = 50 →
  tagged_caught = 10 →
  (tagged_caught : ℚ) / second_catch = tagged_fish / total_fish →
  second_catch = 50 := by
sorry

end NUMINAMATH_CALUDE_fish_catch_problem_l1254_125439


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_progression_l1254_125461

/-- The first term of the arithmetic progression -/
def a₁ : ℤ := 113

/-- The common difference of the arithmetic progression -/
def d : ℤ := -4

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

/-- The n-th term of the arithmetic progression -/
def aₙ (n : ℕ) : ℤ := a₁ + (n - 1) * d

/-- The maximum number of terms before the sequence becomes non-positive -/
def max_n : ℕ := 29

theorem max_sum_arithmetic_progression :
  ∀ n : ℕ, S n ≤ S max_n ∧ S max_n = 1653 :=
sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_progression_l1254_125461


namespace NUMINAMATH_CALUDE_juliet_supporter_in_capulet_l1254_125435

-- Define the population distribution
def montague_pop : ℚ := 4/6
def capulet_pop : ℚ := 1/6
def verona_pop : ℚ := 1/6

-- Define the support percentages for Juliet
def montague_juliet : ℚ := 1/5  -- 20% support Juliet (100% - 80%)
def capulet_juliet : ℚ := 7/10
def verona_juliet : ℚ := 3/5

-- Theorem statement
theorem juliet_supporter_in_capulet :
  let total_juliet := montague_pop * montague_juliet + capulet_pop * capulet_juliet + verona_pop * verona_juliet
  (capulet_pop * capulet_juliet) / total_juliet = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_juliet_supporter_in_capulet_l1254_125435


namespace NUMINAMATH_CALUDE_hadley_books_l1254_125455

theorem hadley_books (initial_books : ℕ) 
  (h1 : initial_books - 50 + 40 - 30 = 60) : initial_books = 100 := by
  sorry

end NUMINAMATH_CALUDE_hadley_books_l1254_125455


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1254_125472

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h : ArithmeticSequence a) 
    (eq : a 1 + 3 * a 8 + a 15 = 120) : 
  3 * a 9 - a 11 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1254_125472


namespace NUMINAMATH_CALUDE_band_photo_arrangement_min_band_members_l1254_125457

theorem band_photo_arrangement (n : ℕ) : n > 0 ∧ 9 ∣ n ∧ 10 ∣ n ∧ 11 ∣ n → n ≥ 990 := by
  sorry

theorem min_band_members : ∃ n : ℕ, n > 0 ∧ 9 ∣ n ∧ 10 ∣ n ∧ 11 ∣ n ∧ n = 990 := by
  sorry

end NUMINAMATH_CALUDE_band_photo_arrangement_min_band_members_l1254_125457


namespace NUMINAMATH_CALUDE_parking_lot_pathway_distance_l1254_125479

theorem parking_lot_pathway_distance 
  (base1 : ℝ) 
  (height1 : ℝ) 
  (side2 : ℝ) 
  (h : base1 = 10) 
  (i : height1 = 30) 
  (j : side2 = 60) : 
  (base1 * height1) / side2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_pathway_distance_l1254_125479


namespace NUMINAMATH_CALUDE_bird_families_remaining_l1254_125480

theorem bird_families_remaining (initial : ℕ) (flew_away : ℕ) (remaining : ℕ) : 
  initial = 41 → flew_away = 27 → remaining = initial - flew_away → remaining = 14 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_remaining_l1254_125480


namespace NUMINAMATH_CALUDE_smallest_debt_resolution_l1254_125490

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 280

/-- A debt resolution is valid if it can be expressed as a combination of pigs and goats -/
def is_valid_resolution (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = pig_value * p + goat_value * g

/-- The smallest positive debt that can be resolved -/
def smallest_resolvable_debt : ℕ := 800

theorem smallest_debt_resolution :
  (is_valid_resolution smallest_resolvable_debt) ∧
  (∀ d : ℕ, d > 0 ∧ d < smallest_resolvable_debt → ¬(is_valid_resolution d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_debt_resolution_l1254_125490


namespace NUMINAMATH_CALUDE_no_solution_system_l1254_125483

theorem no_solution_system :
  ¬ ∃ x : ℝ, x > 2 ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l1254_125483


namespace NUMINAMATH_CALUDE_find_m_l1254_125420

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | (x : ℝ)^2 + m * x = 0}

theorem find_m :
  ∃ m : ℝ, (U \ A m = {1, 2}) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1254_125420


namespace NUMINAMATH_CALUDE_factorial_8_divisors_l1254_125445

def factorial_8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

theorem factorial_8_divisors :
  (factorial_8 = 2^7 * 3^2 * 5 * 7) →
  (∃ (even_divisors : Finset ℕ) (even_divisors_multiple_2_3 : Finset ℕ),
    (∀ d ∈ even_divisors, d ∣ factorial_8 ∧ 2 ∣ d) ∧
    (∀ d ∈ even_divisors_multiple_2_3, d ∣ factorial_8 ∧ 2 ∣ d ∧ 3 ∣ d) ∧
    even_divisors.card = 84 ∧
    even_divisors_multiple_2_3.card = 56) :=
by sorry

end NUMINAMATH_CALUDE_factorial_8_divisors_l1254_125445


namespace NUMINAMATH_CALUDE_playlist_song_length_l1254_125442

theorem playlist_song_length 
  (n_unknown : ℕ) 
  (n_known : ℕ) 
  (known_length : ℕ) 
  (total_duration : ℕ) : 
  n_unknown = 10 → 
  n_known = 15 → 
  known_length = 2 → 
  total_duration = 60 → 
  ∃ (unknown_length : ℕ), 
    unknown_length = 3 ∧ 
    n_unknown * unknown_length + n_known * known_length = total_duration :=
by sorry

end NUMINAMATH_CALUDE_playlist_song_length_l1254_125442


namespace NUMINAMATH_CALUDE_unique_remainder_mod_11_l1254_125438

theorem unique_remainder_mod_11 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_11_l1254_125438


namespace NUMINAMATH_CALUDE_number_puzzle_l1254_125401

theorem number_puzzle (x : ℝ) : 9 * (((x + 1.4) / 3) - 0.7) = 5.4 → x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1254_125401


namespace NUMINAMATH_CALUDE_range_of_fraction_l1254_125434

-- Define a monotonically decreasing function on ℝ
def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- Define symmetry of f(x-1) with respect to (1,0)
def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = -f x

-- Main theorem
theorem range_of_fraction (f : ℝ → ℝ) (h_decr : monotonically_decreasing f)
    (h_sym : symmetric_about_one f) :
    ∀ t : ℝ, f (t^2 - 2*t) + f (-3) > 0 → (t - 1) / (t - 3) < 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_fraction_l1254_125434


namespace NUMINAMATH_CALUDE_octagon_area_theorem_l1254_125478

/-- The area of an octagon formed by the intersection of two unit squares with the same center -/
def octagon_area (side_length : ℚ) : ℚ :=
  8 * (side_length * (1 / 2) * (1 / 2))

/-- The theorem stating the area of the octagon given the side length -/
theorem octagon_area_theorem (h : octagon_area (43 / 99) = 86 / 99) : True := by
  sorry

#eval octagon_area (43 / 99)

end NUMINAMATH_CALUDE_octagon_area_theorem_l1254_125478


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1254_125422

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x > y - 2) ∧
  ¬(∀ x y : ℝ, x > y - 2 → x > y) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1254_125422


namespace NUMINAMATH_CALUDE_work_scaling_l1254_125421

/-- Represents the time taken to complete a work given the number of people and amount of work -/
def timeTaken (people : ℕ) (work : ℕ) : ℕ := sorry

/-- The theorem states that if 3 people can do 3 times the work in 3 days, 
    then 9 people can do 9 times the work in the same number of days -/
theorem work_scaling (baseTime : ℕ) :
  timeTaken 3 3 = baseTime → timeTaken 9 9 = baseTime := by sorry

end NUMINAMATH_CALUDE_work_scaling_l1254_125421


namespace NUMINAMATH_CALUDE_triangle_8_8_15_l1254_125482

/-- Triangle Inequality Theorem: A set of three line segments can form a triangle
    if and only if the sum of the lengths of any two sides is greater than
    the length of the remaining side. -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of line segments with lengths 8cm, 8cm, and 15cm can form a triangle. -/
theorem triangle_8_8_15 : canFormTriangle 8 8 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_8_8_15_l1254_125482


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1254_125469

def A : Set ℤ := {x : ℤ | x^2 - 4*x ≤ 0}
def B : Set ℤ := {x : ℤ | -1 ≤ x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1254_125469


namespace NUMINAMATH_CALUDE_bird_feeder_theorem_l1254_125453

/-- Given a bird feeder with specified capacity and feeding rate, and accounting for stolen seed, 
    calculate the number of birds fed weekly. -/
theorem bird_feeder_theorem (feeder_capacity : ℚ) (birds_per_cup : ℕ) (stolen_amount : ℚ) : 
  feeder_capacity = 2 → 
  birds_per_cup = 14 → 
  stolen_amount = 1/2 → 
  (feeder_capacity - stolen_amount) * birds_per_cup = 21 := by
  sorry

end NUMINAMATH_CALUDE_bird_feeder_theorem_l1254_125453


namespace NUMINAMATH_CALUDE_sum_of_digits_of_calculation_l1254_125447

def calculation : ℕ := 100 * 1 + 50 * 2 + 25 * 4 + 2010

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10 + sum_of_digits (n / 10))

theorem sum_of_digits_of_calculation :
  sum_of_digits calculation = 303 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_calculation_l1254_125447


namespace NUMINAMATH_CALUDE_johns_extra_hours_l1254_125418

/-- Given John's work conditions, prove the number of extra hours he works for the bonus -/
theorem johns_extra_hours (regular_wage : ℝ) (regular_hours : ℝ) (bonus : ℝ) (bonus_hourly_rate : ℝ)
  (h1 : regular_wage = 80)
  (h2 : regular_hours = 8)
  (h3 : bonus = 20)
  (h4 : bonus_hourly_rate = 10) :
  (regular_wage + bonus) / bonus_hourly_rate - regular_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_extra_hours_l1254_125418


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l1254_125458

/-- Given a paint mixture with a ratio of red:yellow:white as 5:3:7,
    if 21 quarts of white paint are used, then 15 quarts of red paint should be used. -/
theorem paint_mixture_ratio (red yellow white : ℚ) :
  red / white = 5 / 7 →
  yellow / white = 3 / 7 →
  white = 21 →
  red = 15 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l1254_125458


namespace NUMINAMATH_CALUDE_correct_rows_per_bus_l1254_125433

/-- Represents the number of rows in each bus -/
def rows_per_bus : ℕ := 10

/-- Represents the number of columns in each bus -/
def columns_per_bus : ℕ := 4

/-- Represents the total number of buses -/
def total_buses : ℕ := 6

/-- Represents the total number of students that can be accommodated -/
def total_students : ℕ := 240

/-- Theorem stating that the number of rows per bus is correct -/
theorem correct_rows_per_bus : 
  rows_per_bus * columns_per_bus * total_buses = total_students := by
  sorry

end NUMINAMATH_CALUDE_correct_rows_per_bus_l1254_125433


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l1254_125430

theorem x_squared_plus_y_squared_equals_four
  (x y : ℝ)
  (h1 : x^3 = 3*y^2*x + 5 - Real.sqrt 7)
  (h2 : y^3 = 3*x^2*y + 5 + Real.sqrt 7) :
  x^2 + y^2 = 4 := by sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l1254_125430


namespace NUMINAMATH_CALUDE_smallest_class_size_l1254_125404

theorem smallest_class_size (n : ℕ) (h1 : n > 50) 
  (h2 : ∃ x : ℕ, n = 3*x + 2*(x+1)) : n ≥ 52 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l1254_125404


namespace NUMINAMATH_CALUDE_candied_yams_order_l1254_125497

theorem candied_yams_order (total_shoppers : ℕ) (buying_frequency : ℕ) (packages_per_box : ℕ) : 
  total_shoppers = 375 →
  buying_frequency = 3 →
  packages_per_box = 25 →
  (total_shoppers / buying_frequency) / packages_per_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_candied_yams_order_l1254_125497


namespace NUMINAMATH_CALUDE_cricket_bat_profit_l1254_125425

/-- Calculates the profit amount for a cricket bat sale -/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 900 →
  profit_percentage = 33.33 →
  ∃ (cost_price : ℝ), 
    cost_price > 0 ∧
    selling_price = cost_price * (1 + profit_percentage / 100) ∧
    selling_price - cost_price = 225 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_l1254_125425


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l1254_125486

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then (2*a + 3)*x - 4*a + 3 else a^x

-- State the theorem
theorem increasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l1254_125486


namespace NUMINAMATH_CALUDE_geometric_sequence_a9_l1254_125468

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a9 (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 = 1/2 →
  a 2 * a 8 = 2 * a 5 + 3 →
  a 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a9_l1254_125468


namespace NUMINAMATH_CALUDE_remaining_fabric_is_294_l1254_125402

/-- Represents the flag-making scenario with given initial conditions -/
structure FlagScenario where
  totalFabric : ℕ
  squareFlagSize : ℕ
  wideFlagWidth : ℕ
  wideFlagHeight : ℕ
  tallFlagWidth : ℕ
  tallFlagHeight : ℕ
  squareFlagsMade : ℕ
  wideFlagsMade : ℕ
  tallFlagsMade : ℕ

/-- Calculates the remaining fabric after making flags -/
def remainingFabric (scenario : FlagScenario) : ℕ :=
  scenario.totalFabric -
  (scenario.squareFlagSize * scenario.squareFlagSize * scenario.squareFlagsMade +
   scenario.wideFlagWidth * scenario.wideFlagHeight * scenario.wideFlagsMade +
   scenario.tallFlagWidth * scenario.tallFlagHeight * scenario.tallFlagsMade)

/-- Theorem stating that the remaining fabric is 294 square feet -/
theorem remaining_fabric_is_294 (scenario : FlagScenario)
  (h1 : scenario.totalFabric = 1000)
  (h2 : scenario.squareFlagSize = 4)
  (h3 : scenario.wideFlagWidth = 5)
  (h4 : scenario.wideFlagHeight = 3)
  (h5 : scenario.tallFlagWidth = 3)
  (h6 : scenario.tallFlagHeight = 5)
  (h7 : scenario.squareFlagsMade = 16)
  (h8 : scenario.wideFlagsMade = 20)
  (h9 : scenario.tallFlagsMade = 10) :
  remainingFabric scenario = 294 := by
  sorry

end NUMINAMATH_CALUDE_remaining_fabric_is_294_l1254_125402


namespace NUMINAMATH_CALUDE_spending_difference_l1254_125499

def akeno_spending : ℕ := 2985

def lev_spending : ℕ := akeno_spending / 3

def ambrocio_spending : ℕ := lev_spending - 177

def total_difference : ℕ := akeno_spending - (lev_spending + ambrocio_spending)

theorem spending_difference :
  total_difference = 1172 :=
by sorry

end NUMINAMATH_CALUDE_spending_difference_l1254_125499


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1254_125471

theorem geometric_sequence_sixth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^7 = 2) : 
  a * r^5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1254_125471


namespace NUMINAMATH_CALUDE_min_bird_species_l1254_125459

theorem min_bird_species (total_birds : ℕ) (h_total : total_birds = 2021) :
  let min_species := (total_birds + 1) / 2
  ∀ (num_species : ℕ),
    (∀ (i j : ℕ) (species : ℕ → ℕ),
      i < j ∧ j < total_birds ∧ species i = species j →
      ∃ (k : ℕ), k ∈ Finset.range (j - i - 1) ∧ species (i + k + 1) ≠ species i) →
    num_species ≥ min_species :=
by sorry

end NUMINAMATH_CALUDE_min_bird_species_l1254_125459


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1254_125417

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (5 * x - 4 < 3 - 2 * x) → x ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1254_125417


namespace NUMINAMATH_CALUDE_point_reflection_x_axis_l1254_125492

/-- Given a point A(2,3) in the Cartesian coordinate system, 
    its coordinates with respect to the x-axis are (2,-3). -/
theorem point_reflection_x_axis : 
  let A : ℝ × ℝ := (2, 3)
  let reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  reflect_x A = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_reflection_x_axis_l1254_125492
