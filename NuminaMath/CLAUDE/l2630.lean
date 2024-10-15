import Mathlib

namespace NUMINAMATH_CALUDE_square_root_of_8_l2630_263065

-- Define the square root property
def is_square_root (x : ℝ) (y : ℝ) : Prop := x * x = y

-- Theorem statement
theorem square_root_of_8 :
  ∃ (x : ℝ), is_square_root x 8 ∧ x = Real.sqrt 8 ∨ x = -Real.sqrt 8 :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_8_l2630_263065


namespace NUMINAMATH_CALUDE_five_sundays_in_july_l2630_263047

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month -/
structure Month where
  days : ℕ
  first_day : DayOfWeek

/-- Given a day of the week, returns the next day -/
def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the occurrences of a specific day in a month -/
def count_day_occurrences (m : Month) (d : DayOfWeek) : ℕ :=
  sorry

/-- Theorem: If June has five Fridays and 30 days, July (with 31 days) must have five Sundays -/
theorem five_sundays_in_july 
  (june : Month) 
  (july : Month) 
  (h1 : june.days = 30)
  (h2 : july.days = 31)
  (h3 : count_day_occurrences june DayOfWeek.Friday = 5)
  (h4 : july.first_day = next_day june.first_day) :
  count_day_occurrences july DayOfWeek.Sunday = 5 :=
sorry

end NUMINAMATH_CALUDE_five_sundays_in_july_l2630_263047


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l2630_263095

/-- Calculates the man's speed with the current given his speed against the current and the current's speed. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the man's speed against the current and the current's speed,
    the man's speed with the current is 12 km/hr. -/
theorem mans_speed_with_current :
  speed_with_current 8 2 = 12 := by
  sorry

#eval speed_with_current 8 2

end NUMINAMATH_CALUDE_mans_speed_with_current_l2630_263095


namespace NUMINAMATH_CALUDE_bottle_caps_found_at_park_l2630_263064

/-- Represents Danny's collection of bottle caps and wrappers --/
structure Collection where
  bottleCaps : ℕ
  wrappers : ℕ

/-- Represents the items Danny found at the park --/
structure ParkFindings where
  bottleCaps : ℕ
  wrappers : ℕ

/-- Theorem stating the number of bottle caps Danny found at the park --/
theorem bottle_caps_found_at_park 
  (initialCollection : Collection)
  (parkFindings : ParkFindings)
  (finalCollection : Collection)
  (h1 : parkFindings.wrappers = 18)
  (h2 : finalCollection.wrappers = 67)
  (h3 : finalCollection.bottleCaps = 35)
  (h4 : finalCollection.wrappers = finalCollection.bottleCaps + 32)
  (h5 : finalCollection.bottleCaps = initialCollection.bottleCaps + parkFindings.bottleCaps)
  (h6 : finalCollection.wrappers = initialCollection.wrappers + parkFindings.wrappers) :
  parkFindings.bottleCaps = 18 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_found_at_park_l2630_263064


namespace NUMINAMATH_CALUDE_greater_number_proof_l2630_263069

theorem greater_number_proof (x y : ℝ) (h1 : x > y) (h2 : x * y = 2688) (h3 : x + y - (x - y) = 64) : x = 84 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l2630_263069


namespace NUMINAMATH_CALUDE_grocer_sale_problem_l2630_263084

theorem grocer_sale_problem (sale1 sale2 sale3 sale5 average_sale : ℕ) 
  (h1 : sale1 = 5700)
  (h2 : sale2 = 8550)
  (h3 : sale3 = 6855)
  (h5 : sale5 = 14045)
  (h_avg : average_sale = 7800) :
  ∃ sale4 : ℕ, 
    sale4 = 3850 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = average_sale :=
by sorry

end NUMINAMATH_CALUDE_grocer_sale_problem_l2630_263084


namespace NUMINAMATH_CALUDE_cube_root_squared_eq_81_l2630_263025

theorem cube_root_squared_eq_81 (x : ℝ) :
  (x ^ (1/3)) ^ 2 = 81 → x = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_squared_eq_81_l2630_263025


namespace NUMINAMATH_CALUDE_percentage_of_red_cars_l2630_263072

theorem percentage_of_red_cars (total_cars : ℕ) (honda_cars : ℕ) 
  (honda_red_percentage : ℚ) (non_honda_red_percentage : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  honda_red_percentage = 90 / 100 →
  non_honda_red_percentage = 225 / 1000 →
  (((honda_red_percentage * honda_cars) + 
    (non_honda_red_percentage * (total_cars - honda_cars))) / total_cars) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_red_cars_l2630_263072


namespace NUMINAMATH_CALUDE_ab_zero_necessary_not_sufficient_l2630_263094

def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

theorem ab_zero_necessary_not_sufficient (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) → a * b = 0 ∧
  ∃ a b, a * b = 0 ∧ ¬(∀ x, f a b x = -f a b (-x)) :=
sorry

end NUMINAMATH_CALUDE_ab_zero_necessary_not_sufficient_l2630_263094


namespace NUMINAMATH_CALUDE_part_one_part_two_l2630_263062

/-- Given vectors in R^2 -/
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (0, 1)
def c : ℝ × ℝ := (1, -2)

/-- The theorem for the first part of the problem -/
theorem part_one : ∃ (m n : ℝ), a = m • b + n • c ∧ m = 3 ∧ n = 2 := by sorry

/-- The theorem for the second part of the problem -/
theorem part_two : 
  (∃ (d : ℝ × ℝ), ∃ (k : ℝ), k ≠ 0 ∧ (a + d) = k • (b + c)) ∧ 
  (∀ (d : ℝ × ℝ), (∃ (k : ℝ), k ≠ 0 ∧ (a + d) = k • (b + c)) → 
    Real.sqrt 2 / 2 ≤ Real.sqrt (d.1^2 + d.2^2)) ∧
  (∃ (d : ℝ × ℝ), (∃ (k : ℝ), k ≠ 0 ∧ (a + d) = k • (b + c)) ∧ 
    Real.sqrt (d.1^2 + d.2^2) = Real.sqrt 2 / 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2630_263062


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_20pi_l2630_263077

/-- Represents a triangular pyramid with vertex P and base ABC. -/
structure TriangularPyramid where
  PA : ℝ
  AB : ℝ
  angleCBA : ℝ
  perpendicular : Bool

/-- Calculates the surface area of the circumscribed sphere of a triangular pyramid. -/
def circumscribedSphereSurfaceArea (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The surface area of the circumscribed sphere of the given triangular pyramid is 20π. -/
theorem circumscribed_sphere_surface_area_is_20pi :
  let pyramid := TriangularPyramid.mk 2 2 (π/6) true
  circumscribedSphereSurfaceArea pyramid = 20 * π :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_20pi_l2630_263077


namespace NUMINAMATH_CALUDE_parallelogram_bisector_slope_l2630_263037

/-- A parallelogram with given vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- Predicate to check if a line cuts a parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- The main theorem -/
theorem parallelogram_bisector_slope (p : Parallelogram) (l : Line) :
  p.v1 = (5, 20) →
  p.v2 = (5, 50) →
  p.v3 = (20, 100) →
  p.v4 = (20, 70) →
  cuts_into_congruent_polygons p l →
  l.slope = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_bisector_slope_l2630_263037


namespace NUMINAMATH_CALUDE_closest_multiple_of_17_to_2502_l2630_263078

theorem closest_multiple_of_17_to_2502 :
  ∀ k : ℤ, k ≠ 147 → |2502 - 17 * 147| ≤ |2502 - 17 * k| :=
sorry

end NUMINAMATH_CALUDE_closest_multiple_of_17_to_2502_l2630_263078


namespace NUMINAMATH_CALUDE_oranges_per_box_l2630_263030

theorem oranges_per_box (total_oranges : ℕ) (total_boxes : ℕ) 
  (h1 : total_oranges = 2650) 
  (h2 : total_boxes = 265) :
  total_oranges / total_boxes = 10 := by
sorry

end NUMINAMATH_CALUDE_oranges_per_box_l2630_263030


namespace NUMINAMATH_CALUDE_number_base_conversion_l2630_263013

theorem number_base_conversion :
  ∃! (x y z b : ℕ),
    (x * b^2 + y * b + z = 1989) ∧
    (b^2 ≤ 1989) ∧
    (1989 < b^3) ∧
    (x + y + z = 27) ∧
    (0 ≤ x) ∧ (x < b) ∧
    (0 ≤ y) ∧ (y < b) ∧
    (0 ≤ z) ∧ (z < b) ∧
    (x = 5 ∧ y = 9 ∧ z = 13 ∧ b = 19) := by
  sorry

end NUMINAMATH_CALUDE_number_base_conversion_l2630_263013


namespace NUMINAMATH_CALUDE_tire_circumference_l2630_263014

/-- The circumference of a tire given its rotations per minute and the car's speed -/
theorem tire_circumference (rotations_per_minute : ℝ) (car_speed_kmh : ℝ) : 
  rotations_per_minute = 400 → car_speed_kmh = 24 → 
  (car_speed_kmh * 1000 / 60) / rotations_per_minute = 1 := by
  sorry

#check tire_circumference

end NUMINAMATH_CALUDE_tire_circumference_l2630_263014


namespace NUMINAMATH_CALUDE_budgets_equal_in_1996_l2630_263035

/-- Represents the year when the budgets of projects Q and V are equal -/
def year_budgets_equal (initial_q initial_v increase_q decrease_v : ℕ) : ℕ :=
  let n := (initial_v - initial_q) / (increase_q + decrease_v)
  1990 + n

/-- Theorem stating that the budgets of projects Q and V are equal in 1996 -/
theorem budgets_equal_in_1996 :
  year_budgets_equal 540000 780000 30000 10000 = 1996 := by
  sorry

#eval year_budgets_equal 540000 780000 30000 10000

end NUMINAMATH_CALUDE_budgets_equal_in_1996_l2630_263035


namespace NUMINAMATH_CALUDE_burger_combinations_count_l2630_263067

/-- The number of condiment options available. -/
def num_condiments : ℕ := 10

/-- The number of choices for meat patties. -/
def num_patty_choices : ℕ := 3

/-- The number of choices for bun types. -/
def num_bun_choices : ℕ := 2

/-- Calculates the total number of different burger combinations. -/
def total_burger_combinations : ℕ := 2^num_condiments * num_patty_choices * num_bun_choices

/-- Theorem stating that the total number of different burger combinations is 6144. -/
theorem burger_combinations_count : total_burger_combinations = 6144 := by
  sorry

end NUMINAMATH_CALUDE_burger_combinations_count_l2630_263067


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l2630_263008

theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 50*x + c = (x + a)^2) → c = 625 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l2630_263008


namespace NUMINAMATH_CALUDE_solution_in_interval_l2630_263009

theorem solution_in_interval :
  ∃ x₀ ∈ Set.Ioo 2 3, Real.log x₀ + x₀ - 4 = 0 := by sorry

end NUMINAMATH_CALUDE_solution_in_interval_l2630_263009


namespace NUMINAMATH_CALUDE_intersection_implies_solution_l2630_263041

/-- Two lines intersecting at a point imply the solution to a related system of equations -/
theorem intersection_implies_solution (b k : ℝ) : 
  (∃ (x y : ℝ), y = -3*x + b ∧ y = -k*x + 1 ∧ x = 1 ∧ y = -2) →
  (∀ (x y : ℝ), 3*x + y = b ∧ k*x + y = 1 ↔ x = 1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_solution_l2630_263041


namespace NUMINAMATH_CALUDE_complement_of_A_l2630_263046

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | (x - 1) * (x + 2) > 0}

theorem complement_of_A : Set.compl A = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2630_263046


namespace NUMINAMATH_CALUDE_even_function_sum_l2630_263028

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_sum (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : f 4 = 3) :
  f 4 + f (-4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l2630_263028


namespace NUMINAMATH_CALUDE_assignment_operation_l2630_263027

theorem assignment_operation (A : Int) : A = 15 → -A + 5 = -10 := by
  sorry

end NUMINAMATH_CALUDE_assignment_operation_l2630_263027


namespace NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l2630_263049

theorem complex_arithmetic_evaluation :
  6 - 5 * (7 - (Real.sqrt 16 + 2)^2) * 3 = -429 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l2630_263049


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a1_l2630_263057

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 3 = -6 ∧
  a 7 = a 5 + 4

/-- Theorem stating that under given conditions, a_1 = -10 -/
theorem arithmetic_sequence_a1 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a1_l2630_263057


namespace NUMINAMATH_CALUDE_juan_has_64_marbles_l2630_263019

/-- The number of marbles Connie has -/
def connie_marbles : ℕ := 39

/-- The number of additional marbles Juan has compared to Connie -/
def juan_extra_marbles : ℕ := 25

/-- The total number of marbles Juan has -/
def juan_marbles : ℕ := connie_marbles + juan_extra_marbles

theorem juan_has_64_marbles : juan_marbles = 64 := by
  sorry

end NUMINAMATH_CALUDE_juan_has_64_marbles_l2630_263019


namespace NUMINAMATH_CALUDE_shadow_boundary_is_constant_l2630_263093

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The xy-plane -/
def xyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Light source position -/
def lightSource : Point3D := ⟨0, -4, 3⟩

/-- The sphere in the problem -/
def problemSphere : Sphere := ⟨⟨0, 0, 2⟩, 2⟩

/-- A point on the boundary of the shadow -/
structure ShadowBoundaryPoint where
  x : ℝ
  y : ℝ

/-- The boundary function of the shadow -/
def shadowBoundary (p : ShadowBoundaryPoint) : Prop :=
  p.y = -19/4

theorem shadow_boundary_is_constant (s : Sphere) (l : Point3D) :
  s = problemSphere →
  l = lightSource →
  ∀ p : ShadowBoundaryPoint, shadowBoundary p := by
  sorry

#check shadow_boundary_is_constant

end NUMINAMATH_CALUDE_shadow_boundary_is_constant_l2630_263093


namespace NUMINAMATH_CALUDE_star_three_two_l2630_263079

/-- The star operation defined as a^3 + 3a^2b + 3ab^2 + b^3 -/
def star (a b : ℝ) : ℝ := a^3 + 3*a^2*b + 3*a*b^2 + b^3

/-- Theorem stating that 3 star 2 equals 125 -/
theorem star_three_two : star 3 2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_star_three_two_l2630_263079


namespace NUMINAMATH_CALUDE_marbles_problem_l2630_263015

theorem marbles_problem (fabian kyle miles : ℕ) : 
  fabian = 36 ∧ 
  fabian = 4 * kyle ∧ 
  fabian = 9 * miles → 
  kyle + miles = 13 := by
sorry

end NUMINAMATH_CALUDE_marbles_problem_l2630_263015


namespace NUMINAMATH_CALUDE_power_equation_solution_l2630_263089

theorem power_equation_solution (m : ℤ) : (7 : ℝ) ^ (2 * m) = (1 / 7 : ℝ) ^ (m - 30) → m = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2630_263089


namespace NUMINAMATH_CALUDE_unique_composite_with_square_predecessor_divisors_l2630_263036

/-- A natural number is composite if it has a proper divisor greater than 1 -/
def IsComposite (n : ℕ) : Prop := ∃ d : ℕ, 1 < d ∧ d < n ∧ n % d = 0

/-- A natural number is a perfect square if it's equal to some integer squared -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- Property: for every natural divisor d of n, d-1 is a perfect square -/
def HasSquarePredecessorDivisors (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 0 → n % d = 0 → IsPerfectSquare (d - 1)

theorem unique_composite_with_square_predecessor_divisors :
  ∃! n : ℕ, IsComposite n ∧ HasSquarePredecessorDivisors n ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_unique_composite_with_square_predecessor_divisors_l2630_263036


namespace NUMINAMATH_CALUDE_division_problem_l2630_263052

theorem division_problem (L S Q : ℝ) : 
  L - S = 1356 →
  S = 268.2 →
  L = S * Q + 15 →
  Q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2630_263052


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2630_263012

theorem shaded_area_calculation (r : Real) (h : r = 1) : 
  6 * (π * r^2) + 4 * (1/2 * π * r^2) = 8 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2630_263012


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l2630_263010

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ

/-- Calculates the maximum number of intersections between two convex polygons -/
def max_intersections (P₁ P₂ : ConvexPolygon) (k : ℕ) : ℕ :=
  k * P₂.sides

/-- Theorem stating the maximum number of intersections between two convex polygons -/
theorem max_intersections_theorem 
  (P₁ P₂ : ConvexPolygon) 
  (k : ℕ) 
  (h₁ : P₁.sides ≤ P₂.sides) 
  (h₂ : k ≤ P₁.sides) : 
  max_intersections P₁ P₂ k = k * P₂.sides :=
by
  sorry

#check max_intersections_theorem

end NUMINAMATH_CALUDE_max_intersections_theorem_l2630_263010


namespace NUMINAMATH_CALUDE_inequality_proof_l2630_263082

theorem inequality_proof (a b x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2630_263082


namespace NUMINAMATH_CALUDE_victors_class_size_l2630_263092

theorem victors_class_size (total_skittles : ℕ) (skittles_per_classmate : ℕ) 
  (h1 : total_skittles = 25)
  (h2 : skittles_per_classmate = 5) :
  total_skittles / skittles_per_classmate = 5 :=
by sorry

end NUMINAMATH_CALUDE_victors_class_size_l2630_263092


namespace NUMINAMATH_CALUDE_two_digit_product_less_than_five_digit_l2630_263016

theorem two_digit_product_less_than_five_digit : ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  a * b < 10000 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_less_than_five_digit_l2630_263016


namespace NUMINAMATH_CALUDE_product_minimum_value_l2630_263059

-- Define the functions h and k
def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

-- State the theorem
theorem product_minimum_value (x : ℝ) :
  (∀ x, -3 ≤ h x ∧ h x ≤ 4) →
  (∀ x, -1 ≤ k x ∧ k x ≤ 3) →
  -12 ≤ h x * k x :=
sorry

end NUMINAMATH_CALUDE_product_minimum_value_l2630_263059


namespace NUMINAMATH_CALUDE_fraction_unchanged_l2630_263091

theorem fraction_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x) / (2 * (x + y)) = x / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l2630_263091


namespace NUMINAMATH_CALUDE_game_result_l2630_263050

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 1

def allie_rolls : List ℕ := [5, 6, 1, 2, 3]
def betty_rolls : List ℕ := [6, 1, 1, 2, 3]

def calculate_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : 
  calculate_points allie_rolls * calculate_points betty_rolls = 169 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2630_263050


namespace NUMINAMATH_CALUDE_calculate_expression_l2630_263001

theorem calculate_expression : 
  4 * Real.sin (60 * π / 180) + (-1/3)⁻¹ - Real.sqrt 12 + abs (-5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2630_263001


namespace NUMINAMATH_CALUDE_greater_solution_quadratic_l2630_263038

theorem greater_solution_quadratic : 
  ∃ (x : ℝ), x^2 + 14*x - 88 = 0 ∧ 
  (∀ (y : ℝ), y^2 + 14*y - 88 = 0 → y ≤ x) ∧
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_greater_solution_quadratic_l2630_263038


namespace NUMINAMATH_CALUDE_trigonometric_form_of_negative_3i_l2630_263066

theorem trigonometric_form_of_negative_3i :
  ∀ z : ℂ, z = -3 * Complex.I →
  z = 3 * (Complex.cos (3 * Real.pi / 2) + Complex.I * Complex.sin (3 * Real.pi / 2)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_form_of_negative_3i_l2630_263066


namespace NUMINAMATH_CALUDE_sum_of_combinations_l2630_263044

theorem sum_of_combinations : Nat.choose 8 2 + Nat.choose 8 3 = 84 := by sorry

end NUMINAMATH_CALUDE_sum_of_combinations_l2630_263044


namespace NUMINAMATH_CALUDE_airplane_passengers_survey_is_census_l2630_263097

/-- A survey type -/
inductive SurveyType
| FrozenFood
| AirplanePassengers
| RefrigeratorLifespan
| EnvironmentalAwareness

/-- Predicate for whether a survey requires examining every individual -/
def requiresExaminingAll (s : SurveyType) : Prop :=
  match s with
  | .AirplanePassengers => True
  | _ => False

/-- Definition of a census -/
def isCensus (s : SurveyType) : Prop :=
  requiresExaminingAll s

theorem airplane_passengers_survey_is_census :
  isCensus SurveyType.AirplanePassengers := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_survey_is_census_l2630_263097


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2630_263060

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (2, x + 2)
  are_parallel a b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2630_263060


namespace NUMINAMATH_CALUDE_solve_for_q_l2630_263031

theorem solve_for_q : ∀ (k r q : ℚ),
  (4 / 5 : ℚ) = k / 90 →
  (4 / 5 : ℚ) = (k + r) / 105 →
  (4 / 5 : ℚ) = (q - r) / 150 →
  q = 132 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l2630_263031


namespace NUMINAMATH_CALUDE_triangle_property_l2630_263017

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C,
    prove that if sin A(a^2 + b^2 - c^2) = ab(2sin B - sin C),
    then A = π/3 and 3/2 < sin B + sin C ≤ √3 -/
theorem triangle_property (a b c A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_triangle : A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : Real.sin A * (a^2 + b^2 - c^2) = a * b * (2 * Real.sin B - Real.sin C)) :
  A = π/3 ∧ 3/2 < Real.sin B + Real.sin C ∧ Real.sin B + Real.sin C ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l2630_263017


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l2630_263075

/-- Proves that for a rectangular plot with given conditions, the length is 60 meters more than the breadth. -/
theorem rectangular_plot_length_difference (length breadth : ℝ) : 
  length = 80 ∧ 
  length > breadth ∧ 
  (4 * breadth + 2 * (length - breadth)) * 26.5 = 5300 →
  length - breadth = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l2630_263075


namespace NUMINAMATH_CALUDE_emily_marbles_problem_l2630_263033

theorem emily_marbles_problem (emily_initial : ℕ) (emily_final : ℕ) : 
  emily_initial = 6 →
  emily_final = 8 →
  ∃ (additional_marbles : ℕ),
    emily_final = emily_initial + 2 * emily_initial - 
      ((emily_initial + 2 * emily_initial) / 2 + additional_marbles) ∧
    additional_marbles = 1 :=
by sorry

end NUMINAMATH_CALUDE_emily_marbles_problem_l2630_263033


namespace NUMINAMATH_CALUDE_triangle_ratio_proof_l2630_263011

theorem triangle_ratio_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b^2 = a * c →
  a^2 + b * c = c^2 + a * c →
  c / (b * Real.sin B) = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_proof_l2630_263011


namespace NUMINAMATH_CALUDE_jessica_bank_balance_l2630_263032

theorem jessica_bank_balance (B : ℝ) : 
  B > 0 → 
  200 = (2/5) * B → 
  let remaining := B - 200
  let deposit := (1/5) * remaining
  remaining + deposit = 360 := by
sorry

end NUMINAMATH_CALUDE_jessica_bank_balance_l2630_263032


namespace NUMINAMATH_CALUDE_part_one_part_two_l2630_263042

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem part_one (a : ℝ) (h : a ≤ 2) :
  {x : ℝ | f a x ≥ 2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 5/2} := by sorry

-- Part 2
theorem part_two :
  {a : ℝ | a > 1 ∧ ∀ x, f a x + |x - 1| ≥ 1} = {a : ℝ | a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2630_263042


namespace NUMINAMATH_CALUDE_three_numbers_proof_l2630_263034

theorem three_numbers_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : (a + b + c) / 3 = b) (h4 : c - a = 321) (h5 : a + c = 777) : 
  a = 228 ∧ b = 549 ∧ c = 870 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_proof_l2630_263034


namespace NUMINAMATH_CALUDE_bus_ride_difference_l2630_263024

/-- Given Oscar's and Charlie's bus ride lengths, prove the difference between them -/
theorem bus_ride_difference (oscar_ride : ℝ) (charlie_ride : ℝ)
  (h1 : oscar_ride = 0.75)
  (h2 : charlie_ride = 0.25) :
  oscar_ride - charlie_ride = 0.50 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l2630_263024


namespace NUMINAMATH_CALUDE_expected_boy_girl_pairs_l2630_263099

/-- The number of boys in the line -/
def num_boys : ℕ := 9

/-- The number of girls in the line -/
def num_girls : ℕ := 15

/-- The total number of people in the line -/
def total_people : ℕ := num_boys + num_girls

/-- The number of adjacent pairs in the line -/
def num_pairs : ℕ := total_people - 1

/-- The probability of a boy-girl pair at any given position -/
def prob_boy_girl_pair : ℚ := (2 * (num_boys - 1) * (num_girls - 1)) / ((total_people - 2) * (total_people - 3))

theorem expected_boy_girl_pairs :
  (num_pairs : ℚ) * prob_boy_girl_pair = 920 / 77 := by sorry

end NUMINAMATH_CALUDE_expected_boy_girl_pairs_l2630_263099


namespace NUMINAMATH_CALUDE_arcade_vending_machines_total_beverages_in_arcade_l2630_263021

/-- Given the conditions of vending machines in an arcade, calculate the total number of beverages --/
theorem arcade_vending_machines (num_machines : ℕ) 
  (front_position : ℕ) (back_position : ℕ) 
  (top_position : ℕ) (bottom_position : ℕ) : ℕ :=
  let beverages_per_column := front_position + back_position - 1
  let rows_per_machine := top_position + bottom_position - 1
  let beverages_per_machine := beverages_per_column * rows_per_machine
  num_machines * beverages_per_machine

/-- Prove that the total number of beverages in the arcade is 3696 --/
theorem total_beverages_in_arcade : 
  arcade_vending_machines 28 14 20 3 2 = 3696 := by
  sorry

end NUMINAMATH_CALUDE_arcade_vending_machines_total_beverages_in_arcade_l2630_263021


namespace NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l2630_263045

theorem and_sufficient_not_necessary_for_or :
  (∀ p q : Prop, p ∧ q → p ∨ q) ∧
  (∃ p q : Prop, p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l2630_263045


namespace NUMINAMATH_CALUDE_girls_together_arrangement_person_not_in_middle_l2630_263051

-- Define the number of boys and girls
def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_people : ℕ := num_boys + num_girls

-- Define permutation and combination functions
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement A
theorem girls_together_arrangement :
  (A num_girls num_girls) * (A (num_boys + 1) (num_boys + 1)) =
  A num_girls num_girls * A 5 5 := by sorry

-- Statement C
theorem person_not_in_middle :
  (C (total_people - 1) 1) * (A (total_people - 1) (total_people - 1)) =
  C 6 1 * A 6 6 := by sorry

end NUMINAMATH_CALUDE_girls_together_arrangement_person_not_in_middle_l2630_263051


namespace NUMINAMATH_CALUDE_max_min_difference_c_l2630_263061

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 6) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 18) : 
  ∃ (c_max c_min : ℝ), 
    (∀ x : ℝ, (∃ y z : ℝ, x + y + z = 6 ∧ x^2 + y^2 + z^2 = 18) → c_min ≤ x ∧ x ≤ c_max) ∧
    c_max - c_min = 4 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l2630_263061


namespace NUMINAMATH_CALUDE_square_difference_l2630_263096

theorem square_difference : (169 * 169) - (168 * 168) = 337 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2630_263096


namespace NUMINAMATH_CALUDE_subset_union_of_product_zero_l2630_263080

variable {X : Type*}
variable (f g : X → ℝ)

def M (f : X → ℝ) := {x : X | f x = 0}
def N (g : X → ℝ) := {x : X | g x = 0}
def P (f g : X → ℝ) := {x : X | f x * g x = 0}

theorem subset_union_of_product_zero (hM : M f ≠ ∅) (hN : N g ≠ ∅) (hP : P f g ≠ ∅) :
  P f g ⊆ M f ∪ N g := by
  sorry

end NUMINAMATH_CALUDE_subset_union_of_product_zero_l2630_263080


namespace NUMINAMATH_CALUDE_equal_angles_with_vectors_l2630_263068

/-- Given two vectors a and b in ℝ², prove that the vector c satisfies the condition
    that the angle between c and a is equal to the angle between c and b. -/
theorem equal_angles_with_vectors (a b c : ℝ × ℝ) : 
  a = (1, 0) → b = (1, -Real.sqrt 3) → c = (Real.sqrt 3, -1) →
  (c.1 * a.1 + c.2 * a.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (a.1^2 + a.2^2)) =
  (c.1 * b.1 + c.2 * b.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (b.1^2 + b.2^2)) := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_with_vectors_l2630_263068


namespace NUMINAMATH_CALUDE_log_roots_sum_l2630_263007

theorem log_roots_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 2 * (Real.log a)^2 + 4 * (Real.log a) + 1 = 0 ∧ 
       2 * (Real.log b)^2 + 4 * (Real.log b) + 1 = 0) : 
  (Real.log a)^2 + Real.log (a^2) + a * b = Real.exp (-2) - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_log_roots_sum_l2630_263007


namespace NUMINAMATH_CALUDE_inverse_proportion_m_range_l2630_263026

/-- Given an inverse proportion function y = (1-m)/x passing through points (1, y₁) and (2, y₂),
    where y₁ > y₂, prove that m < 1 -/
theorem inverse_proportion_m_range (y₁ y₂ m : ℝ) : 
  y₁ = 1 - m → 
  y₂ = (1 - m) / 2 → 
  y₁ > y₂ → 
  m < 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_m_range_l2630_263026


namespace NUMINAMATH_CALUDE_line_equation_l2630_263022

/-- The ellipse in the first quadrant -/
def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/3 = 1 ∧ x > 0 ∧ y > 0

/-- The line l -/
def line_l (x y : ℝ) : Prop := ∃ (k m : ℝ), y = k*x + m ∧ k < 0 ∧ m > 0

/-- Points A and B on the ellipse and line l -/
def point_A_B (xA yA xB yB : ℝ) : Prop :=
  ellipse xA yA ∧ ellipse xB yB ∧ line_l xA yA ∧ line_l xB yB

/-- Points M and N on the axes -/
def point_M_N (xM yM xN yN : ℝ) : Prop :=
  xM < 0 ∧ yM = 0 ∧ xN = 0 ∧ yN > 0 ∧ line_l xM yM ∧ line_l xN yN

/-- Equal distances |MA| = |NB| -/
def equal_distances (xA yA xB yB xM yM xN yN : ℝ) : Prop :=
  (xA - xM)^2 + yA^2 = xB^2 + (yB - yN)^2

/-- Distance |MN| = 2√3 -/
def distance_MN (xM yM xN yN : ℝ) : Prop :=
  (xM - xN)^2 + (yM - yN)^2 = 12

theorem line_equation (xA yA xB yB xM yM xN yN : ℝ) :
  point_A_B xA yA xB yB →
  point_M_N xM yM xN yN →
  equal_distances xA yA xB yB xM yM xN yN →
  distance_MN xM yM xN yN →
  ∃ (x y : ℝ), x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0 ∧ line_l x y :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2630_263022


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2630_263085

/-- The measure of the largest angle in a pentagon ABCDE with specific angle conditions -/
theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 108 ∧ 
  B = 72 ∧ 
  C = D ∧ 
  E = 3 * C ∧ 
  A + B + C + D + E = 540 →
  (max A (max B (max C (max D E)))) = 216 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2630_263085


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l2630_263071

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
theorem retailer_profit_percent
  (purchase_price : ℚ)
  (overhead_expenses : ℚ)
  (selling_price : ℚ)
  (h1 : purchase_price = 225)
  (h2 : overhead_expenses = 15)
  (h3 : selling_price = 300) :
  (selling_price - (purchase_price + overhead_expenses)) / (purchase_price + overhead_expenses) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l2630_263071


namespace NUMINAMATH_CALUDE_inscribed_square_area_ratio_l2630_263054

/-- Given a square ABCD with side length s, and an inscribed square A'B'C'D' where each vertex
    of A'B'C'D' is on a diagonal of ABCD and equidistant from the center of ABCD,
    the area of A'B'C'D' is 1/5 of the area of ABCD. -/
theorem inscribed_square_area_ratio (s : ℝ) (h : s > 0) :
  let abcd_area := s^2
  let apbpcpdp_side := s / Real.sqrt 5
  let apbpcpdp_area := apbpcpdp_side^2
  apbpcpdp_area / abcd_area = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_square_area_ratio_l2630_263054


namespace NUMINAMATH_CALUDE_helen_raisin_cookies_l2630_263005

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_chocolate : ℕ := 519

/-- The number of raisin cookies Helen baked yesterday -/
def yesterday_raisin : ℕ := 300

/-- The number of chocolate chip cookies Helen baked today -/
def today_chocolate : ℕ := 359

/-- The difference in raisin cookies baked between yesterday and today -/
def raisin_difference : ℕ := 20

/-- The number of raisin cookies Helen baked today -/
def today_raisin : ℕ := yesterday_raisin - raisin_difference

theorem helen_raisin_cookies : today_raisin = 280 := by
  sorry

end NUMINAMATH_CALUDE_helen_raisin_cookies_l2630_263005


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2630_263056

theorem tan_alpha_plus_pi_fourth (α : Real) (h : Real.tan α = 2) : 
  Real.tan (α + π/4) = -3 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2630_263056


namespace NUMINAMATH_CALUDE_units_digit_of_17_pow_2007_l2630_263003

theorem units_digit_of_17_pow_2007 : ∃ n : ℕ, 17^2007 ≡ 3 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_17_pow_2007_l2630_263003


namespace NUMINAMATH_CALUDE_points_in_first_quadrant_l2630_263039

theorem points_in_first_quadrant (x y : ℝ) : 
  y > -x + 3 ∧ y > 3*x - 1 → x > 0 ∧ y > 0 :=
sorry

end NUMINAMATH_CALUDE_points_in_first_quadrant_l2630_263039


namespace NUMINAMATH_CALUDE_currency_conversion_l2630_263098

-- Define the conversion rates
def cents_per_jiao : ℝ := 10
def cents_per_yuan : ℝ := 100

-- Define the theorem
theorem currency_conversion :
  (5 / cents_per_jiao = 0.5) ∧ 
  (5 / cents_per_yuan = 0.05) ∧ 
  (3.25 * cents_per_yuan = 325) := by
sorry


end NUMINAMATH_CALUDE_currency_conversion_l2630_263098


namespace NUMINAMATH_CALUDE_minimize_y_l2630_263023

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2

/-- The theorem stating that (a+b)/2 minimizes y -/
theorem minimize_y (a b : ℝ) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y x_min a b ≤ y x a b ∧ x_min = (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_minimize_y_l2630_263023


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l2630_263083

/-- Represents a workshop with its production quantity -/
structure Workshop where
  quantity : ℕ

/-- Calculates the total sample size for stratified sampling -/
def calculateSampleSize (workshops : List Workshop) (sampledUnits : ℕ) (sampledWorkshopQuantity : ℕ) : ℕ :=
  let totalQuantity := workshops.map (·.quantity) |>.sum
  (sampledUnits * totalQuantity) / sampledWorkshopQuantity

theorem stratified_sampling_size :
  let workshops := [
    { quantity := 120 },  -- Workshop A
    { quantity := 80 },   -- Workshop B
    { quantity := 60 }    -- Workshop C
  ]
  let sampledUnits := 3
  let sampledWorkshopQuantity := 60
  calculateSampleSize workshops sampledUnits sampledWorkshopQuantity = 13 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_size_l2630_263083


namespace NUMINAMATH_CALUDE_randys_trip_length_l2630_263018

theorem randys_trip_length :
  ∀ (total_length : ℚ),
  (1 / 4 : ℚ) * total_length +  -- First part (gravel road)
  30 +                          -- Second part (pavement)
  (1 / 6 : ℚ) * total_length    -- Third part (dirt road)
  = total_length                -- Sum of all parts equals total length
  →
  total_length = 360 / 7 := by
sorry

end NUMINAMATH_CALUDE_randys_trip_length_l2630_263018


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l2630_263048

/-- Given a polynomial function g(x) = px^4 + qx^3 + rx^2 + sx + t 
    where g(-1) = 1, prove that 16p - 8q + 4r - 2s + t = 1 -/
theorem polynomial_value_at_negative_one 
  (p q r s t : ℝ) 
  (g : ℝ → ℝ)
  (h1 : ∀ x, g x = p * x^4 + q * x^3 + r * x^2 + s * x + t)
  (h2 : g (-1) = 1) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l2630_263048


namespace NUMINAMATH_CALUDE_coprime_sum_not_divides_power_sum_l2630_263058

theorem coprime_sum_not_divides_power_sum
  (x y n : ℕ)
  (h_coprime : Nat.Coprime x y)
  (h_positive : 0 < x ∧ 0 < y)
  (h_not_one : x * y ≠ 1)
  (h_even : Even n)
  (h_pos : 0 < n) :
  ¬ (x + y ∣ x^n + y^n) :=
sorry

end NUMINAMATH_CALUDE_coprime_sum_not_divides_power_sum_l2630_263058


namespace NUMINAMATH_CALUDE_larger_square_perimeter_l2630_263063

theorem larger_square_perimeter
  (small_square_perimeter : ℝ)
  (shaded_area : ℝ)
  (h1 : small_square_perimeter = 72)
  (h2 : shaded_area = 160) :
  let small_side := small_square_perimeter / 4
  let small_area := small_side ^ 2
  let large_area := small_area + shaded_area
  let large_side := Real.sqrt large_area
  let large_perimeter := 4 * large_side
  large_perimeter = 88 := by
sorry

end NUMINAMATH_CALUDE_larger_square_perimeter_l2630_263063


namespace NUMINAMATH_CALUDE_angle_terminal_side_x_value_l2630_263076

theorem angle_terminal_side_x_value (x : ℝ) (θ : ℝ) :
  x < 0 →
  (∃ y : ℝ, y = 3 ∧ (x^2 + y^2).sqrt * Real.cos θ = x) →
  Real.cos θ = (Real.sqrt 10 / 10) * x →
  x = -1 :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_x_value_l2630_263076


namespace NUMINAMATH_CALUDE_somu_present_age_l2630_263020

/-- Somu's age -/
def somu_age : ℕ := sorry

/-- Somu's father's age -/
def father_age : ℕ := sorry

/-- Somu's age is one-third of his father's age -/
axiom current_age_ratio : somu_age = father_age / 3

/-- 5 years ago, Somu's age was one-fifth of his father's age -/
axiom past_age_ratio : somu_age - 5 = (father_age - 5) / 5

theorem somu_present_age : somu_age = 10 := by sorry

end NUMINAMATH_CALUDE_somu_present_age_l2630_263020


namespace NUMINAMATH_CALUDE_garden_tilling_time_l2630_263006

/-- Calculates the time required to till a rectangular plot -/
def tillingTime (width : ℕ) (length : ℕ) (swathWidth : ℕ) (tillRate : ℚ) : ℚ :=
  let rows := width / swathWidth
  let totalDistance := rows * length
  let totalSeconds := totalDistance * tillRate
  totalSeconds / 60

theorem garden_tilling_time :
  tillingTime 110 120 2 2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_garden_tilling_time_l2630_263006


namespace NUMINAMATH_CALUDE_surface_generates_solid_by_rotation_l2630_263029

/-- A right-angled triangle -/
structure RightTriangle where
  /-- The triangle has a right angle -/
  has_right_angle : Bool

/-- A cone -/
structure Cone where
  /-- The cone is formed by rotation -/
  formed_by_rotation : Bool

/-- Rotation of a triangle around one of its perpendicular sides -/
def rotate_triangle (t : RightTriangle) : Cone :=
  { formed_by_rotation := true }

/-- A theorem stating that rotating a right-angled triangle around one of its perpendicular sides
    demonstrates that a surface can generate a solid through rotation -/
theorem surface_generates_solid_by_rotation (t : RightTriangle) :
  ∃ (c : Cone), c = rotate_triangle t ∧ c.formed_by_rotation :=
by sorry

end NUMINAMATH_CALUDE_surface_generates_solid_by_rotation_l2630_263029


namespace NUMINAMATH_CALUDE_projectile_speed_proof_l2630_263090

/-- Proves that the speed of the first projectile is 445 km/h given the problem conditions -/
theorem projectile_speed_proof (v : ℝ) : 
  (v + 545) * (84 / 60) = 1386 → v = 445 := by
  sorry

end NUMINAMATH_CALUDE_projectile_speed_proof_l2630_263090


namespace NUMINAMATH_CALUDE_field_goal_percentage_l2630_263053

theorem field_goal_percentage (total_attempts : ℕ) (miss_ratio : ℚ) (wide_right : ℕ) : 
  total_attempts = 60 →
  miss_ratio = 1/4 →
  wide_right = 3 →
  (wide_right : ℚ) / (miss_ratio * total_attempts) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_field_goal_percentage_l2630_263053


namespace NUMINAMATH_CALUDE_solve_equation_l2630_263088

theorem solve_equation : ∃ x : ℝ, (2 * x + 5) / 7 = 15 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2630_263088


namespace NUMINAMATH_CALUDE_power_fraction_equality_l2630_263086

theorem power_fraction_equality : 
  (3^2015 - 3^2013 + 3^2011) / (3^2015 + 3^2013 - 3^2011) = 73/89 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l2630_263086


namespace NUMINAMATH_CALUDE_rhombuses_in_grid_of_25_l2630_263043

/-- Represents a triangular grid of equilateral triangles -/
structure TriangularGrid where
  side_length : ℕ
  total_triangles : ℕ

/-- Calculates the number of rhombuses in a triangular grid -/
def count_rhombuses (grid : TriangularGrid) : ℕ :=
  3 * (grid.side_length - 1) * grid.side_length

/-- Theorem: In a triangular grid with 25 triangles (5 per side), there are 30 rhombuses -/
theorem rhombuses_in_grid_of_25 :
  let grid : TriangularGrid := { side_length := 5, total_triangles := 25 }
  count_rhombuses grid = 30 := by
  sorry


end NUMINAMATH_CALUDE_rhombuses_in_grid_of_25_l2630_263043


namespace NUMINAMATH_CALUDE_min_value_product_squares_l2630_263070

theorem min_value_product_squares (a b c d e f g h i j k l m n o p : ℝ) 
  (h1 : a * b * c * d = 16)
  (h2 : e * f * g * h = 16)
  (h3 : i * j * k * l = 16)
  (h4 : m * n * o * p = 16) :
  (a * e * i * m)^2 + (b * f * j * n)^2 + (c * g * k * o)^2 + (d * h * l * p)^2 ≥ 1024 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_squares_l2630_263070


namespace NUMINAMATH_CALUDE_det_sum_of_matrices_l2630_263055

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -2; 3, 4]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 3; -1, 2]

theorem det_sum_of_matrices : Matrix.det (A + B) = 34 := by sorry

end NUMINAMATH_CALUDE_det_sum_of_matrices_l2630_263055


namespace NUMINAMATH_CALUDE_tan_sum_x_y_pi_third_l2630_263000

theorem tan_sum_x_y_pi_third (x y m : ℝ) 
  (hx : x^3 + Real.sin (2*x) = m)
  (hy : y^3 + Real.sin (2*y) = -m)
  (hx_bound : x > -π/4 ∧ x < π/4)
  (hy_bound : y > -π/4 ∧ y < π/4) :
  Real.tan (x + y + π/3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_x_y_pi_third_l2630_263000


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2630_263074

/-- Given two vectors a and b in ℝ², if they are parallel and a = (4,2) and b = (x,3), then x = 6. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![4, 2]
  let b : Fin 2 → ℝ := ![x, 3]
  (∃ (k : ℝ), b = k • a) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2630_263074


namespace NUMINAMATH_CALUDE_solve_for_a_l2630_263002

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | -5/3 < x ∧ x < 1/3}

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := |a * x - 2| < 3

-- Theorem statement
theorem solve_for_a : 
  (∀ x, x ∈ solution_set a ↔ inequality a x) → a = -3 :=
sorry

end NUMINAMATH_CALUDE_solve_for_a_l2630_263002


namespace NUMINAMATH_CALUDE_min_sum_cube_relation_l2630_263004

theorem min_sum_cube_relation (m n : ℕ+) (h : 90 * m.val = n.val ^ 3) : 
  (∀ (x y : ℕ+), 90 * x.val = y.val ^ 3 → m.val + n.val ≤ x.val + y.val) → 
  m.val + n.val = 330 := by
sorry

end NUMINAMATH_CALUDE_min_sum_cube_relation_l2630_263004


namespace NUMINAMATH_CALUDE_energy_drink_cost_l2630_263073

theorem energy_drink_cost (cupcakes : Nat) (cupcake_price : ℚ) 
  (cookies : Nat) (cookie_price : ℚ) (basketballs : Nat) 
  (basketball_price : ℚ) (energy_drinks : Nat) :
  cupcakes = 50 →
  cupcake_price = 2 →
  cookies = 40 →
  cookie_price = 1/2 →
  basketballs = 2 →
  basketball_price = 40 →
  energy_drinks = 20 →
  (cupcakes * cupcake_price + cookies * cookie_price - basketballs * basketball_price) / energy_drinks = 2 := by
sorry


end NUMINAMATH_CALUDE_energy_drink_cost_l2630_263073


namespace NUMINAMATH_CALUDE_triangle_existence_and_area_l2630_263087

theorem triangle_existence_and_area 
  (a b c : ℝ) 
  (h : |a - Real.sqrt 8| + Real.sqrt (b^2 - 5) + (c - Real.sqrt 3)^2 = 0) : 
  ∃ (s : ℝ), s = (a + b + c) / 2 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = Real.sqrt 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_and_area_l2630_263087


namespace NUMINAMATH_CALUDE_smallest_multiple_l2630_263040

theorem smallest_multiple : ∃ (a : ℕ), 
  (a % 3 = 0) ∧ 
  ((a - 1) % 4 = 0) ∧ 
  ((a - 2) % 5 = 0) ∧ 
  (∀ b : ℕ, b < a → ¬((b % 3 = 0) ∧ ((b - 1) % 4 = 0) ∧ ((b - 2) % 5 = 0))) ∧
  a = 57 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2630_263040


namespace NUMINAMATH_CALUDE_tournament_committee_count_l2630_263081

/-- The number of teams in the frisbee league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members selected from the host team for the committee -/
def host_committee_size : ℕ := 4

/-- The number of members selected from each non-host team for the committee -/
def non_host_committee_size : ℕ := 2

/-- The total number of members in the tournament committee -/
def total_committee_size : ℕ := 12

/-- Theorem stating the total number of possible tournament committees -/
theorem tournament_committee_count :
  (num_teams : ℕ) * (Nat.choose team_size host_committee_size) *
  (Nat.choose team_size non_host_committee_size ^ (num_teams - 1)) = 340342925 := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l2630_263081
