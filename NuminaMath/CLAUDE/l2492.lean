import Mathlib

namespace NUMINAMATH_CALUDE_work_completion_time_l2492_249244

/-- The number of days it takes for A to finish the work alone -/
def days_A : ℝ := 22.5

/-- The number of days it takes for B to finish the work alone -/
def days_B : ℝ := 15

/-- The total wage when A and B work together -/
def total_wage : ℝ := 3400

/-- A's wage when working together with B -/
def wage_A : ℝ := 2040

theorem work_completion_time :
  days_B = 15 ∧ 
  wage_A / total_wage = 2040 / 3400 →
  days_A = 22.5 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2492_249244


namespace NUMINAMATH_CALUDE_remainder_sum_l2492_249285

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) :
  (n % 2) + (n % 3) + (n % 9) = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l2492_249285


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l2492_249231

theorem triangle_trigonometric_identities (A B C : ℝ) 
  (h : A + B + C = π) : 
  (Real.sin A + Real.sin B + Real.sin C = 
    4 * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2)) ∧
  (Real.tan A + Real.tan B + Real.tan C = 
    Real.tan A * Real.tan B * Real.tan C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l2492_249231


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2492_249211

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 - 4 * p^3 + 3 * p + 2) + (-3 * p^4 + 2 * p^3 - 7 * p^2 + 8) =
  2 * p^4 - 2 * p^3 - 7 * p^2 + 3 * p + 10 := by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2492_249211


namespace NUMINAMATH_CALUDE_ellipse_constant_expression_l2492_249265

/-- Ellipse with semi-major axis √5 and semi-minor axis 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 = 1}

/-- Foci of the ellipse -/
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

/-- A line passing through F₁ -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 + 2)}

/-- Dot product of two 2D vectors -/
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The origin point -/
def O : ℝ × ℝ := (0, 0)

theorem ellipse_constant_expression (M N : ℝ × ℝ) (k : ℝ) 
    (hM : M ∈ Ellipse ∩ Line k) (hN : N ∈ Ellipse ∩ Line k) : 
    dot (M - O) (N - O) - 11 * dot (M - F₁) (N - F₁) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_constant_expression_l2492_249265


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l2492_249276

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 9

-- Define the bisecting line m
def line_m (x y : ℝ) : Prop :=
  2 * x - y = 4

-- Define the intersecting line
def intersecting_line (x y a : ℝ) : Prop :=
  x - y + a = 0

-- Define the perpendicularity condition
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem circle_intersection_theorem :
  ∃ (a : ℝ), a = -4 ∨ a = 1 ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    intersecting_line x₁ y₁ a ∧ intersecting_line x₂ y₂ a ∧
    perpendicular_condition x₁ y₁ x₂ y₂) ∧
  circle_C 1 1 ∧ circle_C (-2) (-2) ∧
  (∀ (x y : ℝ), circle_C x y → line_m x y) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l2492_249276


namespace NUMINAMATH_CALUDE_base_10_300_equals_base_6_1220_l2492_249235

/-- Converts a list of digits in a given base to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

/-- Theorem stating that 300 in base 10 is equal to 1220 in base 6 -/
theorem base_10_300_equals_base_6_1220 : 
  300 = to_decimal [0, 2, 2, 1] 6 := by
  sorry

end NUMINAMATH_CALUDE_base_10_300_equals_base_6_1220_l2492_249235


namespace NUMINAMATH_CALUDE_berry_swap_difference_l2492_249254

/-- The number of blueberries in each blue box -/
def blueberries_per_box : ℕ := 20

/-- The increase in total berries when swapping one blue box for one red box -/
def berry_increase : ℕ := 10

/-- The number of strawberries in each red box -/
def strawberries_per_box : ℕ := blueberries_per_box + berry_increase

/-- The change in the difference between total strawberries and total blueberries -/
def difference_change : ℕ := strawberries_per_box + blueberries_per_box

theorem berry_swap_difference :
  difference_change = 50 :=
sorry

end NUMINAMATH_CALUDE_berry_swap_difference_l2492_249254


namespace NUMINAMATH_CALUDE_swim_club_members_l2492_249257

theorem swim_club_members :
  ∀ (total_members : ℕ) 
    (passed_test : ℕ) 
    (not_passed_with_course : ℕ) 
    (not_passed_without_course : ℕ),
  passed_test = (30 * total_members) / 100 →
  not_passed_with_course = 5 →
  not_passed_without_course = 30 →
  total_members = passed_test + not_passed_with_course + not_passed_without_course →
  total_members = 50 := by
sorry

end NUMINAMATH_CALUDE_swim_club_members_l2492_249257


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2492_249260

/-- Theorem: For a rhombus with area 150 cm² and one diagonal of 30 cm, the other diagonal is 10 cm -/
theorem rhombus_diagonal (area : ℝ) (d2 : ℝ) (d1 : ℝ) :
  area = 150 ∧ d2 = 30 ∧ area = (d1 * d2) / 2 → d1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2492_249260


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l2492_249207

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 2 < b ∧ b < 4) :
  -4 < a - b ∧ a - b < -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l2492_249207


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2492_249219

theorem least_addition_for_divisibility :
  ∃ (x : ℕ), x = 4 ∧ 
  (28 ∣ (1056 + x)) ∧ 
  (∀ (y : ℕ), y < x → ¬(28 ∣ (1056 + y))) :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2492_249219


namespace NUMINAMATH_CALUDE_product_equals_3700_l2492_249290

theorem product_equals_3700 : 4 * 37 * 25 = 3700 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_3700_l2492_249290


namespace NUMINAMATH_CALUDE_track_length_not_approximately_200mm_l2492_249249

/-- Represents the length of a school's track and field in millimeters -/
def track_length : ℝ := 200000 -- Assuming 200 meters = 200000 mm

/-- Represents a reasonable range for "approximately 200 mm" -/
def approximate_range : Set ℝ := {x | 190 ≤ x ∧ x ≤ 210}

theorem track_length_not_approximately_200mm : 
  track_length ∉ approximate_range := by sorry

end NUMINAMATH_CALUDE_track_length_not_approximately_200mm_l2492_249249


namespace NUMINAMATH_CALUDE_converse_inequality_l2492_249266

theorem converse_inequality (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_converse_inequality_l2492_249266


namespace NUMINAMATH_CALUDE_automobile_repair_cost_l2492_249218

/-- The cost of fixing Leila's automobile given her supermarket expenses and total spending -/
def cost_to_fix_automobile (supermarket_expense : ℝ) (total_spent : ℝ) : ℝ :=
  3 * supermarket_expense + 50

/-- Theorem: Given the conditions, the cost to fix Leila's automobile is $350 -/
theorem automobile_repair_cost :
  ∃ (supermarket_expense : ℝ),
    cost_to_fix_automobile supermarket_expense 450 + supermarket_expense = 450 ∧
    cost_to_fix_automobile supermarket_expense 450 = 350 := by
  sorry

end NUMINAMATH_CALUDE_automobile_repair_cost_l2492_249218


namespace NUMINAMATH_CALUDE_prime_before_non_prime_probability_l2492_249268

def prime_numbers : List ℕ := [2, 3, 5, 7, 11]
def non_prime_numbers : List ℕ := [1, 4, 6, 8, 9, 10, 12]

def total_numbers : ℕ := prime_numbers.length + non_prime_numbers.length

theorem prime_before_non_prime_probability :
  let favorable_permutations := (prime_numbers.length.factorial * non_prime_numbers.length.factorial : ℚ)
  let total_permutations := total_numbers.factorial
  (favorable_permutations / total_permutations : ℚ) = 1 / 792 := by
  sorry

end NUMINAMATH_CALUDE_prime_before_non_prime_probability_l2492_249268


namespace NUMINAMATH_CALUDE_hana_stamp_collection_l2492_249220

/-- Represents the fraction of Hana's stamp collection that was sold -/
def fraction_sold : ℚ := 28 / 49

/-- The amount Hana received for the part of the collection she sold -/
def amount_received : ℕ := 28

/-- The total value of Hana's entire stamp collection -/
def total_value : ℕ := 49

theorem hana_stamp_collection :
  fraction_sold = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_hana_stamp_collection_l2492_249220


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2492_249264

theorem inequality_and_equality_condition (a b n : ℕ+) (h1 : a > b) (h2 : a * b - 1 = n ^ 2) :
  (a : ℝ) - b ≥ Real.sqrt (4 * n - 3) ∧
  (∃ m : ℕ, n = m ^ 2 + m + 1 ∧ a = (m + 1) ^ 2 + 1 ∧ b = m ^ 2 + 1 ↔ (a : ℝ) - b = Real.sqrt (4 * n - 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2492_249264


namespace NUMINAMATH_CALUDE_oyster_feast_l2492_249232

/-- The number of oysters Squido eats -/
def squido_oysters : ℕ := 200

/-- Crabby eats at least twice as many oysters as Squido -/
def crabby_oysters_condition (c : ℕ) : Prop := c ≥ 2 * squido_oysters

/-- The total number of oysters eaten by Squido and Crabby -/
def total_oysters (c : ℕ) : ℕ := squido_oysters + c

theorem oyster_feast (c : ℕ) (h : crabby_oysters_condition c) : 
  total_oysters c ≥ 600 := by
  sorry

end NUMINAMATH_CALUDE_oyster_feast_l2492_249232


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2492_249247

theorem average_speed_calculation (total_distance : ℝ) (first_half_distance : ℝ) (second_half_distance : ℝ) 
  (first_half_speed : ℝ) (second_half_speed : ℝ) 
  (h1 : total_distance = 60)
  (h2 : first_half_distance = 30)
  (h3 : second_half_distance = 30)
  (h4 : first_half_speed = 48)
  (h5 : second_half_speed = 24)
  (h6 : total_distance = first_half_distance + second_half_distance) :
  (total_distance / (first_half_distance / first_half_speed + second_half_distance / second_half_speed)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2492_249247


namespace NUMINAMATH_CALUDE_equation_solutions_l2492_249277

theorem equation_solutions : 
  {x : ℝ | (47 - 2*x)^(1/4) + (35 + 2*x)^(1/4) = 4} = {23, -17} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2492_249277


namespace NUMINAMATH_CALUDE_problem_solution_l2492_249263

theorem problem_solution (x y : ℝ) : (x - 2)^2 + |y + 1/3| = 0 → y^x = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2492_249263


namespace NUMINAMATH_CALUDE_worker_y_fraction_l2492_249245

theorem worker_y_fraction (total : ℝ) (x y : ℝ) 
  (h1 : x + y = total) 
  (h2 : 0.005 * x + 0.008 * y = 0.0065 * total) 
  (h3 : total > 0) :
  y / total = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_worker_y_fraction_l2492_249245


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_equals_circumference_l2492_249281

/-- Represents a circle with a given circumference -/
structure Circle where
  circumference : ℝ

/-- Represents a configuration of four identical circles arranged in a straight line -/
structure CircleConfiguration where
  circle : Circle
  num_circles : Nat
  are_tangent : Bool
  are_identical : Bool
  are_in_line : Bool

/-- Calculates the perimeter of the shaded region between the first and last circle -/
def shaded_region_perimeter (config : CircleConfiguration) : ℝ :=
  config.circle.circumference

/-- Theorem stating that the perimeter of the shaded region is equal to the circumference of one circle -/
theorem shaded_region_perimeter_equals_circumference 
  (config : CircleConfiguration) 
  (h1 : config.num_circles = 4) 
  (h2 : config.are_tangent) 
  (h3 : config.are_identical) 
  (h4 : config.are_in_line) 
  (h5 : config.circle.circumference = 24) :
  shaded_region_perimeter config = 24 := by
  sorry

#check shaded_region_perimeter_equals_circumference

end NUMINAMATH_CALUDE_shaded_region_perimeter_equals_circumference_l2492_249281


namespace NUMINAMATH_CALUDE_tetrahedron_unique_large_angle_sum_l2492_249270

/-- A tetrahedron is a structure with four vertices and six edges. -/
structure Tetrahedron :=
  (A B C D : Point)

/-- The plane angle between two edges at a vertex of a tetrahedron. -/
def planeAngle (t : Tetrahedron) (v1 v2 v3 : Point) : ℝ := sorry

/-- The property that the sum of any two plane angles at a vertex is greater than 180°. -/
def hasLargeAngleSum (t : Tetrahedron) (v : Point) : Prop :=
  ∀ (v1 v2 v3 : Point), v1 ≠ v2 → v1 ≠ v3 → v2 ≠ v3 →
    planeAngle t v v1 v2 + planeAngle t v v1 v3 > 180

/-- Theorem: No more than one vertex of a tetrahedron can have the large angle sum property. -/
theorem tetrahedron_unique_large_angle_sum (t : Tetrahedron) :
  ¬∃ (v1 v2 : Point), v1 ≠ v2 ∧ hasLargeAngleSum t v1 ∧ hasLargeAngleSum t v2 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_unique_large_angle_sum_l2492_249270


namespace NUMINAMATH_CALUDE_tims_soda_cans_l2492_249217

theorem tims_soda_cans (S : ℕ) : 
  (S - 10) + (S - 10) / 2 + 10 = 34 → S = 26 :=
by sorry

end NUMINAMATH_CALUDE_tims_soda_cans_l2492_249217


namespace NUMINAMATH_CALUDE_triangle_circles_area_sum_l2492_249293

/-- Represents a right triangle with circles centered at its vertices -/
structure TriangleWithCircles where
  /-- The length of the shortest side of the triangle -/
  a : ℝ
  /-- The length of the middle side of the triangle -/
  b : ℝ
  /-- The length of the hypotenuse of the triangle -/
  c : ℝ
  /-- The radius of the circle centered at the vertex opposite to side a -/
  r : ℝ
  /-- The radius of the circle centered at the vertex opposite to side b -/
  s : ℝ
  /-- The radius of the circle centered at the vertex opposite to side c -/
  t : ℝ
  /-- The triangle is a right triangle -/
  right_triangle : a^2 + b^2 = c^2
  /-- The circles are mutually externally tangent -/
  tangent_circles : r + s = a ∧ r + t = b ∧ s + t = c

/-- The theorem stating that for a 6-8-10 right triangle with mutually externally tangent 
    circles centered at its vertices, the sum of the areas of these circles is 56π -/
theorem triangle_circles_area_sum (triangle : TriangleWithCircles) 
    (h1 : triangle.a = 6) (h2 : triangle.b = 8) (h3 : triangle.c = 10) : 
    π * (triangle.r^2 + triangle.s^2 + triangle.t^2) = 56 * π :=
  sorry

end NUMINAMATH_CALUDE_triangle_circles_area_sum_l2492_249293


namespace NUMINAMATH_CALUDE_equation_solution_l2492_249221

theorem equation_solution : 
  let f (x : ℝ) := (x^3 + x^2 + x + 1) / (x + 1)
  let g (x : ℝ) := x^2 + 4*x + 4
  ∀ x : ℝ, f x = g x ↔ x = -3/4 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2492_249221


namespace NUMINAMATH_CALUDE_rectangle_circle_tangent_l2492_249212

theorem rectangle_circle_tangent (r : ℝ) (h1 : r = 3) : 
  let circle_area := π * r^2
  let rectangle_area := 3 * circle_area
  let short_side := 2 * r
  let long_side := rectangle_area / short_side
  long_side = 4.5 * π := by sorry

end NUMINAMATH_CALUDE_rectangle_circle_tangent_l2492_249212


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l2492_249201

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N + 2) * (N + 1)) / Nat.factorial (N + 3) = (N + 1) / (N + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l2492_249201


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2492_249202

-- Define the quadratic function f
def f (x : ℝ) : ℝ := sorry

-- Define the conditions for f
axiom f_zero : f 0 = 0
axiom f_recurrence (x : ℝ) : f (x + 1) = f x + x + 1

-- Define the minimum value function g
def g (t : ℝ) : ℝ := sorry

-- Theorem to prove
theorem quadratic_function_properties :
  -- Part 1: Expression for f(x)
  (∀ x, f x = (1/2) * x^2 + (1/2) * x) ∧
  -- Part 2: Expression for g(t)
  (∀ t, g t = if t ≤ -3/2 then (1/2) * t^2 + (3/2) * t + 1
              else if t < -1/2 then -1/8
              else (1/2) * t^2 + (1/2) * t) ∧
  -- Part 3: Range of m
  (∀ m, (∀ t, g t + m ≥ 0) ↔ m ≥ 1/8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2492_249202


namespace NUMINAMATH_CALUDE_james_hourly_wage_l2492_249216

theorem james_hourly_wage (main_wage : ℝ) (second_wage : ℝ) (main_hours : ℝ) (second_hours : ℝ) (total_earnings : ℝ) :
  second_wage = 0.8 * main_wage →
  main_hours = 30 →
  second_hours = main_hours / 2 →
  total_earnings = main_wage * main_hours + second_wage * second_hours →
  total_earnings = 840 →
  main_wage = 20 := by
sorry

end NUMINAMATH_CALUDE_james_hourly_wage_l2492_249216


namespace NUMINAMATH_CALUDE_odd_function_property_l2492_249210

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def has_max_on (f : ℝ → ℝ) (a b M : ℝ) : Prop :=
  (∀ x, a ≤ x → x ≤ b → f x ≤ M) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = M)

def has_min_on (f : ℝ → ℝ) (a b m : ℝ) : Prop :=
  (∀ x, a ≤ x → x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem odd_function_property (f : ℝ → ℝ) :
  is_odd f →
  increasing_on f 3 6 →
  has_max_on f 3 6 2 →
  has_min_on f 3 6 (-1) →
  2 * f (-6) + f (-3) = -3 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_property_l2492_249210


namespace NUMINAMATH_CALUDE_arithmetic_sequence_26th_term_l2492_249213

/-- Given an arithmetic sequence with first term 3 and second term 13, 
    the 26th term is 253. -/
theorem arithmetic_sequence_26th_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
    a 0 = 3 →                            -- first term is 3
    a 1 = 13 →                           -- second term is 13
    a 25 = 253 :=                        -- 26th term (index 25) is 253
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_26th_term_l2492_249213


namespace NUMINAMATH_CALUDE_prime_sum_100_l2492_249226

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the sum of a list of natural numbers -/
def listSum (l : List ℕ) : ℕ := sorry

theorem prime_sum_100 :
  ∃ (l : List ℕ), 
    (∀ x ∈ l, isPrime x) ∧ 
    (listSum l = 100) ∧ 
    (l.length = 9) ∧
    (∀ (m : List ℕ), (∀ y ∈ m, isPrime y) → (listSum m = 100) → m.length ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_prime_sum_100_l2492_249226


namespace NUMINAMATH_CALUDE_mountain_climb_time_l2492_249284

/-- Represents a climber with ascending and descending speeds -/
structure Climber where
  ascendSpeed : ℝ
  descendSpeed : ℝ

/-- The mountain climbing scenario -/
structure MountainClimb where
  a : Climber
  b : Climber
  mountainHeight : ℝ
  meetingDistance : ℝ
  meetingTime : ℝ

theorem mountain_climb_time (mc : MountainClimb) : 
  mc.a.descendSpeed = 1.5 * mc.a.ascendSpeed →
  mc.b.descendSpeed = 1.5 * mc.b.ascendSpeed →
  mc.a.ascendSpeed > mc.b.ascendSpeed →
  mc.meetingTime = 1 →
  mc.meetingDistance = 600 →
  (mc.mountainHeight / mc.a.ascendSpeed + mc.mountainHeight / mc.a.descendSpeed = 1.5) :=
by sorry

end NUMINAMATH_CALUDE_mountain_climb_time_l2492_249284


namespace NUMINAMATH_CALUDE_male_contestants_l2492_249294

theorem male_contestants (total : ℕ) (female_ratio : ℚ) (h1 : total = 18) (h2 : female_ratio = 1/3) :
  (1 - female_ratio) * total = 12 := by
  sorry

end NUMINAMATH_CALUDE_male_contestants_l2492_249294


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2492_249258

/-- A point in the second quadrant with specific properties has coordinates (-2, 1) -/
theorem point_in_second_quadrant (P : ℝ × ℝ) :
  (P.1 < 0 ∧ P.2 > 0) →  -- Second quadrant condition
  (abs P.1 = 2) →        -- |x| = 2 condition
  (P.2^2 = 1) →          -- y is square root of 1 condition
  P = (-2, 1) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2492_249258


namespace NUMINAMATH_CALUDE_interior_angle_sum_difference_l2492_249298

/-- The sum of interior angles of an n-sided polygon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The difference in sum of interior angles between an (n+1)-sided polygon and an n-sided polygon is 180° -/
theorem interior_angle_sum_difference (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles (n + 1) - sum_interior_angles n = 180 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_difference_l2492_249298


namespace NUMINAMATH_CALUDE_no_k_with_prime_roots_l2492_249253

/-- A quadratic equation x^2 - 65x + k = 0 with prime roots -/
def has_prime_roots (k : ℤ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ 
  (p : ℤ) + (q : ℤ) = 65 ∧ (p : ℤ) * (q : ℤ) = k

/-- There are no integer values of k for which the quadratic equation has prime roots -/
theorem no_k_with_prime_roots : ¬∃ k : ℤ, has_prime_roots k := by
  sorry

end NUMINAMATH_CALUDE_no_k_with_prime_roots_l2492_249253


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2492_249205

/-- Given r and s are solutions to the equation 3x^2 - 5x + 2 = 0,
    prove that (9r^3 - 9s^3)(r - s)^{-1} = 19 -/
theorem cubic_equation_solution (r s : ℝ) 
  (h1 : 3 * r^2 - 5 * r + 2 = 0)
  (h2 : 3 * s^2 - 5 * s + 2 = 0)
  (h3 : r ≠ s) : 
  (9 * r^3 - 9 * s^3) / (r - s) = 19 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2492_249205


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2492_249282

theorem negative_fraction_comparison : -1/3 > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2492_249282


namespace NUMINAMATH_CALUDE_expression_simplification_l2492_249209

theorem expression_simplification :
  (((3 + 5 + 6 - 2) * 2) / 4) + ((3 * 4 + 6 - 4) / 3) = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2492_249209


namespace NUMINAMATH_CALUDE_power_of_power_three_l2492_249246

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l2492_249246


namespace NUMINAMATH_CALUDE_tangent_slope_minimum_value_l2492_249242

theorem tangent_slope_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 2) :
  (8 * a + b) / (a * b) ≥ 9 ∧
  ((8 * a + b) / (a * b) = 9 ↔ a = 1/3 ∧ b = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_minimum_value_l2492_249242


namespace NUMINAMATH_CALUDE_bake_sale_group_composition_l2492_249295

theorem bake_sale_group_composition (total : ℕ) (boys : ℕ) : 
  (boys : ℚ) / total = 35 / 100 →
  ((boys - 3 : ℚ) / total) = 40 / 100 →
  boys = 21 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_group_composition_l2492_249295


namespace NUMINAMATH_CALUDE_product_101_squared_l2492_249233

theorem product_101_squared : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_product_101_squared_l2492_249233


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l2492_249261

def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l2492_249261


namespace NUMINAMATH_CALUDE_min_value_theorem_l2492_249299

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 3) :
  (a^2 + b^2 + 22) / (a + b) ≥ 8 ∧ ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' * b' = 3 ∧ (a'^2 + b'^2 + 22) / (a' + b') = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2492_249299


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l2492_249200

/-- Calculates the corrected mean of a set of observations after fixing an error -/
def corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * initial_mean - incorrect_value + correct_value) / n

/-- Theorem stating the corrected mean for the given problem -/
theorem corrected_mean_problem :
  let n : ℕ := 50
  let initial_mean : ℚ := 36
  let incorrect_value : ℚ := 23
  let correct_value : ℚ := 60
  corrected_mean n initial_mean incorrect_value correct_value = 36.74 := by
sorry

#eval corrected_mean 50 36 23 60

end NUMINAMATH_CALUDE_corrected_mean_problem_l2492_249200


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l2492_249286

theorem cubic_root_ratio (a b c d : ℝ) (h : a ≠ 0) :
  (∃ x y z : ℝ, x = 1 ∧ y = (1/2 : ℝ) ∧ z = 4 ∧
    ∀ t : ℝ, a * t^3 + b * t^2 + c * t + d = 0 ↔ t = x ∨ t = y ∨ t = z) →
  c / d = -(13/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l2492_249286


namespace NUMINAMATH_CALUDE_triangle_angle_contradiction_l2492_249238

theorem triangle_angle_contradiction (α β γ : ℝ) : 
  (α > 60 ∧ β > 60 ∧ γ > 60) → 
  (α + β + γ = 180) → 
  False :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_contradiction_l2492_249238


namespace NUMINAMATH_CALUDE_y_derivative_l2492_249273

noncomputable def y (x : ℝ) : ℝ := 3 * Real.arcsin (3 / (4 * x + 1)) + 2 * Real.sqrt (4 * x^2 + 2 * x - 2)

theorem y_derivative (x : ℝ) (h : 4 * x + 1 > 0) :
  deriv y x = (7 * (4 * x + 1)) / (2 * Real.sqrt (4 * x^2 + 2 * x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_y_derivative_l2492_249273


namespace NUMINAMATH_CALUDE_symmetry_of_A_and_D_l2492_249280

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x - 2 * y - 4 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define P as the midpoint of AB
def P_midpoint (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = -1/2

-- Define Q on line l
def Q_on_l : Prop := line_l 4 0

-- Define A between B and Q
def A_between_B_Q (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ A.1 = t * B.1 + (1 - t) * 4 ∧ A.2 = t * B.2

-- Define the right focus F
def right_focus (F : ℝ × ℝ) : Prop := F.1 = 1 ∧ F.2 = 0

-- Define D as the intersection of BF and C
def D_intersection (B D F : ℝ × ℝ) : Prop :=
  ellipse_C D.1 D.2 ∧ ∃ t : ℝ, D.1 = B.1 + t * (F.1 - B.1) ∧ D.2 = B.2 + t * (F.2 - B.2)

-- Define symmetry with respect to x-axis
def symmetric_x_axis (A D : ℝ × ℝ) : Prop := A.1 = D.1 ∧ A.2 = -D.2

-- Main theorem
theorem symmetry_of_A_and_D (A B D F : ℝ × ℝ) :
  intersection_points A B →
  P_midpoint A B →
  Q_on_l →
  A_between_B_Q A B →
  right_focus F →
  D_intersection B D F →
  symmetric_x_axis A D :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_A_and_D_l2492_249280


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficient_l2492_249223

/-- Given a polynomial Q(x) = x^3 + 3x^2 + dx + 15 where (x - 3) is a factor,
    prove that the coefficient d equals -23. -/
theorem polynomial_factor_coefficient (d : ℝ) : 
  (∀ x, x^3 + 3*x^2 + d*x + 15 = (x - 3) * (x^2 + (3 + 3)*x + (d + 9 + 3*3))) → 
  d = -23 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficient_l2492_249223


namespace NUMINAMATH_CALUDE_compare_expressions_l2492_249225

theorem compare_expressions (x : ℝ) : x^2 - x > x - 2 := by sorry

end NUMINAMATH_CALUDE_compare_expressions_l2492_249225


namespace NUMINAMATH_CALUDE_solution_set_equals_interval_l2492_249251

-- Define the solution set of |x-3| < 5
def solution_set : Set ℝ := {x : ℝ | |x - 3| < 5}

-- State the theorem
theorem solution_set_equals_interval : solution_set = Set.Ioo (-2) 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equals_interval_l2492_249251


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l2492_249275

theorem min_value_sum_of_squares (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l2492_249275


namespace NUMINAMATH_CALUDE_unique_solution_system_l2492_249271

theorem unique_solution_system (x y z : ℝ) : 
  x^3 = 3*x - 12*y + 50 ∧ 
  y^3 = 12*y + 3*z - 2 ∧ 
  z^3 = 27*z + 27*x → 
  x = 2 ∧ y = 4 ∧ z = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2492_249271


namespace NUMINAMATH_CALUDE_problem_solving_probability_l2492_249272

theorem problem_solving_probability (p_xavier p_yvonne p_zelda : ℝ) 
  (h1 : p_xavier = 1/5)
  (h2 : p_yvonne = 1/2)
  (h3 : p_xavier * p_yvonne * (1 - p_zelda) = 0.0375) :
  p_zelda = 0.625 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l2492_249272


namespace NUMINAMATH_CALUDE_program_count_proof_l2492_249262

/-- The number of thirty-minute programs in a television schedule where
    one-fourth of the airing time is spent on commercials and
    45 minutes are spent on commercials for the whole duration of these programs. -/
def number_of_programs : ℕ := 6

/-- The duration of each program in minutes. -/
def program_duration : ℕ := 30

/-- The fraction of airing time spent on commercials. -/
def commercial_fraction : ℚ := 1/4

/-- The total time spent on commercials for all programs in minutes. -/
def total_commercial_time : ℕ := 45

theorem program_count_proof :
  number_of_programs = total_commercial_time / (commercial_fraction * program_duration) :=
by sorry

end NUMINAMATH_CALUDE_program_count_proof_l2492_249262


namespace NUMINAMATH_CALUDE_min_value_of_f_l2492_249289

/-- The quadratic function f(x) = x^2 - 2x - 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- The minimum value of f(x) = x^2 - 2x - 1 for x ∈ ℝ is -2 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2492_249289


namespace NUMINAMATH_CALUDE_set_A_equals_one_two_l2492_249279

def A : Set ℕ := {x | x^2 - 3*x < 0 ∧ x > 0}

theorem set_A_equals_one_two : A = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_A_equals_one_two_l2492_249279


namespace NUMINAMATH_CALUDE_cos_sum_equals_one_l2492_249206

theorem cos_sum_equals_one (x : ℝ) (h : Real.cos (x - Real.pi / 6) = Real.sqrt 3 / 3) :
  Real.cos x + Real.cos (x - Real.pi / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_equals_one_l2492_249206


namespace NUMINAMATH_CALUDE_superhero_speed_in_mph_l2492_249227

-- Define the superhero's speed in kilometers per minute
def speed_km_per_min : ℝ := 1000

-- Define the conversion factor from km to miles
def km_to_miles : ℝ := 0.6

-- Define the number of minutes in an hour
def minutes_per_hour : ℝ := 60

-- Theorem statement
theorem superhero_speed_in_mph : 
  speed_km_per_min * km_to_miles * minutes_per_hour = 36000 := by
  sorry

end NUMINAMATH_CALUDE_superhero_speed_in_mph_l2492_249227


namespace NUMINAMATH_CALUDE_same_point_on_bisector_l2492_249248

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The angle bisector of the first and third quadrants -/
def firstThirdQuadrantBisector : Set Point2D :=
  {p : Point2D | p.x = p.y}

/-- Theorem: If A(a, b) and B(b, a) represent the same point, 
    then this point lies on the angle bisector of the first and third quadrants -/
theorem same_point_on_bisector (a b : ℝ) :
  Point2D.mk a b = Point2D.mk b a → 
  Point2D.mk a b ∈ firstThirdQuadrantBisector := by
  sorry

end NUMINAMATH_CALUDE_same_point_on_bisector_l2492_249248


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2492_249296

/-- A quadratic function with leading coefficient a -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The solution set of f(x) > -2x is (1,3) -/
def solution_set (a b c : ℝ) : Prop :=
  ∀ x, (1 < x ∧ x < 3) ↔ f a b c x > -2 * x

/-- The equation f(x) + 6a = 0 has two equal real roots -/
def equal_roots (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x, f a b c x + 6 * a = 0 ↔ x = r

theorem quadratic_function_theorem (a b c : ℝ) 
  (h1 : solution_set a b c)
  (h2 : equal_roots a b c)
  (h3 : a < 0) :
  ∀ x, f a b c x = -x^2 - x - 3/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2492_249296


namespace NUMINAMATH_CALUDE_largest_solution_and_fraction_l2492_249256

theorem largest_solution_and_fraction (x : ℝ) :
  (7 * x) / 4 + 2 = 8 / x →
  ∃ (a b c d : ℤ),
    x = (a + b * Real.sqrt c) / d ∧
    a = -4 ∧ b = 8 ∧ c = 15 ∧ d = 7 ∧
    x ≤ (-4 + 8 * Real.sqrt 15) / 7 ∧
    (a * c * d : ℚ) / b = -105/2 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_and_fraction_l2492_249256


namespace NUMINAMATH_CALUDE_eleven_divides_six_digit_repeat_l2492_249228

/-- A six-digit positive integer where the first three digits are the same as the last three digits in the same order -/
def SixDigitRepeat (z : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    z = 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c

theorem eleven_divides_six_digit_repeat (z : ℕ) (h : SixDigitRepeat z) : 
  11 ∣ z := by
  sorry

end NUMINAMATH_CALUDE_eleven_divides_six_digit_repeat_l2492_249228


namespace NUMINAMATH_CALUDE_pencil_price_l2492_249297

theorem pencil_price (total_cost : ℝ) (num_pens num_pencils : ℕ) (avg_pen_price : ℝ) :
  total_cost = 690 →
  num_pens = 30 →
  num_pencils = 75 →
  avg_pen_price = 18 →
  (total_cost - num_pens * avg_pen_price) / num_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_l2492_249297


namespace NUMINAMATH_CALUDE_stamp_collection_theorem_l2492_249230

/-- The face value of Xiaoming's stamps in jiao -/
def xiaoming_stamp_value : ℕ := 16

/-- The face value of Xiaoliang's stamps in jiao -/
def xiaoliang_stamp_value : ℕ := 2

/-- The number of stamps Xiaoming exchanges -/
def xiaoming_exchange_count : ℕ := 2

/-- The ratio of Xiaoliang's stamps to Xiaoming's before exchange -/
def pre_exchange_ratio : ℕ := 5

/-- The ratio of Xiaoliang's stamps to Xiaoming's after exchange -/
def post_exchange_ratio : ℕ := 3

/-- The total number of stamps Xiaoming and Xiaoliang have -/
def total_stamps : ℕ := 168

theorem stamp_collection_theorem :
  let xiaoming_initial := xiaoming_exchange_count * xiaoming_stamp_value / xiaoliang_stamp_value
  let xiaoming_final := xiaoming_initial + xiaoming_exchange_count * xiaoming_stamp_value / xiaoliang_stamp_value - xiaoming_exchange_count
  let xiaoliang_initial := pre_exchange_ratio * xiaoming_initial
  let xiaoliang_final := xiaoliang_initial - xiaoming_exchange_count * xiaoming_stamp_value / xiaoliang_stamp_value + xiaoming_exchange_count
  (xiaoliang_final = post_exchange_ratio * xiaoming_final) →
  (xiaoming_initial + xiaoliang_initial = total_stamps) := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_theorem_l2492_249230


namespace NUMINAMATH_CALUDE_inscribed_tangent_circle_exists_l2492_249287

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle -/
structure Angle where
  vertex : Point
  side1 : Point
  side2 : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a circle is inscribed in an angle -/
def isInscribed (c : Circle) (a : Angle) : Prop := sorry

/-- Predicate to check if two circles are tangent -/
def isTangent (c1 c2 : Circle) : Prop := sorry

/-- Theorem stating that given an angle and a circle, there exists an inscribed circle tangent to the given circle -/
theorem inscribed_tangent_circle_exists (a : Angle) (c : Circle) :
  ∃ (inscribed_circle : Circle), isInscribed inscribed_circle a ∧ isTangent inscribed_circle c := by
  sorry

end NUMINAMATH_CALUDE_inscribed_tangent_circle_exists_l2492_249287


namespace NUMINAMATH_CALUDE_magician_earnings_l2492_249224

-- Define the problem parameters
def initial_decks : ℕ := 20
def final_decks : ℕ := 5
def full_price : ℚ := 7
def discount_percentage : ℚ := 20 / 100

-- Define the number of decks sold at full price and discounted price
def full_price_sales : ℕ := 7
def discounted_sales : ℕ := 8

-- Calculate the discounted price
def discounted_price : ℚ := full_price * (1 - discount_percentage)

-- Calculate the total earnings
def total_earnings : ℚ := 
  (full_price_sales : ℚ) * full_price + 
  (discounted_sales : ℚ) * discounted_price

-- Theorem statement
theorem magician_earnings : 
  initial_decks - final_decks = full_price_sales + discounted_sales ∧ 
  total_earnings = 93.8 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_l2492_249224


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l2492_249214

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l2492_249214


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l2492_249291

/-- Given a point (a, b) and a line x + y = 0, find the symmetric point -/
def symmetricPoint (a b : ℝ) : ℝ × ℝ :=
  (-b, -a)

/-- The theorem states that the point symmetric to (2, 5) with respect to x + y = 0 is (-5, -2) -/
theorem symmetric_point_theorem :
  symmetricPoint 2 5 = (-5, -2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l2492_249291


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l2492_249243

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose n k

/-- The number of triangles with exactly one side coinciding with a side of the decagon -/
def one_side_triangles : ℕ := n * (n - 4)

/-- The number of triangles with two sides coinciding with sides of the decagon 
    (i.e., formed by three consecutive vertices) -/
def two_side_triangles : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side 
    coinciding with a side of the decagon) -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of a randomly chosen triangle having at least one side 
    that is also a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l2492_249243


namespace NUMINAMATH_CALUDE_quadratic_inequality_three_integer_solutions_l2492_249241

theorem quadratic_inequality_three_integer_solutions (α : ℝ) : 
  (∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ (w : ℤ), 2 * (w : ℝ)^2 - 17 * (w : ℝ) + α ≤ 0 ↔ w = x ∨ w = y ∨ w = z)) →
  -33 ≤ α ∧ α < -30 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_three_integer_solutions_l2492_249241


namespace NUMINAMATH_CALUDE_first_year_after_2000_with_digit_sum_15_l2492_249278

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isAfter2000 (year : ℕ) : Prop :=
  year > 2000

theorem first_year_after_2000_with_digit_sum_15 :
  ∀ year : ℕ, isAfter2000 year → sumOfDigits year = 15 → year ≥ 2049 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2000_with_digit_sum_15_l2492_249278


namespace NUMINAMATH_CALUDE_farther_from_theorem_l2492_249234

theorem farther_from_theorem :
  -- Part 1
  ∀ x : ℝ, |x^2 - 1| > 1 ↔ x < -Real.sqrt 2 ∨ x > Real.sqrt 2

  -- Part 2
  ∧ ∀ a b : ℝ, a > 0 → b > 0 → a ≠ b →
    |a^3 + b^3 - (a^2*b + a*b^2)| > |2*a*b*Real.sqrt (a*b) - (a^2*b + a*b^2)| :=
by sorry

end NUMINAMATH_CALUDE_farther_from_theorem_l2492_249234


namespace NUMINAMATH_CALUDE_unique_solution_equation_l2492_249240

theorem unique_solution_equation : ∃! x : ℝ, 3 * x + 3 * 12 + 3 * 13 + 11 = 134 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l2492_249240


namespace NUMINAMATH_CALUDE_davids_math_marks_l2492_249239

theorem davids_math_marks
  (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℚ)
  (h1 : english = 45)
  (h2 : physics = 52)
  (h3 : chemistry = 47)
  (h4 : biology = 55)
  (h5 : average = 46.8)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 35 := by
  sorry

end NUMINAMATH_CALUDE_davids_math_marks_l2492_249239


namespace NUMINAMATH_CALUDE_phone_bill_increase_l2492_249283

theorem phone_bill_increase (original_bill : ℝ) (increase_percent : ℝ) (months : ℕ) : 
  original_bill = 50 ∧ 
  increase_percent = 10 ∧ 
  months = 12 → 
  original_bill * (1 + increase_percent / 100) * months = 660 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_increase_l2492_249283


namespace NUMINAMATH_CALUDE_sqrt_pattern_main_problem_l2492_249215

theorem sqrt_pattern (n : ℕ) (h : n > 0) :
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / n - 1 / (n + 1) :=
sorry

theorem main_problem :
  Real.sqrt (50 / 49 + 1 / 64) = 1 + 1 / 56 :=
sorry

end NUMINAMATH_CALUDE_sqrt_pattern_main_problem_l2492_249215


namespace NUMINAMATH_CALUDE_wendy_distance_difference_l2492_249288

theorem wendy_distance_difference (ran walked : ℝ) 
  (h1 : ran = 19.83) (h2 : walked = 9.17) : 
  ran - walked = 10.66 := by sorry

end NUMINAMATH_CALUDE_wendy_distance_difference_l2492_249288


namespace NUMINAMATH_CALUDE_escalator_time_l2492_249267

/-- Time taken to cover the length of an escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (length : ℝ) 
  (h1 : escalator_speed = 11)
  (h2 : person_speed = 3)
  (h3 : length = 126) :
  length / (escalator_speed + person_speed) = 9 := by
sorry

end NUMINAMATH_CALUDE_escalator_time_l2492_249267


namespace NUMINAMATH_CALUDE_original_price_from_reduced_l2492_249222

/-- Given a shirt with a reduced price that is 25% of its original price,
    prove that if the reduced price is $6, then the original price was $24. -/
theorem original_price_from_reduced (reduced_price : ℝ) (original_price : ℝ) : 
  reduced_price = 6 → reduced_price = 0.25 * original_price → original_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_original_price_from_reduced_l2492_249222


namespace NUMINAMATH_CALUDE_smallest_period_scaled_l2492_249204

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x - p) = f x

theorem smallest_period_scaled (f : ℝ → ℝ) (h : is_periodic f 30) :
  ∃ a : ℝ, a > 0 ∧ (∀ x, f ((x - a) / 6) = f (x / 6)) ∧
  ∀ b, b > 0 → (∀ x, f ((x - b) / 6) = f (x / 6)) → a ≤ b :=
sorry

end NUMINAMATH_CALUDE_smallest_period_scaled_l2492_249204


namespace NUMINAMATH_CALUDE_distance_to_incenter_value_l2492_249259

/-- Represents a right isosceles triangle ABC with incenter I -/
structure RightIsoscelesTriangle where
  -- Length of side AB
  side_length : ℝ
  -- Incenter of the triangle
  incenter : ℝ × ℝ

/-- The distance from vertex A to the incenter I in a right isosceles triangle -/
def distance_to_incenter (t : RightIsoscelesTriangle) : ℝ :=
  -- Define the distance calculation here
  sorry

/-- Theorem: In a right isosceles triangle ABC with AB = 6√2, 
    the distance AI from vertex A to the incenter I is 6 - 3√2 -/
theorem distance_to_incenter_value :
  ∀ (t : RightIsoscelesTriangle),
  t.side_length = 6 * Real.sqrt 2 →
  distance_to_incenter t = 6 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_incenter_value_l2492_249259


namespace NUMINAMATH_CALUDE_abs_x_minus_four_plus_x_l2492_249250

theorem abs_x_minus_four_plus_x (x : ℝ) (h : |x - 3| + x - 3 = 0) : |x - 4| + x = 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_four_plus_x_l2492_249250


namespace NUMINAMATH_CALUDE_green_marbles_fraction_l2492_249203

theorem green_marbles_fraction (total : ℚ) (h1 : total > 0) : 
  let blue : ℚ := 2/3 * total
  let red : ℚ := 1/6 * total
  let green : ℚ := total - blue - red
  let new_total : ℚ := total + blue
  (green / new_total) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_green_marbles_fraction_l2492_249203


namespace NUMINAMATH_CALUDE_shopkeeper_additional_cards_l2492_249292

/-- The number of cards in a standard deck -/
def standard_deck : ℕ := 52

/-- The number of complete decks the shopkeeper has -/
def complete_decks : ℕ := 6

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 319

/-- The number of additional cards the shopkeeper has -/
def additional_cards : ℕ := total_cards - (complete_decks * standard_deck)

theorem shopkeeper_additional_cards : additional_cards = 7 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_additional_cards_l2492_249292


namespace NUMINAMATH_CALUDE_train_length_l2492_249269

/-- The length of a train given its speed and time to cross a stationary observer -/
theorem train_length (speed_kmh : ℝ) (time_seconds : ℝ) : 
  speed_kmh = 48 → time_seconds = 12 → speed_kmh * (5/18) * time_seconds = 480 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2492_249269


namespace NUMINAMATH_CALUDE_xy_value_l2492_249208

theorem xy_value (x y : ℕ+) (h1 : x + y = 36) (h2 : 4 * x * y + 12 * x = 5 * y + 390) : x * y = 252 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2492_249208


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2492_249237

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) →  -- arithmetic sequence sum formula
  (S 6 / S 3 = 4) →                                         -- given condition
  (S 9 / S 6 = 9 / 4) :=                                    -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2492_249237


namespace NUMINAMATH_CALUDE_race_heartbeats_l2492_249255

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

theorem race_heartbeats :
  total_heartbeats 160 6 30 = 28800 := by
  sorry

end NUMINAMATH_CALUDE_race_heartbeats_l2492_249255


namespace NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l2492_249236

-- Define the function for the nth odd positive integer
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

-- Theorem statement
theorem hundredthOddPositiveInteger : nthOddPositiveInteger 100 = 199 := by
  sorry

end NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l2492_249236


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l2492_249274

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  (a / b = d / e) ∧ (b ≠ 0) ∧ (e ≠ 0)

theorem parallel_lines_condition (a : ℝ) :
  (a = 1 → are_parallel a 2 (-1) 1 (a + 1) (-4)) ∧
  (∃ b : ℝ, b ≠ 1 ∧ are_parallel b 2 (-1) 1 (b + 1) (-4)) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l2492_249274


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2492_249252

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2492_249252


namespace NUMINAMATH_CALUDE_square_root_of_25_l2492_249229

theorem square_root_of_25 : ∃ (x y : ℝ), x^2 = 25 ∧ y^2 = 25 ∧ x = 5 ∧ y = -5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_25_l2492_249229
